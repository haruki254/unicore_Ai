//+------------------------------------------------------------------+
//|                  Unicore AI-Only Trader                          |
//|    Trades on AI final_decision — toggle to override BLOCK        |
//+------------------------------------------------------------------+
#property copyright "Unicore AI"
#property link      ""
#property version   "1.03"
#property strict

#include <Trade\Trade.mqh>
CTrade trade;

// ==================== SETTINGS =====================
string   AI_API_URL     = "http://127.0.0.1:8000/predict";   // Change if needed
string   AI_API_KEY     = "@Youtube2017";
double   LotSize        = 0.01;                               // Fixed lot
int      MagicNumber    = 88888;
int      Slippage       = 30;
bool     AutoTrade      = true;

// Global variables
datetime lastCheckTime = 0;
int      PollingInterval = 5;   // seconds

// ==================== OVERRIDE / DASHBOARD STATE =====================
// AiOverride: when false (default), the EA respects the AI's full decision
// — including BLOCK — exactly like the original file. When true, BLOCK is
// ignored and the EA trades whatever trade_direction the AI last sent
// anyway. This is a toggle only — there is no manual Buy/Sell UI; the AI
// still decides direction, override just decides whether BLOCK is obeyed.
// IMPORTANT: this always starts false on load/reattach (see OnInit) —
// it is intentionally NOT restored from GlobalVariable/file storage, so a
// restart, recompile, or terminal restart always comes back up in
// AI-only (BLOCK-respecting) mode and must be flipped ON by hand each time.
bool     AiOverride     = false;

string   g_lastDecision  = "—";
string   g_lastDirection = "—";
datetime g_lastAiReply   = 0;
string   g_lastAiError   = "";

// ==================== UI OBJECT NAMES =====================
#define PFX "UnicoreDash_"
string OBJ_BG          = PFX + "bg";
string OBJ_TITLE       = PFX + "title";
string OBJ_STATUS_LBL  = PFX + "status_lbl";
string OBJ_STATUS_VAL  = PFX + "status_val";
string OBJ_DECISION_LBL= PFX + "decision_lbl";
string OBJ_DECISION_VAL= PFX + "decision_val";
string OBJ_DIR_LBL     = PFX + "dir_lbl";
string OBJ_DIR_VAL     = PFX + "dir_val";
string OBJ_TIME_LBL    = PFX + "time_lbl";
string OBJ_TIME_VAL    = PFX + "time_val";
string OBJ_OVERRIDE_BTN= PFX + "override_btn";

//+------------------------------------------------------------------+
//| Expert initialization                                            |
//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetDeviationInPoints(Slippage);

   // Override always resets to OFF on every load/reattach/recompile.
   // Do not read this from any persisted store (GlobalVariable, file, etc.)
   // — the whole point is that it must be turned on manually each time.
   AiOverride = false;

   Print("=== Unicore AI-Only Trader Started ===");
   Print("AI Server : ", AI_API_URL);
   Print("Lot Size  : ", LotSize);
   Print("AI Override: OFF (default on load)");

   CreateDashboard();
   UpdateDashboard();

   EventSetTimer(PollingInterval);
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();
   ObjectsDeleteAll(0, PFX);
}

//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
{
   if(!AutoTrade) return;

   datetime now = TimeCurrent();
   if(now - lastCheckTime < PollingInterval) return;

   lastCheckTime = now;
   CheckForNewSignals();
}

//+------------------------------------------------------------------+
//| Chart event handler — dashboard toggle click                     |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
   if(id != CHARTEVENT_OBJECT_CLICK) return;

   if(sparam == OBJ_OVERRIDE_BTN)
   {
      AiOverride = !AiOverride;
      Print("AI Override toggled -> ", (AiOverride ? "ON (BLOCK ignored, AI direction trades anyway)" : "OFF (BLOCK respected)"));
      UpdateDashboard();
      ObjectSetInteger(0, OBJ_OVERRIDE_BTN, OBJPROP_STATE, false); // un-press the button visually
      ChartRedraw(0);
      return;
   }
}

//+------------------------------------------------------------------+
//| Main AI Request                                                  |
//+------------------------------------------------------------------+
void CheckForNewSignals()
{
   double bidPrice   = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double spreadPips = (double)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);

   string json = StringFormat(
      "{\"symbol\":\"%s\",\"ea_signal\":\"BUY\",\"price\":%.5f,\"spread_pips\":%.1f}",
      _Symbol, bidPrice, spreadPips
   );
   Print("[DEBUG] NOTE: ea_signal is hardcoded to \"BUY\" in this request regardless of actual market conditions — if your backend's null/BLOCK logic depends on ea_signal matching something specific, this is worth checking.");

   char data[], result[];
   StringToCharArray(json, data, 0, StringLen(json), CP_UTF8);
   // NOTE: passing an explicit length here means StringToCharArray copies
   // exactly that many bytes and does NOT append a trailing null — so unlike
   // some MQL5 examples online, no ArrayResize(...,-1) trim is needed (or wanted).

   string headers = "X-API-Key: " + AI_API_KEY + "\r\nContent-Type: application/json\r\n";
   string responseHeaders;

   Print("Sending JSON: ", json);

   int res = WebRequest("POST", AI_API_URL, headers, 15000, data, result, responseHeaders);

   if(res == 200)
   {
      string response = CharArrayToString(result);
      g_lastAiReply = TimeCurrent();
      g_lastAiError = "";
      Print("[DEBUG] Raw AI response (", StringLen(response), " chars): ", response);
      ProcessAIResponse(response);
   }
   else
   {
      g_lastAiError = "HTTP " + IntegerToString(res) + " / err " + IntegerToString(GetLastError());
      Print("AI Request Failed. Code: ", res, " | Error: ", GetLastError(), " | Body: ", CharArrayToString(result));
   }

   UpdateDashboard();
}

//+------------------------------------------------------------------+
//| Process AI Response                                              |
//+------------------------------------------------------------------+
void ProcessAIResponse(string json)
{
   string final_decision = ExtractJSONValue(json, "final_decision");
   string trade_dir      = ExtractJSONValue(json, "trade_direction");

   g_lastDecision  = (final_decision == "" ? "—" : final_decision);
   g_lastDirection = (trade_dir == "" ? "—" : trade_dir);

   Print("AI Decision: ", final_decision, " | Direction: ", trade_dir);

   // ── Diagnostic summary — classifies exactly what state trade_dir is in ──
   if(trade_dir == "")
      Print("[DEBUG] trade_direction is EMPTY STRING -> key \"trade_direction\" was not found in the JSON at all, OR the response body is empty/malformed. Check the [DEBUG] Raw AI response line above and the ExtractJSONValue trace above it.");
   else if(trade_dir == "null")
      Print("[DEBUG] trade_direction is literal 'null' -> the key EXISTS in the JSON but the API sent JSON null as its value (not \"BUY\"/\"SELL\"). This is a backend-side issue: your /predict endpoint is returning trade_direction: null for this request. Since this won't match \"BUY\" or \"SELL\", no trade will open — check what conditions cause your model/endpoint to emit null instead of a direction.");
   else if(trade_dir != "BUY" && trade_dir != "SELL")
      Print("[DEBUG] trade_direction is an UNEXPECTED VALUE -> '", trade_dir, "' (not empty, not null, not BUY, not SELL). Check for a typo/case mismatch between what the API sends and what this EA compares against.");
   else
      Print("[DEBUG] trade_direction parsed correctly as '", trade_dir, "'.");

   if(final_decision == "BLOCK")
   {
      if(AiOverride)
      {
         // Override ON: BLOCK is ignored on purpose. The AI still decides
         // direction — we trade trade_dir anyway instead of closing/staying flat.
         Print("AI said BLOCK but AI Override is ON — trading direction anyway: ", trade_dir);
         if(trade_dir == "BUY")
            OpenBuy();
         else if(trade_dir == "SELL")
            OpenSell();
         return;
      }

      // Override OFF (default) — original behavior, unchanged.
      CloseAllPositions();
      return;
   }

   if(trade_dir == "BUY")
      OpenBuy();
   else if(trade_dir == "SELL")
      OpenSell();
}

//+------------------------------------------------------------------+
//| Trade Helpers                                                    |
//+------------------------------------------------------------------+
void OpenBuy()
{
   if(PositionsTotal() == 0)
      trade.Buy(LotSize, _Symbol);
}

void OpenSell()
{
   if(PositionsTotal() == 0)
      trade.Sell(LotSize, _Symbol);
}

void CloseAllPositions()
{
   for(int i = PositionsTotal()-1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket > 0) trade.PositionClose(ticket);
   }
}

//+------------------------------------------------------------------+
//| JSON Helper                                                      |
//+------------------------------------------------------------------+
string ExtractJSONValue(string json, string key)
{
   string sk = "\"" + key + "\":";
   int sp = StringFind(json, sk);

   if(sp == -1)
   {
      Print("[DEBUG] ExtractJSONValue('", key, "'): key not found in JSON (StringFind returned -1). Searched for literal: ", sk);
      return "";
   }

   Print("[DEBUG] ExtractJSONValue('", key, "'): found key at index ", sp);

   sp += StringLen(sk);
   while(sp < StringLen(json) && (StringSubstr(json,sp,1)==" " || StringSubstr(json,sp,1)=="\t")) sp++;

   string nextChar = StringSubstr(json, sp, 1);
   Print("[DEBUG] ExtractJSONValue('", key, "'): after key+whitespace, next char at index ", sp, " is '", nextChar, "'");

   if(nextChar == "\"")  // string value
   {
      sp++;
      int ep = StringFind(json, "\"", sp);
      if(ep == -1)
      {
         Print("[DEBUG] ExtractJSONValue('", key, "'): opening quote found but no closing quote — malformed JSON or truncated response. Returning empty.");
         return "";
      }
      string val = StringSubstr(json, sp, ep-sp);
      Print("[DEBUG] ExtractJSONValue('", key, "'): parsed as QUOTED STRING -> '", val, "'");
      return val;
   }
   else  // number or boolean (or bare `null` — this parser does NOT special-case JSON null)
   {
      int ep = StringFind(json, ",", sp);
      if(ep == -1) ep = StringFind(json, "}", sp);
      if(ep == -1)
      {
         Print("[DEBUG] ExtractJSONValue('", key, "'): no ',' or '}' found after value — malformed JSON or truncated response. Returning empty.");
         return "";
      }
      string val = StringSubstr(json, sp, ep-sp);
      if(val == "null")
         Print("[DEBUG] ExtractJSONValue('", key, "'): parsed as UNQUOTED NULL (literal JSON null, not a missing key) -> '", val, "'. This will NOT equal \"BUY\"/\"SELL\" in comparisons — check why your API is sending null for this field.");
      else
         Print("[DEBUG] ExtractJSONValue('", key, "'): parsed as UNQUOTED VALUE (number/bool) -> '", val, "'");
      return val;
   }
}

//+------------------------------------------------------------------+
//| Dashboard — creation                                             |
//+------------------------------------------------------------------+
void CreateDashboard()
{
   int x = 20, y = 20;
   int panelW = 260, panelH = 145;

   CreateRect(OBJ_BG, x, y, panelW, panelH, C'20,20,20', clrSilver);
   CreateLabel(OBJ_TITLE, x+10, y+8, "UNICORE AI DASHBOARD", clrWhite, 10, true);

   CreateLabel(OBJ_STATUS_LBL,   x+10,  y+32, "AI Override:", clrSilver, 8, false);
   CreateLabel(OBJ_STATUS_VAL,   x+120, y+32, "OFF",          clrOrange, 8, true);

   CreateLabel(OBJ_DECISION_LBL, x+10,  y+50, "Last Decision:", clrSilver, 8, false);
   CreateLabel(OBJ_DECISION_VAL, x+120, y+50, "—",              clrWhite, 8, true);

   CreateLabel(OBJ_DIR_LBL,      x+10,  y+68, "Last Direction:", clrSilver, 8, false);
   CreateLabel(OBJ_DIR_VAL,      x+120, y+68, "—",               clrWhite, 8, true);

   CreateLabel(OBJ_TIME_LBL,     x+10,  y+86, "Last AI Reply:", clrSilver, 8, false);
   CreateLabel(OBJ_TIME_VAL,     x+120, y+86, "—",              clrWhite, 8, true);

   CreateButton(OBJ_OVERRIDE_BTN, x+10, y+110, 240, 26, "AI OVERRIDE: OFF", C'60,20,20', clrWhite);
}

//+------------------------------------------------------------------+
//| Dashboard — refresh values                                       |
//+------------------------------------------------------------------+
void UpdateDashboard()
{
   if(AiOverride)
   {
      ObjectSetString(0, OBJ_STATUS_VAL, OBJPROP_TEXT, "ON (BLOCK ignored)");
      ObjectSetInteger(0, OBJ_STATUS_VAL, OBJPROP_COLOR, clrLime);

      ObjectSetString(0, OBJ_OVERRIDE_BTN, OBJPROP_TEXT, "AI OVERRIDE: ON");
      ObjectSetInteger(0, OBJ_OVERRIDE_BTN, OBJPROP_BGCOLOR, C'20,60,20');
   }
   else
   {
      ObjectSetString(0, OBJ_STATUS_VAL, OBJPROP_TEXT, "OFF (BLOCK respected)");
      ObjectSetInteger(0, OBJ_STATUS_VAL, OBJPROP_COLOR, clrOrange);

      ObjectSetString(0, OBJ_OVERRIDE_BTN, OBJPROP_TEXT, "AI OVERRIDE: OFF");
      ObjectSetInteger(0, OBJ_OVERRIDE_BTN, OBJPROP_BGCOLOR, C'60,20,20');
   }

   ObjectSetString(0, OBJ_DECISION_VAL, OBJPROP_TEXT, g_lastDecision);
   ObjectSetString(0, OBJ_DIR_VAL,      OBJPROP_TEXT, g_lastDirection);

   if(g_lastAiError != "")
      ObjectSetString(0, OBJ_TIME_VAL, OBJPROP_TEXT, "ERR: " + g_lastAiError);
   else if(g_lastAiReply > 0)
      ObjectSetString(0, OBJ_TIME_VAL, OBJPROP_TEXT, TimeToString(g_lastAiReply, TIME_MINUTES|TIME_SECONDS));
   else
      ObjectSetString(0, OBJ_TIME_VAL, OBJPROP_TEXT, "—");

   ChartRedraw(0);
}

//+------------------------------------------------------------------+
//| Dashboard — UI primitives                                        |
//+------------------------------------------------------------------+
void CreateRect(string name, int x, int y, int w, int h, color bg, color border)
{
   ObjectCreate(0, name, OBJ_RECTANGLE_LABEL, 0, 0, 0);
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
   ObjectSetInteger(0, name, OBJPROP_XSIZE, w);
   ObjectSetInteger(0, name, OBJPROP_YSIZE, h);
   ObjectSetInteger(0, name, OBJPROP_BGCOLOR, bg);
   ObjectSetInteger(0, name, OBJPROP_BORDER_TYPE, BORDER_FLAT);
   ObjectSetInteger(0, name, OBJPROP_COLOR, border);
   ObjectSetInteger(0, name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
   ObjectSetInteger(0, name, OBJPROP_BACK, false);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
}

void CreateLabel(string name, int x, int y, string text, color clr, int fontSize, bool bold)
{
   ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
   ObjectSetInteger(0, name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
   ObjectSetString(0, name, OBJPROP_TEXT, text);
   ObjectSetString(0, name, OBJPROP_FONT, bold ? "Arial Bold" : "Arial");
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, fontSize);
   ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
}

void CreateButton(string name, int x, int y, int w, int h, string text, color bg, color txt)
{
   ObjectCreate(0, name, OBJ_BUTTON, 0, 0, 0);
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
   ObjectSetInteger(0, name, OBJPROP_XSIZE, w);
   ObjectSetInteger(0, name, OBJPROP_YSIZE, h);
   ObjectSetInteger(0, name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
   ObjectSetString(0, name, OBJPROP_TEXT, text);
   ObjectSetString(0, name, OBJPROP_FONT, "Arial Bold");
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, 8);
   ObjectSetInteger(0, name, OBJPROP_COLOR, txt);
   ObjectSetInteger(0, name, OBJPROP_BGCOLOR, bg);
   ObjectSetInteger(0, name, OBJPROP_BORDER_COLOR, clrBlack);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
   ObjectSetInteger(0, name, OBJPROP_STATE, false);
}