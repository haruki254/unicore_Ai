//+------------------------------------------------------------------+
//|                     MarketStructureVisualizer.mq5                |
//|            Market Structure EA — Visual + Trade Exit Engine      |
//|     Detects HH, HL, LH, LL and auto-closes trades on confirm    |
//|                                                                    |
//|  v5.01 — FIX: CloseTradesInProfit() now sets g_trade's magic to  |
//|  each position's own magic number before closing it, instead of  |
//|  the hard-coded 0 from OnInit(). Previously every closing deal   |
//|  MSV generated was stamped with magic=0, which downstream EAs    |
//|  (e.g. unicore trainer.mq5's IsUnicoreMagic() check) reject —    |
//|  so trade closes were never reported to /trade/update and        |
//|  trade_history stayed empty even though trades were closing.     |
//+------------------------------------------------------------------+
#property copyright   "Market Structure Visualizer"
#property link        ""
#property version     "5.01"
#property description "Visualizes HH/HL/LH/LL + closes trades on swing confirmation AND level breach."

#include <Trade/Trade.mqh>

//============================================================
//  INPUT PARAMETERS
//============================================================

input group "=== Swing Detection ==="
input int    InpSwingStrength     = 5;
input int    InpMaxHistory        = 100;

input group "=== Display Toggles ==="
input bool   InpShowLabels        = true;
input bool   InpShowLines         = true;
input bool   InpShowBOS           = true;
input bool   InpShowCHOCH         = true;

input group "=== Colors ==="
input color  InpBullishColor      = clrLime;
input color  InpBearishColor      = clrRed;
input color  InpLineColor         = clrDodgerBlue;
input color  InpBOSColor          = clrGold;
input color  InpCHOCHColor        = clrMagenta;

input group "=== Label Style ==="
input int    InpLabelFontSize     = 9;
input string InpLabelFont         = "Consolas";
input int    InpLineWidth         = 1;
input ENUM_LINE_STYLE InpLineStyle = STYLE_SOLID;

input group "=== Alerts ==="
input bool   InpAlertNewSwing     = false;
input bool   InpAlertBOS          = false;
input bool   InpAlertCHOCH        = false;

input group "=== Multi-Timeframe Overlay ==="
input bool   InpMTFEnabled        = false;
input ENUM_TIMEFRAMES InpMTFPeriod = PERIOD_H1;
input color  InpMTFColor          = clrOrange;

input group "=== Structure-Based Trade Exit ==="
input bool   InpEnableAutoClose   = false;
input bool   InpCloseOnHL         = true;
input bool   InpCloseOnHH         = true;
input bool   InpCloseOnLH         = true;
input bool   InpCloseOnLL         = true;
input double InpMinProfitToClose  = 0.0;
input int    InpMagicFilter       = -1;

input group "=== Level Breach Exit ==="
input bool   InpEnableBreachExit  = true;
input bool   InpBreachOnHL        = true;
input bool   InpBreachOnLL        = true;
input bool   InpBreachOnLH        = true;
input bool   InpBreachOnHH        = true;
input bool   InpAlertBreach       = true;

//============================================================
//  INTERNAL ENUMS & STRUCTURES
//============================================================

enum ENUM_SWING_TYPE
{
   SWING_NONE = 0,
   SWING_HIGH,
   SWING_LOW
};

enum ENUM_STRUCT_LABEL
{
   LABEL_NONE = 0,
   LABEL_HH,
   LABEL_HL,
   LABEL_LH,
   LABEL_LL
};

struct SwingPoint
{
   datetime          time;
   double            price;
   ENUM_SWING_TYPE   swingType;
   ENUM_STRUCT_LABEL label;
   int               barIndex;
   bool              isBOS;
   bool              isCHOCH;
   bool              exitFired;
   bool              breachFired;
};

//============================================================
//  GLOBALS
//============================================================

SwingPoint g_swings[];
SwingPoint g_mtfSwings[];
int        g_swingCount    = 0;
int        g_mtfSwingCount = 0;

datetime   g_lastExitTime   = 0;
datetime   g_lastBreachTime = 0;

// Tracks previous Bid to detect actual crossings (not just "price is above/below").
// On startup this is 0 so the first tick is skipped, preventing a false
// initialization sweep that would mark every historical level as already breached.
double     g_prevBid        = 0;

string     g_prefix        = "MSV_";
datetime   g_lastBarTime   = 0;
bool       g_isInit        = false;

CTrade     g_trade;

//============================================================
//  UTILITY
//============================================================

string ObjName(string tag, int idx, bool isMTF = false)
{
   return g_prefix + (isMTF ? "MTF_" : "") + tag + "_" + IntegerToString(idx);
}

string LabelToString(ENUM_STRUCT_LABEL lbl)
{
   switch(lbl)
   {
      case LABEL_HH: return "HH";
      case LABEL_HL: return "HL";
      case LABEL_LH: return "LH";
      case LABEL_LL: return "LL";
      default:       return "??";
   }
}

//============================================================
//  DELETE CHART OBJECTS
//============================================================

void DeleteAllObjects(bool mtfOnly = false)
{
   string needle = mtfOnly ? g_prefix + "MTF_" : g_prefix;
   int total = ObjectsTotal(0, -1, -1);
   for(int i = total - 1; i >= 0; i--)
   {
      string name = ObjectName(0, i, -1, -1);
      if(StringFind(name, needle) == 0)
         ObjectDelete(0, name);
   }
}

//============================================================
//  ON INIT
//============================================================

int OnInit()
{
   ArrayResize(g_swings,    0);
   ArrayResize(g_mtfSwings, 0);
   g_swingCount     = 0;
   g_mtfSwingCount  = 0;
   g_lastBarTime    = 0;
   g_lastExitTime   = 0;
   g_lastBreachTime = 0;
   g_prevBid        = 0;   // reset so first tick seeds the price without firing

   if(InpSwingStrength < 1)
   {
      Alert("MSV: SwingStrength must be >= 1");
      return INIT_PARAMETERS_INCORRECT;
   }

   g_trade.SetExpertMagicNumber(0);
   g_trade.SetDeviationInPoints(10);

   EventSetTimer(1);

   FullRecalculate(false);
   if(InpMTFEnabled) FullRecalculate(true);

   g_isInit = true;
   Print("MSV v5.00 initialized | SwingStrength=", InpSwingStrength,
         " | AutoClose=", InpEnableAutoClose,
         " | MagicFilter=", InpMagicFilter);
   return INIT_SUCCEEDED;
}

//============================================================
//  ON DEINIT
//============================================================

void OnDeinit(const int reason)
{
   EventKillTimer();
   DeleteAllObjects();
   ChartRedraw();
}

//============================================================
//  ON TICK
//============================================================

void OnTick()
{
   if(g_isInit && InpEnableBreachExit)
      CheckLevelBreaches();

   datetime currentBar = iTime(_Symbol, _Period, 0);
   if(currentBar == g_lastBarTime) return;
   g_lastBarTime = currentBar;

   IncrementalUpdate(false);
   if(InpMTFEnabled) IncrementalUpdate(true);
   ChartRedraw();
}

//============================================================
//  ON TIMER
//============================================================

void OnTimer()
{
   if(!g_isInit) return;
   datetime currentBar = iTime(_Symbol, _Period, 0);
   if(currentBar != g_lastBarTime)
   {
      g_lastBarTime = currentBar;
      IncrementalUpdate(false);
      if(InpMTFEnabled) IncrementalUpdate(true);
      ChartRedraw();
   }
}

//============================================================
//  FULL RECALCULATE
//============================================================

void FullRecalculate(bool isMTF)
{
   ENUM_TIMEFRAMES tf     = isMTF ? InpMTFPeriod : _Period;
   string          sym    = _Symbol;
   int             bars   = Bars(sym, tf);
   int             left   = InpSwingStrength;
   int             right  = InpSwingStrength;
   int             needed = left + right + 1;

   if(bars < needed + 2)
   {
      Print("MSV: Not enough bars on ", EnumToString(tf));
      return;
   }

   if(isMTF)
   {
      DeleteAllObjects(true);
      ArrayResize(g_mtfSwings, 0);
      g_mtfSwingCount = 0;
   }
   else
   {
      DeleteAllObjects(false);
      ArrayResize(g_swings, 0);
      g_swingCount = 0;
   }

   SwingPoint rawSwings[];
   int rawCount = 0;

   int scanEnd = bars - needed;
   if(scanEnd < 1) scanEnd = 1;

   for(int i = scanEnd; i >= right; i--)
   {
      double   hi          = iHigh(sym, tf, i);
      double   lo          = iLow(sym, tf, i);
      datetime t           = iTime(sym, tf, i);
      bool     isSwingHigh = true;
      bool     isSwingLow  = true;

      for(int k = 1; k <= left; k++)
         if(iHigh(sym, tf, i + k) >= hi) { isSwingHigh = false; break; }
      if(isSwingHigh)
         for(int k = 1; k <= right; k++)
            if(iHigh(sym, tf, i - k) >= hi) { isSwingHigh = false; break; }

      for(int k = 1; k <= left; k++)
         if(iLow(sym, tf, i + k) <= lo) { isSwingLow = false; break; }
      if(isSwingLow)
         for(int k = 1; k <= right; k++)
            if(iLow(sym, tf, i - k) <= lo) { isSwingLow = false; break; }

      if(isSwingHigh)
      {
         ArrayResize(rawSwings, rawCount + 1);
         rawSwings[rawCount].time        = t;
         rawSwings[rawCount].price       = hi;
         rawSwings[rawCount].swingType   = SWING_HIGH;
         rawSwings[rawCount].barIndex    = i;
         rawSwings[rawCount].label       = LABEL_NONE;
         rawSwings[rawCount].isBOS       = false;
         rawSwings[rawCount].isCHOCH     = false;
         rawSwings[rawCount].exitFired   = (t <= g_lastExitTime);
         rawSwings[rawCount].breachFired = (t <= g_lastExitTime);
         rawCount++;
      }
      if(isSwingLow)
      {
         ArrayResize(rawSwings, rawCount + 1);
         rawSwings[rawCount].time        = t;
         rawSwings[rawCount].price       = lo;
         rawSwings[rawCount].swingType   = SWING_LOW;
         rawSwings[rawCount].barIndex    = i;
         rawSwings[rawCount].label       = LABEL_NONE;
         rawSwings[rawCount].isBOS       = false;
         rawSwings[rawCount].isCHOCH     = false;
         rawSwings[rawCount].exitFired   = (t <= g_lastExitTime);
         rawSwings[rawCount].breachFired = (t <= g_lastExitTime);
         rawCount++;
      }
   }

   // Sort ascending by time
   for(int i = 1; i < rawCount; i++)
   {
      SwingPoint key = rawSwings[i];
      int j = i - 1;
      while(j >= 0 && rawSwings[j].time > key.time)
      {
         rawSwings[j + 1] = rawSwings[j];
         j--;
      }
      rawSwings[j + 1] = key;
   }

   ClassifySwings(rawSwings, rawCount);

   int startIdx = (rawCount > InpMaxHistory) ? rawCount - InpMaxHistory : 0;

   if(isMTF)
   {
      ArrayResize(g_mtfSwings, rawCount - startIdx);
      for(int i = startIdx; i < rawCount; i++)
         g_mtfSwings[g_mtfSwingCount++] = rawSwings[i];
   }
   else
   {
      ArrayResize(g_swings, rawCount - startIdx);
      for(int i = startIdx; i < rawCount; i++)
      {
         rawSwings[i].breachFired = (rawSwings[i].time <= g_lastBreachTime);
         g_swings[g_swingCount++] = rawSwings[i];
      }
   }

   DrawAll(isMTF);
}

//============================================================
//  INCREMENTAL UPDATE
//============================================================

void IncrementalUpdate(bool isMTF)
{
   ENUM_TIMEFRAMES tf    = isMTF ? InpMTFPeriod : _Period;
   string          sym   = _Symbol;
   int             left  = InpSwingStrength;
   int             right = InpSwingStrength;
   int             bars  = Bars(sym, tf);

   if(bars < left + right + 3) return;

   bool changed = false;

   for(int i = right; i <= right + 2 && i < bars - left; i++)
   {
      double   hi = iHigh(sym, tf, i);
      double   lo = iLow(sym, tf, i);
      datetime t  = iTime(sym, tf, i);

      bool isSwingHigh = true;
      bool isSwingLow  = true;

      for(int k = 1; k <= left; k++)
         if(iHigh(sym, tf, i + k) >= hi) { isSwingHigh = false; break; }
      if(isSwingHigh)
         for(int k = 1; k <= right; k++)
            if(iHigh(sym, tf, i - k) >= hi) { isSwingHigh = false; break; }

      for(int k = 1; k <= left; k++)
         if(iLow(sym, tf, i + k) <= lo) { isSwingLow = false; break; }
      if(isSwingLow)
         for(int k = 1; k <= right; k++)
            if(iLow(sym, tf, i - k) <= lo) { isSwingLow = false; break; }

      bool hiExists = false, loExists = false;
      if(isMTF)
      {
         for(int j = 0; j < g_mtfSwingCount; j++)
         {
            if(g_mtfSwings[j].time == t)
            {
               if(g_mtfSwings[j].swingType == SWING_HIGH) hiExists = true;
               if(g_mtfSwings[j].swingType == SWING_LOW)  loExists = true;
            }
         }
      }
      else
      {
         for(int j = 0; j < g_swingCount; j++)
         {
            if(g_swings[j].time == t)
            {
               if(g_swings[j].swingType == SWING_HIGH) hiExists = true;
               if(g_swings[j].swingType == SWING_LOW)  loExists = true;
            }
         }
      }

      if(isSwingHigh && !hiExists) { changed = true; break; }
      if(isSwingLow  && !loExists) { changed = true; break; }
   }

   if(!changed) return;

   FullRecalculate(isMTF);

   if(!isMTF && InpEnableAutoClose)
      ProcessTradeExits();
}

//============================================================
//  CLASSIFY SWINGS
//============================================================

void ClassifySwings(SwingPoint &pts[], int count)
{
   if(count < 2) return;

   double            prevHigh      = -1;
   double            prevLow       = DBL_MAX;
   ENUM_STRUCT_LABEL lastHighLabel = LABEL_NONE;
   ENUM_STRUCT_LABEL lastLowLabel  = LABEL_NONE;
   bool              bullishBias   = false;
   bool              biasSet       = false;

   for(int i = 0; i < count; i++)
   {
      if(pts[i].swingType == SWING_HIGH)
      {
         if(prevHigh < 0)
         {
            pts[i].label  = LABEL_HH;
            prevHigh       = pts[i].price;
            lastHighLabel  = LABEL_HH;
            if(!biasSet) { bullishBias = true; biasSet = true; }
            continue;
         }

         ENUM_STRUCT_LABEL lbl;
         if(pts[i].price > prevHigh)
         {
            lbl = LABEL_HH;
            if(InpShowBOS   && lastLowLabel == LABEL_HL)                  pts[i].isBOS   = true;
            if(InpShowCHOCH && !bullishBias && lastHighLabel == LABEL_LH) pts[i].isCHOCH = true;
            bullishBias = true;
         }
         else
         {
            lbl = LABEL_LH;
            if(InpShowCHOCH && bullishBias && lastHighLabel == LABEL_HH)  pts[i].isCHOCH = true;
            bullishBias = false;
         }
         pts[i].label = lbl;
         lastHighLabel = lbl;
         prevHigh      = pts[i].price;
      }
      else
      {
         if(prevLow == DBL_MAX)
         {
            pts[i].label = LABEL_HL;
            prevLow       = pts[i].price;
            lastLowLabel  = LABEL_HL;
            continue;
         }

         ENUM_STRUCT_LABEL lbl;
         if(pts[i].price > prevLow)
         {
            lbl = LABEL_HL;
            if(InpShowBOS && lastHighLabel == LABEL_LH) pts[i].isBOS = true;
         }
         else
         {
            lbl = LABEL_LL;
            if(InpShowCHOCH && bullishBias && lastLowLabel == LABEL_HL) pts[i].isCHOCH = true;
         }
         pts[i].label = lbl;
         lastLowLabel  = lbl;
         prevLow       = pts[i].price;
      }
   }
}

//============================================================
//  PROCESS TRADE EXITS
//============================================================

void ProcessTradeExits()
{
   for(int i = g_swingCount - 1; i >= 0; i--)
   {
      SwingPoint sp = g_swings[i];

      if(sp.exitFired) break;
      if(sp.label == LABEL_NONE) continue;

      bool doCloseBuy  = false;
      bool doCloseSell = false;

      switch(sp.label)
      {
         case LABEL_HL: if(InpCloseOnHL) doCloseSell = true; break;
         case LABEL_HH: if(InpCloseOnHH) doCloseBuy  = true; break;
         case LABEL_LH: if(InpCloseOnLH) doCloseBuy  = true; break;
         case LABEL_LL: if(InpCloseOnLL) doCloseSell = true; break;
         default: break;
      }

      if(doCloseBuy || doCloseSell)
      {
         string reason = LabelToString(sp.label) + " confirmed @ " +
                         DoubleToString(sp.price, _Digits) +
                         " [" + TimeToString(sp.time, TIME_DATE | TIME_MINUTES) + "]";

         int closed = 0;
         if(doCloseBuy)  closed += CloseTradesInProfit(ORDER_TYPE_BUY,  reason);
         if(doCloseSell) closed += CloseTradesInProfit(ORDER_TYPE_SELL, reason);

         g_swings[i].exitFired = true;
         g_lastExitTime = sp.time;

         if(closed > 0 && InpAlertNewSwing)
            Alert("MSV | ", _Symbol, " | ", reason, " → ", closed, " trade(s) closed");

         Print("MSV | Exit fired: ", reason, " | Closed=", closed);
      }
      else
      {
         g_swings[i].exitFired = true;
         g_lastExitTime = sp.time;
      }

      break;
   }
}

//============================================================
//  CHECK LEVEL BREACHES
//
//  Runs on every tick. Fires only when price CROSSES a level
//  (transitions from one side to the other), not merely when
//  price happens to be above or below it.
//
//  On the very first tick g_prevBid == 0 so the function
//  seeds the previous price and returns — this prevents the
//  initialization sweep that previously marked every
//  historical level as already breached before any trades
//  were open, causing all real breaches to be silently skipped.
//
//  Crossing DOWN through any swing low  (HL / LL) → close ALL in profit
//  Crossing UP   through any swing high (LH / HH) → close ALL in profit
//============================================================

void CheckLevelBreaches()
{
   if(!InpEnableBreachExit) return;
   if(g_swingCount == 0)    return;

   double   currentBid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   datetime now        = TimeCurrent();

   // First tick after init: seed prevBid without checking anything.
   // This is the key guard against false initialization sweeps.
   if(g_prevBid == 0)
   {
      g_prevBid = currentBid;
      return;
   }

   double prevBid = g_prevBid;   // local snapshot for this tick's comparisons
   g_prevBid      = currentBid;  // update immediately so the next tick uses fresh data

   // No price movement this tick — nothing to check
   if(currentBid == prevBid) return;

   for(int i = 0; i < g_swingCount; i++)
   {
      if(g_swings[i].breachFired)         continue;
      if(g_swings[i].label == LABEL_NONE) continue;
      if(g_swings[i].time >= now)         continue;

      double lvl = g_swings[i].price;

      // A crossing requires price to have been on the "safe" side last tick
      // and now be on the "breached" side this tick.
      //
      // Swing LOW  (HL / LL): breach = price crosses DOWN through the level
      //   previous tick was AT or ABOVE the level; current tick is BELOW it
      bool bearishCross = (g_swings[i].swingType == SWING_LOW)
                          && (prevBid >= lvl)
                          && (currentBid < lvl);

      // Swing HIGH (LH / HH): breach = price crosses UP through the level
      //   previous tick was AT or BELOW the level; current tick is ABOVE it
      bool bullishCross = (g_swings[i].swingType == SWING_HIGH)
                          && (prevBid <= lvl)
                          && (currentBid > lvl);

      if(!bearishCross && !bullishCross) continue;

      string dir    = bearishCross ? "crossed DOWN through" : "crossed UP through";
      string reason = "Price " + dir + " " + LabelToString(g_swings[i].label)
                      + " @ " + DoubleToString(lvl, _Digits)
                      + " | prevBid=" + DoubleToString(prevBid, _Digits)
                      + " | Bid=" + DoubleToString(currentBid, _Digits);

      // Close ALL profitable trades — both BUY and SELL
      int closed = 0;
      closed += CloseTradesInProfit(ORDER_TYPE_BUY,  reason);
      closed += CloseTradesInProfit(ORDER_TYPE_SELL, reason);

      g_swings[i].breachFired = true;
      g_lastBreachTime        = g_swings[i].time;

      DrawBreachMarker(g_swings[i], i, iTime(_Symbol, _Period, 0), currentBid);

      Print("MSV | BREACH: ", reason, " | Closed=", closed,
            " | Time=", TimeToString(now, TIME_DATE|TIME_MINUTES|TIME_SECONDS));

      if(InpAlertBreach)
         Alert("MSV | ", _Symbol, " ", EnumToString(_Period),
               " | BREACH: ", reason);
   }
}

//============================================================
//  DRAW BREACH MARKER
//============================================================

void DrawBreachMarker(const SwingPoint &sp, int spIdx,
                      datetime breachTime, double breachPrice)
{
   string name = ObjName("BRK", spIdx, false);
   if(ObjectFind(0, name) >= 0) ObjectDelete(0, name);

   bool isBearish = (sp.swingType == SWING_LOW);
   color clr      = isBearish ? InpBearishColor : InpBullishColor;

   ENUM_OBJECT arrowType = isBearish ? OBJ_ARROW_DOWN : OBJ_ARROW_UP;
   ObjectCreate(0, name, arrowType, 0, breachTime, breachPrice);
   ObjectSetInteger(0, name, OBJPROP_COLOR,      clr);
   ObjectSetInteger(0, name, OBJPROP_WIDTH,      2);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN,     true);

   string tname = ObjName("BRK_T", spIdx, false);
   if(ObjectFind(0, tname) >= 0) ObjectDelete(0, tname);

   string txt = LabelToString(sp.label) + " BREACH";
   ObjectCreate(0, tname, OBJ_TEXT, 0, breachTime, breachPrice);
   ObjectSetString(0,  tname, OBJPROP_TEXT,      txt);
   ObjectSetString(0,  tname, OBJPROP_FONT,      InpLabelFont);
   ObjectSetInteger(0, tname, OBJPROP_FONTSIZE,  InpLabelFontSize);
   ObjectSetInteger(0, tname, OBJPROP_COLOR,     clr);
   ObjectSetInteger(0, tname, OBJPROP_ANCHOR,    isBearish ? ANCHOR_UPPER : ANCHOR_LOWER);
   ObjectSetInteger(0, tname, OBJPROP_SELECTABLE,false);
   ObjectSetInteger(0, tname, OBJPROP_HIDDEN,    true);
}

//============================================================
//  CLOSE TRADES IN PROFIT
//============================================================

int CloseTradesInProfit(ENUM_ORDER_TYPE orderType, string reason)
{
   int closed = 0;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;

      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;

      ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      bool isBuy  = (posType == POSITION_TYPE_BUY);
      bool isSell = (posType == POSITION_TYPE_SELL);

      if(orderType == ORDER_TYPE_BUY  && !isBuy)  continue;
      if(orderType == ORDER_TYPE_SELL && !isSell) continue;

      if(InpMagicFilter != -1)
      {
         long magic = PositionGetInteger(POSITION_MAGIC);
         if(magic != (long)InpMagicFilter) continue;
      }

      double profit = PositionGetDouble(POSITION_PROFIT);
      if(profit <= InpMinProfitToClose) continue;

      // ── FIX (v5.01): preserve the position's own magic on the closing deal ──
      // g_trade's magic was hard-set to 0 in OnInit(), which meant every
      // closing deal MSV generated carried magic=0 regardless of what magic
      // number the position was actually opened with. Downstream consumers
      // (e.g. unicore trainer.mq5's OnTradeTransaction -> IsUnicoreMagic())
      // filter on the CLOSING deal's magic, so they silently ignored every
      // close MSV made and never reported it to /trade/update.
      long ownMagic = PositionGetInteger(POSITION_MAGIC);
      g_trade.SetExpertMagicNumber(ownMagic);

      if(g_trade.PositionClose(ticket))
      {
         Print("MSV | Closed #", ticket, " (", (isBuy ? "BUY" : "SELL"), ")",
               " Magic=", ownMagic,
               " Profit=", DoubleToString(profit, 2),
               " Reason: ", reason);
         closed++;
      }
      else
      {
         Print("MSV | Failed to close #", ticket,
               " Magic=", ownMagic,
               " Error=", GetLastError(),
               " Reason: ", reason);
      }
   }

   return closed;
}

//============================================================
//  DRAW ALL
//============================================================

void DrawAll(bool isMTF)
{
   int cnt = isMTF ? g_mtfSwingCount : g_swingCount;
   if(cnt == 0) return;

   if(InpShowLines)
   {
      for(int i = 1; i < cnt; i++)
      {
         if(isMTF) DrawLine(g_mtfSwings[i-1], g_mtfSwings[i], i, true);
         else       DrawLine(g_swings[i-1],    g_swings[i],    i, false);
      }
   }

   for(int i = 0; i < cnt; i++)
   {
      if(isMTF)
      {
         if(InpShowLabels) DrawLabel(g_mtfSwings[i], i, true);
         if(g_mtfSwings[i].isBOS   && InpShowBOS)   DrawBOSCHOCH(g_mtfSwings[i], i, false, true);
         if(g_mtfSwings[i].isCHOCH && InpShowCHOCH) DrawBOSCHOCH(g_mtfSwings[i], i, true,  true);
      }
      else
      {
         if(InpShowLabels) DrawLabel(g_swings[i], i, false);
         if(g_swings[i].isBOS   && InpShowBOS)   DrawBOSCHOCH(g_swings[i], i, false, false);
         if(g_swings[i].isCHOCH && InpShowCHOCH) DrawBOSCHOCH(g_swings[i], i, true,  false);
      }
   }
}

//============================================================
//  DRAW LABEL
//============================================================

void DrawLabel(const SwingPoint &pt, int idx, bool isMTF)
{
   string lbl;
   color  clr;

   switch(pt.label)
   {
      case LABEL_HH: lbl = "HH"; clr = InpBullishColor; break;
      case LABEL_HL: lbl = "HL"; clr = InpBullishColor; break;
      case LABEL_LH: lbl = "LH"; clr = InpBearishColor; break;
      case LABEL_LL: lbl = "LL"; clr = InpBearishColor; break;
      default: return;
   }
   if(isMTF) clr = InpMTFColor;

   string tag = "";
   if(pt.isBOS)   tag = " [BOS]";
   if(pt.isCHOCH) tag = " [CHoCH]";

   string name = ObjName("LBL", idx, isMTF);
   if(ObjectFind(0, name) >= 0) ObjectDelete(0, name);

   ObjectCreate(0, name, OBJ_TEXT, 0, pt.time, pt.price);
   ObjectSetString(0,  name, OBJPROP_TEXT,     lbl + tag);
   ObjectSetString(0,  name, OBJPROP_FONT,     InpLabelFont);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, InpLabelFontSize);
   ObjectSetInteger(0, name, OBJPROP_COLOR,    clr);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN,   true);
   ObjectSetInteger(0, name, OBJPROP_ANCHOR,
                    pt.swingType == SWING_HIGH ? ANCHOR_LOWER : ANCHOR_UPPER);
}

//============================================================
//  DRAW ZIGZAG LINE
//============================================================

void DrawLine(const SwingPoint &from, const SwingPoint &to, int idx, bool isMTF)
{
   string name = ObjName("LN", idx, isMTF);
   color  clr  = isMTF ? InpMTFColor : InpLineColor;

   if(ObjectFind(0, name) >= 0) ObjectDelete(0, name);

   ObjectCreate(0, name, OBJ_TREND, 0, from.time, from.price, to.time, to.price);
   ObjectSetInteger(0, name, OBJPROP_COLOR,      clr);
   ObjectSetInteger(0, name, OBJPROP_WIDTH,      InpLineWidth);
   ObjectSetInteger(0, name, OBJPROP_STYLE,      InpLineStyle);
   ObjectSetInteger(0, name, OBJPROP_RAY_RIGHT,  false);
   ObjectSetInteger(0, name, OBJPROP_RAY_LEFT,   false);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN,     true);
}

//============================================================
//  DRAW BOS / CHOCH MARKER
//============================================================

void DrawBOSCHOCH(const SwingPoint &pt, int idx, bool isCHOCH, bool isMTF)
{
   string tag   = isCHOCH ? "CHOCH" : "BOS";
   string name  = ObjName(tag, idx, isMTF);
   color  clr   = isCHOCH ? InpCHOCHColor : InpBOSColor;

   if(ObjectFind(0, name) >= 0) ObjectDelete(0, name);

   datetime t1 = pt.time;
   datetime t2 = pt.time + PeriodSeconds(_Period) * 20;

   ObjectCreate(0, name, OBJ_TREND, 0, t1, pt.price, t2, pt.price);
   ObjectSetInteger(0, name, OBJPROP_COLOR,      clr);
   ObjectSetInteger(0, name, OBJPROP_WIDTH,      2);
   ObjectSetInteger(0, name, OBJPROP_STYLE,      STYLE_DASH);
   ObjectSetInteger(0, name, OBJPROP_RAY_RIGHT,  false);
   ObjectSetInteger(0, name, OBJPROP_RAY_LEFT,   false);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN,     true);

   string lname = ObjName(tag + "_T", idx, isMTF);
   if(ObjectFind(0, lname) >= 0) ObjectDelete(0, lname);

   ObjectCreate(0, lname, OBJ_TEXT, 0, t2, pt.price);
   ObjectSetString(0,  lname, OBJPROP_TEXT,      tag);
   ObjectSetString(0,  lname, OBJPROP_FONT,      InpLabelFont);
   ObjectSetInteger(0, lname, OBJPROP_FONTSIZE,  InpLabelFontSize - 1);
   ObjectSetInteger(0, lname, OBJPROP_COLOR,     clr);
   ObjectSetInteger(0, lname, OBJPROP_ANCHOR,    ANCHOR_LEFT);
   ObjectSetInteger(0, lname, OBJPROP_SELECTABLE,false);
   ObjectSetInteger(0, lname, OBJPROP_HIDDEN,    true);

   if(!isCHOCH && InpAlertBOS)
      Alert(_Symbol, " ", EnumToString(_Period), ": BOS @ ", DoubleToString(pt.price, _Digits));
   if(isCHOCH  && InpAlertCHOCH)
      Alert(_Symbol, " ", EnumToString(_Period), ": CHoCH @ ", DoubleToString(pt.price, _Digits));
}

//+------------------------------------------------------------------+
//|  END OF FILE                                                      |
//+------------------------------------------------------------------+