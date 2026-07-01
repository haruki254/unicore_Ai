//+------------------------------------------------------------------+
//|                                                    Unicore.mq5   |
//|         Unicore — XAUUSD | Multi-Strategy Executor               |
//|         v5.20: manual override for AI BLOCK decisions             |
//|           TI_OverrideBlock (input) — when true, a BLOCK verdict   |
//|           no longer stops the trade; it still logs/reports to     |
//|           the AI backend (ai_block_overridden=true) so blocked    |
//|           trades produce real outcomes to train/calibrate on.     |
//|           TI_OverrideDemoOnly (input, default true) — override    |
//|           only takes effect on demo accounts, ignored on live.    |
//|           Comment tag "_OV" + dashboard "Overrides: N" counter    |
//|           make overridden trades visible on-chart and in Journal. |
//|                                                                   |
//|         v5.19: dashboard now shows predict/patch/report status    |
//|           g_ti_predictStatus/Time — TI_CheckSignal() outcome      |
//|           g_ti_patchStatus/Time   — UpdateSignalStatus() outcome  |
//|           g_ti_reportStatus/Time  — TI_ReportTrade() outcome      |
//|           Each shows OK, FAIL http=<code> (err=<mql5 err>), or    |
//|           SKIPPED (AI disabled) — so a disabled/broken stage is   |
//|           visible on-chart instead of only in the Journal.        |
//|           UpdateSignalStatus() also now checks the PATCH result   |
//|           (previously fired-and-forgot it).                       |
//|                                                                   |
//|         v5.18: nuclear trade_history null-column fix                          |
//|                                                                   |
//|         v5.17 changes vs v5.16:                                   |
//|           TicketMeta stores openedAt (TimeCurrent() at open)      |
//|           StoreMeta() / TI_ReportTrade() extended with openedAt   |
//|           OnTradeTransaction() forwards opened_at to /trade/update|
//|                                                                   |
//|         v5.16 changes vs v5.15:                                   |
//|           TI_CheckSignal() now returns full TI result struct      |
//|           (snapshot_id, prediction_id, regime, quality etc.)      |
//|           OpenTrade() stores snapshot_id + prediction_id in a     |
//|           per-ticket lookup table (g_ticketMeta[])                |
//|           TI_ReportTrade() signature extended with snapshot_id,   |
//|           prediction_id, regime, session, lot_size,               |
//|           max_drawdown_pips — all forwarded to /trade/update      |
//|           OnTradeTransaction() looks up meta by position ticket   |
//|           and passes it to TI_ReportTrade()                       |
//|                                                                   |
//|         v5.15 (inherited):                                        |
//|           TI_CheckSignal() passes strategy as ea_id               |
//|           TI_ReportTrade() passes strategy + real wasFlipped      |
//|           OpenTrade() tracks flips, tags comment with _FL suffix  |
//|           OnTradeTransaction() decodes strategy from comment,     |
//|           detects _FL suffix, passes both to TI_ReportTrade       |
//|                                                                   |
//|         v5.14 (inherited):                                        |
//|           Per-strategy magic number routing                       |
//|           MSV gate removed (non-MSV signals always pass)          |
//+------------------------------------------------------------------+
#property copyright "Unicore"
#property link      "https://unicoregoldbot.tiiny.site"
#property version   "5.20"
#property strict

#include <Trade\Trade.mqh>

// ============================================================
// SUPABASE
// ============================================================
input string  Grp2               = "=== Supabase ===";
input string  SupabaseURL        = "https://dwitvurcslwyhiwybzki.supabase.co";
input string  SupabaseKey        = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImR3aXR2dXJjc2x3eWhpd3liemtpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjE3MzMxMTksImV4cCI6MjA3NzMwOTExOX0.lqXY9j8tAS9Zd_PhA-Mm73YHP5e1bH8fCSvqKv4WC6g";
input string  TableName          = "trade_signals";
input int     PollingInterval    = 3;

// ============================================================
// SYMBOL
// ============================================================
input string  Grp3               = "=== Symbol ===";
input string  SignalSymbol       = "XAUUSD";
input bool    TradeOnlyMySymbol  = true;

// ============================================================
// GENERAL
// ============================================================
input string  Grp4               = "=== General ===";
input bool    AutoTrade          = true;
input int     Slippage           = 30;
input bool    SendNotifications  = false;
input bool    DebugMode          = true;

// ============================================================
// LOOKBACK
// ============================================================
input string  Grp5               = "=== Lookback ===";
input int     LookbackHours      = 4;

// ============================================================
// STOP LOSS
// ============================================================
input string  Grp6               = "=== Stop Loss ===";
input double  FixedSLPoints      = 10.0;

// ============================================================
// LOT SIZING
// ============================================================
input string  Grp7               = "=== Lot Sizing ===";
input double  LotPerStep         = 0.02;
input double  EquityStep         = 500.0;

// ============================================================
// PYTHON LOT BRIDGE INPUTS
// ============================================================
input string  PY_Grp             = "=== Python Lot Bridge ===";
input bool    PY_Enabled         = true;
input int     PY_StaleSeconds    = 60;
input double  PY_FallbackMult    = 0.50;
input double  PY_MultFloor       = 0.25;
input double  PY_MultCeil        = 2.00;

// ============================================================
// TRADING INTELLIGENCE AI
// ============================================================
input string  TI_Grp             = "=== Trading Intelligence AI ===";
input bool    TI_Enabled         = true;
input string  TI_API_URL         = "http://127.0.0.1:8000";
input bool    TI_AllowFlip       = false;
input int     TI_Timeout_ms      = 5000;

// v5.20 — manual override for AI BLOCK decisions. Meant for training/
// calibrating the model on a demo account: trades execute regardless of
// a BLOCK verdict, but the verdict + real outcome still get reported to
// the AI backend (ai_block_overridden=true) so it has something to learn
// from instead of only ever seeing outcomes it already approved of.
input bool    TI_OverrideBlock       = false;  // Execute anyway when AI says BLOCK (training mode)
input bool    TI_OverrideDemoOnly    = true;   // Only let the override take effect on a DEMO account

// Hardcoded — must match API_KEY in your Python .env file.
const string  TI_API_KEY         = "@Youtube2017";

// ============================================================
// MAGIC NUMBER ROUTING  (v5.14)
// ============================================================
#define MAGIC_FRACTAL      77701
#define MAGIC_OB           77702
#define MAGIC_TRAP         77703
#define MAGIC_LIQUIDITY    77704
#define MAGIC_AMT          77705
#define MAGIC_LVN          77706
#define MAGIC_SMARTENTRY   77707
#define MAGIC_FVG          777701
#define MAGIC_UTBOT        88801
#define MAGIC_MSV          88888

ulong GetMagicForStrategy(const string strategy)
{
   if(strategy == "FRACTAL")                        return MAGIC_FRACTAL;
   if(strategy == "OB")                             return MAGIC_OB;
   if(strategy == "TRAP")                           return MAGIC_TRAP;
   if(strategy == "LIQ" || strategy == "LIQUIDITY") return MAGIC_LIQUIDITY;
   if(strategy == "AMT")                            return MAGIC_AMT;
   if(strategy == "LVN")                            return MAGIC_LVN;
   if(strategy == "SMARTENTRY")                     return MAGIC_SMARTENTRY;
   if(strategy == "FVG")                            return MAGIC_FVG;
   if(strategy == "UTBOT")                          return MAGIC_UTBOT;
   if(IsMSVStrategy(strategy))                      return MAGIC_MSV;
   Print("[MagicRouter] ⚠️ Unknown strategy '", strategy, "' — defaulting to MAGIC_MSV");
   return MAGIC_MSV;
}

bool IsUnicoreMagic(ulong magic)
{
   return (magic == MAGIC_FRACTAL     ||
           magic == MAGIC_OB          ||
           magic == MAGIC_TRAP        ||
           magic == MAGIC_LIQUIDITY   ||
           magic == MAGIC_AMT         ||
           magic == MAGIC_LVN         ||
           magic == MAGIC_SMARTENTRY  ||
           magic == MAGIC_FVG         ||
           magic == MAGIC_UTBOT       ||
           magic == MAGIC_MSV);
}

// ============================================================
// PYTHON BRIDGE — RUNTIME STATE
// ============================================================
double   g_py_multiplier     = 1.00;
string   g_py_mode           = "INIT";
string   g_py_regime         = "—";
double   g_py_health         = 50.0;
bool     g_py_stale          = true;
string   g_py_status         = "Waiting for bridge...";
bool     g_py_use_fixed_lot  = false;
double   g_py_fixed_lot      = 0.01;

#define PY_FILENAME "unicore_lot_params.json"

// ============================================================
// TRADING INTELLIGENCE — RUNTIME STATE
// ============================================================
string   g_ti_decision       = "—";
string   g_ti_regime         = "—";
double   g_ti_quality        = 0.0;
double   g_ti_buy_prob       = 0.0;
double   g_ti_sell_prob      = 0.0;
int      g_ti_blocks         = 0;
int      g_ti_allows         = 0;
string   g_ti_status         = "Waiting...";
int      g_ti_overrides      = 0;   // v5.20: BLOCK verdicts overridden and executed anyway

// v5.19: per-stage pipeline visibility (predict / patch / report)
string   g_ti_predictStatus  = "—";
datetime g_ti_predictTime    = 0;
string   g_ti_patchStatus    = "—";
datetime g_ti_patchTime      = 0;
string   g_ti_reportStatus   = "—";
datetime g_ti_reportTime     = 0;

// ============================================================
// v5.16 — TI RESULT STRUCT
// Holds everything returned by /predict so OpenTrade() can
// store snapshot_id + prediction_id against the ticket.
// ============================================================
struct TIResult
{
   string decision;       // ALLOW_BUY | ALLOW_SELL | FLIP_TO_BUY | FLIP_TO_SELL | BLOCK
   string snapshotId;     // UUID from /predict response
   string predictionId;   // UUID from /predict response
   string regime;
   double quality;
   double buyProb;
   double sellProb;
};

// ============================================================
// v5.16 — PER-TICKET METADATA TABLE
// v5.17 — extended with openedAt
// Stores snapshot_id + prediction_id + open timestamp keyed
// by MT5 position ticket so OnTradeTransaction() can look
// them up at close.  Max 200 concurrent positions.
// ============================================================
#define MAX_TICKET_META 200

struct TicketMeta
{
   ulong    ticket;
   string   snapshotId;
   string   predictionId;
   string   regime;
   double   lotSize;
   string   strategy;
   datetime openedAt;    // v5.17: stored at order open, sent to /trade/update
   string   openSession; // v5.18: session captured at open time, not close time
};

TicketMeta g_ticketMeta[MAX_TICKET_META];
int        g_ticketMetaCount = 0;

void StoreMeta(ulong ticket, string snapshotId, string predictionId,
               string regime, double lotSize, string strategy,
               datetime openedAt, string openSession)
{
   // Update if already exists
   for(int i = 0; i < g_ticketMetaCount; i++)
   {
      if(g_ticketMeta[i].ticket == ticket)
      {
         g_ticketMeta[i].snapshotId   = snapshotId;
         g_ticketMeta[i].predictionId = predictionId;
         g_ticketMeta[i].regime       = regime;
         g_ticketMeta[i].lotSize      = lotSize;
         g_ticketMeta[i].strategy     = strategy;
         g_ticketMeta[i].openedAt     = openedAt;
         g_ticketMeta[i].openSession  = openSession;
         return;
      }
   }
   // Add new
   if(g_ticketMetaCount >= MAX_TICKET_META)
   {
      // Evict oldest (index 0) by shifting left
      for(int i = 0; i < MAX_TICKET_META - 1; i++)
         g_ticketMeta[i] = g_ticketMeta[i + 1];
      g_ticketMetaCount = MAX_TICKET_META - 1;
   }
   g_ticketMeta[g_ticketMetaCount].ticket       = ticket;
   g_ticketMeta[g_ticketMetaCount].snapshotId   = snapshotId;
   g_ticketMeta[g_ticketMetaCount].predictionId = predictionId;
   g_ticketMeta[g_ticketMetaCount].regime       = regime;
   g_ticketMeta[g_ticketMetaCount].lotSize      = lotSize;
   g_ticketMeta[g_ticketMetaCount].strategy     = strategy;
   g_ticketMeta[g_ticketMetaCount].openedAt     = openedAt;
   g_ticketMeta[g_ticketMetaCount].openSession  = openSession;
   g_ticketMetaCount++;
}

bool GetMeta(ulong ticket, TicketMeta &out)
{
   for(int i = 0; i < g_ticketMetaCount; i++)
   {
      if(g_ticketMeta[i].ticket == ticket)
      {
         out = g_ticketMeta[i];
         return true;
      }
   }
   return false;
}

void RemoveMeta(ulong ticket)
{
   for(int i = 0; i < g_ticketMetaCount; i++)
   {
      if(g_ticketMeta[i].ticket == ticket)
      {
         for(int j = i; j < g_ticketMetaCount - 1; j++)
            g_ticketMeta[j] = g_ticketMeta[j + 1];
         g_ticketMetaCount--;
         return;
      }
   }
}

// ============================================================
// UNICORE STATE
// ============================================================
CTrade g_trade;
datetime g_lastCheckTime  = 0;
#define MAX_PROCESSED 50
string   g_processedIDs[MAX_PROCESSED];
int      g_processedCount = 0;
datetime g_attachTimeGMT  = 0;
datetime g_lastResetDate  = 0;
int      g_todayTradeCount= 0;
datetime g_lastPollTime   = 0;
datetime g_fvgLastWinTime = 0;
string   g_lastSignalID       = "None";
string   g_lastSignalAction   = "None";
string   g_lastSignalStrategy = "None";
string   g_lastSignalStatus   = "Waiting...";
string   g_lastError          = "";
string   g_lastVote           = "—";
int      g_totalWins      = 0; double g_totalWinProfit  = 0;
int      g_totalLosses    = 0; double g_totalLossAmount = 0;
double   g_netProfit      = 0;
double   g_loss_streak    = 0;
double   g_win_streak     = 0;

void DebugPrint(string msg){ if(DebugMode) Print("[DEBUG] ", msg); }
bool IsProcessed(string id){ for(int i=0;i<g_processedCount;i++) if(g_processedIDs[i]==id) return true; return false; }
void MarkProcessed(string id)
{
   if(IsProcessed(id)) return;
   if(g_processedCount >= MAX_PROCESSED)
   {
      for(int i=0;i<MAX_PROCESSED-1;i++) g_processedIDs[i]=g_processedIDs[i+1];
      g_processedCount = MAX_PROCESSED-1;
   }
   g_processedIDs[g_processedCount++] = id;
}

// ============================================================
// ACCOUNT HELPER  (v5.20)
// Used to gate TI_OverrideBlock to demo accounts only.
// ============================================================
bool IsDemoAccount()
{
   return (AccountInfoInteger(ACCOUNT_TRADE_MODE) == ACCOUNT_TRADE_MODE_DEMO);
}

// ============================================================
// SESSION HELPER  (v5.16)
// Determines current trading session from server hour (GMT).
// ============================================================
string GetCurrentSession()
{
   MqlDateTime dt;
   TimeToStruct(TimeGMT(), dt);
   int h = dt.hour;
   if(h >= 22 || h < 7)  return "asian";
   if(h >= 7  && h < 9)  return "overlap_asian_london";
   if(h >= 9  && h < 12) return "london";
   if(h >= 12 && h < 13) return "overlap_london_ny";
   if(h >= 13 && h < 17) return "new_york";
   if(h >= 17 && h < 22) return "off_hours";
   return "off_hours";
}

// ============================================================
// PYTHON LOT BRIDGE FUNCTIONS
// ============================================================
double PY_ParseDouble(string json, string key, double defaultVal = 0.0)
{
   string raw = ExtractJSONValue(json, key);
   if(raw == "" || raw == "null") return defaultVal;
   return StringToDouble(raw);
}

void PY_ReadLotParams()
{
   if(!PY_Enabled)
   {
      g_py_multiplier    = 1.0;
      g_py_use_fixed_lot = false;
      g_py_status        = "Bridge disabled — multiplier 1.00x";
      return;
   }

   int fh = FileOpen(PY_FILENAME, FILE_READ | FILE_TXT | FILE_ANSI);
   if(fh == INVALID_HANDLE)
   {
      g_py_stale  = true;
      g_py_status = "🔴 Bridge file not found — is mt5_lotbridge.py running?";
      if(DebugMode) Print("[PY_Bridge] File not found: ", PY_FILENAME);
      return;
   }

   string json = "";
   while(!FileIsEnding(fh))
      json += FileReadString(fh);
   FileClose(fh);

   if(StringLen(json) < 10)
   {
      g_py_stale  = true;
      g_py_status = "🔴 Bridge file is empty";
      return;
   }

   long mod_unix = (long)FileGetInteger(PY_FILENAME, FILE_MODIFY_DATE);
   int  age      = (int)(TimeCurrent() - (datetime)mod_unix);

   if(age > PY_StaleSeconds)
   {
      g_py_stale         = true;
      g_py_use_fixed_lot = false;
      g_py_status        = StringFormat("⚠️  Stale %ds — fallback %.2fx", age, PY_FallbackMult);
      if(DebugMode)
         Print("[PY_Bridge] Stale by ", age, "s — fallback ", PY_FallbackMult);
      return;
   }

   double mult   = PY_ParseDouble(json, "multiplier",   1.0);
   double health = PY_ParseDouble(json, "health_score", 50.0);
   string mode   = ExtractJSONValue(json, "mode");
   string regime = ExtractJSONValue(json, "regime");
   bool   useFixed  = (ExtractJSONValue(json, "use_fixed_lot") == "true");
   double fixedLot  = PY_ParseDouble(json, "fixed_lot", 0.01);

   mult = MathMax(PY_MultFloor, MathMin(PY_MultCeil, mult));

   g_py_multiplier    = mult;
   g_py_health        = health;
   g_py_mode          = (mode   != "") ? mode   : "UNKNOWN";
   g_py_regime        = (regime != "") ? regime : "UNKNOWN";
   g_py_use_fixed_lot = useFixed;
   g_py_fixed_lot     = (fixedLot > 0) ? fixedLot : 0.01;
   g_py_stale         = false;

   string gateTag = useFixed
      ? StringFormat("⚠️ LOT GATE ACTIVE — fixed=%.2f", fixedLot)
      : "✅ Gate open";

   g_py_status = StringFormat("✅ Health %.1f | %s | %s | %.4fx | %s",
                              health, mode, regime, mult, gateTag);

   if(DebugMode)
      Print(StringFormat(
         "[PY_Bridge] mult=%.4f  health=%.1f  mode=%s  regime=%s  "
         "use_fixed_lot=%s  fixed_lot=%.2f  age=%ds",
         mult, health, mode, regime,
         useFixed ? "true" : "false", fixedLot, age));
}

double PY_GetMultiplier()
{
   if(!PY_Enabled) return 1.0;
   if(g_py_stale)  return PY_FallbackMult;
   return g_py_multiplier;
}

// ============================================================
// LOT SIZING
// ============================================================
double CalcBaseLot()
{
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double steps  = MathFloor(equity / EquityStep);
   if(steps < 1) steps = 1;
   return NormalizeDouble(steps * LotPerStep, 2);
}

double CalcLotSize()
{
   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double lotMin  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double lotMax  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);

   if(PY_Enabled && !g_py_stale && g_py_use_fixed_lot)
   {
      double fixedLot = g_py_fixed_lot;
      fixedLot = MathFloor(fixedLot / lotStep) * lotStep;
      fixedLot = MathMax(fixedLot, lotMin);
      fixedLot = MathMin(fixedLot, lotMax);
      if(DebugMode)
         Print(StringFormat(
            "[LotCalc] ⚠️ LOT GATE ACTIVE — fixed=%.2f  health=%.1f",
            fixedLot, g_py_health));
      return NormalizeDouble(fixedLot, 2);
   }

   double baseLot = CalcBaseLot();
   double mult    = PY_GetMultiplier();
   double lot     = baseLot * mult;

   lot = MathFloor(lot / lotStep) * lotStep;
   lot = MathMax(lot, lotMin);
   lot = MathMin(lot, lotMax);

   if(DebugMode)
      Print(StringFormat(
         "[LotCalc] ✅ Gate open — equity=$%.2f base=%.2f py_mult=%.4f final=%.2f",
         AccountInfoDouble(ACCOUNT_EQUITY), baseLot, mult, lot));

   return NormalizeDouble(lot, 2);
}

// ============================================================
// COMMENT MASKING
// ============================================================
string EncodeStrategy(string strategy)
{
   if(strategy=="UTBOT")                      return "A1";
   if(strategy=="TRAP")                       return "B2";
   if(strategy=="OB")                         return "C3";
   if(strategy=="FRACTAL")                    return "D4";
   if(strategy=="SNIPER")                     return "E5";
   if(strategy=="FVG")                        return "F6";
   if(strategy=="AMT")                        return "G7";
   if(strategy=="LVN")                        return "H8";
   if(strategy=="LIQ"||strategy=="LIQUIDITY") return "I9";
   if(strategy=="SMARTENTRY")                 return "J0";
   if(StringFind(strategy,"MSV")>=0)          return "K1";
   return "Z0";
}

string DecodeStrategy(string code)
{
   if(code=="A1") return "UTBOT";
   if(code=="B2") return "TRAP";
   if(code=="C3") return "OB";
   if(code=="D4") return "FRACTAL";
   if(code=="E5") return "SNIPER";
   if(code=="F6") return "FVG";
   if(code=="G7") return "AMT";
   if(code=="H8") return "LVN";
   if(code=="I9") return "LIQUIDITY";
   if(code=="J0") return "SMARTENTRY";
   if(code=="K1") return "MSV";
   return "default";
}

// ============================================================
// COUNT / CLOSE HELPERS
// ============================================================
int CountStrategyTrades(string strategy)
{
   string tag   = "TF_" + EncodeStrategy(strategy) + "_";
   ulong  magic = GetMagicForStrategy(strategy);
   int    count = 0;
   for(int i = PositionsTotal()-1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i); if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL)  != _Symbol)       continue;
      if(PositionGetInteger(POSITION_MAGIC)  != (long)magic)   continue;
      if(StringFind(PositionGetString(POSITION_COMMENT), tag) >= 0) count++;
   }
   return count;
}

int CountMSVTrades()
{
   int count = 0;
   for(int i = PositionsTotal()-1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i); if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL)  != _Symbol)              continue;
      if(PositionGetInteger(POSITION_MAGIC)  != (long)MAGIC_MSV)      continue;
      if(StringFind(PositionGetString(POSITION_COMMENT),"TF_K1_") >= 0) count++;
   }
   return count;
}

string GetMSVDirection()
{
   int buys = 0, sells = 0;
   for(int i = PositionsTotal()-1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i); if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL)  != _Symbol)         continue;
      if(PositionGetInteger(POSITION_MAGIC)  != (long)MAGIC_MSV) continue;
      if(StringFind(PositionGetString(POSITION_COMMENT),"TF_K1_") < 0) continue;
      long pt = PositionGetInteger(POSITION_TYPE);
      if(pt == POSITION_TYPE_BUY)  buys++;
      if(pt == POSITION_TYPE_SELL) sells++;
   }
   if(buys  > 0 && sells == 0) return "BUY";
   if(sells > 0 && buys  == 0) return "SELL";
   return "";
}

int CloseOppositeDirectionTrades(string keepDirection)
{
   long closeType = (keepDirection == "BUY") ? POSITION_TYPE_SELL : POSITION_TYPE_BUY;
   int  closed    = 0;
   for(int i = PositionsTotal()-1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i); if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL)  != _Symbol) continue;
      if(!IsUnicoreMagic((ulong)PositionGetInteger(POSITION_MAGIC))) continue;
      if(StringFind(PositionGetString(POSITION_COMMENT),"TF_") < 0) continue;
      if(PositionGetInteger(POSITION_TYPE) != closeType)             continue;
      string cmt = PositionGetString(POSITION_COMMENT);
      double pnl = PositionGetDouble(POSITION_PROFIT) + PositionGetDouble(POSITION_SWAP);
      if(g_trade.PositionClose(ticket, Slippage))
      {
         closed++;
         Print("🔄 DIR-FLUSH | #", ticket, " [", cmt, "] P&L: $", DoubleToString(pnl, 2));
      }
   }
   return closed;
}

int CloseStrategyTrades(string strategy)
{
   string tag        = "TF_" + EncodeStrategy(strategy) + "_";
   ulong  magic      = GetMagicForStrategy(strategy);
   int    closedCount = 0;
   for(int i = PositionsTotal()-1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i); if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL)  != _Symbol)       continue;
      if(PositionGetInteger(POSITION_MAGIC)  != (long)magic)   continue;
      string comment = PositionGetString(POSITION_COMMENT);
      if(StringFind(comment, tag) < 0) continue;
      double profit = PositionGetDouble(POSITION_PROFIT) + PositionGetDouble(POSITION_SWAP);
      if(g_trade.PositionClose(ticket, Slippage))
      {
         closedCount++;
         Print("🔄 Manual Close | #", ticket, " [", comment, "] P&L: $", DoubleToString(profit, 2));
      }
   }
   return closedCount;
}

int CountAllOpenTrades()
{
   int count = 0;
   for(int i = PositionsTotal()-1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i); if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL)  != _Symbol) continue;
      if(!IsUnicoreMagic((ulong)PositionGetInteger(POSITION_MAGIC))) continue;
      if(StringFind(PositionGetString(POSITION_COMMENT),"TF_") >= 0) count++;
   }
   return count;
}

string NormalizeStrategy(string raw){ StringToUpper(raw); return raw; }

// ============================================================
// STRATEGY VALIDATION
// ============================================================
bool IsKnownStrategy(string strategy)
{
   if(strategy=="UTBOT"    ||strategy=="TRAP"    ||strategy=="OB"     ||
      strategy=="FRACTAL"  ||strategy=="SNIPER"  ||strategy=="FVG"    ||
      strategy=="AMT"      ||strategy=="LVN"     ||strategy=="LIQ"    ||
      strategy=="LIQUIDITY"||strategy=="SMARTENTRY") return true;
   if(StringFind(strategy,"MSV")>=0 && StringFind(strategy,"BREACH")>=0) return true;
   return false;
}

bool IsMSVStrategy(string strategy)
{
   return(StringFind(strategy,"MSV")>=0 && StringFind(strategy,"BREACH")>=0);
}

// ============================================================
// SIGNAL ROUTING
// ============================================================
void RouteSignal(string id, string action, string strategy, double entry, double sl)
{
   if(!IsKnownStrategy(strategy)){ g_lastError="Unknown strategy: "+strategy; return; }
   if(strategy=="FVG" && g_fvgLastWinTime>0)
   {
      int elapsed=(int)(TimeCurrent()-g_fvgLastWinTime);
      if(elapsed<3600){ g_lastError="FVG cooldown"; return; }
   }
   int existingTrades = CountStrategyTrades(strategy);
   if(existingTrades > 0)
      Print("║ [",strategy,"] Stacking — adding to ",IntegerToString(existingTrades)," existing trade(s)");
   OpenTrade(id, action, strategy, sl);
}

// ============================================================
// TRADING INTELLIGENCE AI — v5.16
// TI_CheckSignal now returns a TIResult struct so snapshot_id
// and prediction_id are available to OpenTrade().
// ============================================================
string TI_Candles(ENUM_TIMEFRAMES tf, int count)
{
   MqlRates rates[];
   int copied = CopyRates(_Symbol, tf, 1, count, rates);
   if(copied <= 0) return "[]";
   string arr = "[";
   for(int i = 0; i < copied; i++)
   {
      if(i > 0) arr += ",";
      arr += StringFormat(
         "{\"open\":%f,\"high\":%f,\"low\":%f,\"close\":%f,\"volume\":%d}",
         rates[i].open, rates[i].high, rates[i].low, rates[i].close,
         (int)rates[i].tick_volume
      );
   }
   return arr + "]";
}

string TI_ParseStr(const string json, const string key)
{
   string search = "\"" + key + "\":\"";
   int s = StringFind(json, search);
   if(s < 0) return "";
   s += StringLen(search);
   int e = StringFind(json, "\"", s);
   return (e < 0) ? "" : StringSubstr(json, s, e - s);
}

double TI_ParseDbl(const string json, const string key)
{
   string search = "\"" + key + "\":";
   int s = StringFind(json, search);
   if(s < 0) return 0.0;
   s += StringLen(search);
   int ec = StringFind(json, ",", s);
   int eb = StringFind(json, "}", s);
   if(ec < 0 && eb < 0) return 0.0;
   int e = (ec < 0) ? eb : (eb < 0) ? ec : MathMin(ec, eb);
   return StringToDouble(StringSubstr(json, s, e - s));
}

// ── v5.16: returns TIResult struct instead of just a decision string ──
TIResult TI_CheckSignal(string action, string strategy)
{
   TIResult res;
   res.decision     = "ALLOW";
   res.snapshotId   = "";
   res.predictionId = "";
   res.regime       = "";
   res.quality      = 0.0;
   res.buyProb      = 0.0;
   res.sellProb     = 0.0;

   if(!TI_Enabled)
   {
      g_ti_predictStatus = "SKIPPED (AI disabled)";
      g_ti_predictTime   = TimeCurrent();
      return res;
   }

   double balance    = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity     = AccountInfoDouble(ACCOUNT_EQUITY);
   double drawdown   = (balance > 0.0) ? (balance - equity) / balance : 0.0;
   double spreadPts  = (double)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
   double spreadPips = spreadPts / 10.0;

   string json = StringFormat(
      "{"
      "\"ea_id\":\"%s\","
      "\"symbol\":\"%s\","
      "\"ea_signal\":\"%s\","
      "\"price\":%f,"
      "\"spread_pips\":%f,"
      "\"candles_m5\":%s,"
      "\"candles_m15\":%s,"
      "\"candles_h1\":%s,"
      "\"candles_h4\":%s,"
      "\"candles_d1\":%s,"
      "\"risk_context\":{"
         "\"account_balance\":%f,"
         "\"account_equity\":%f,"
         "\"account_drawdown_pct\":%f,"
         "\"recent_loss_streak\":%d,"
         "\"recent_win_streak\":%d,"
         "\"trades_today\":%d"
      "}}",
      strategy,
      _Symbol,
      action,
      (action == "BUY") ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                        : SymbolInfoDouble(_Symbol, SYMBOL_BID),
      spreadPips,
      TI_Candles(PERIOD_M5,  50),
      TI_Candles(PERIOD_M15, 30),
      TI_Candles(PERIOD_H1,  20),
      TI_Candles(PERIOD_H4,  10),
      TI_Candles(PERIOD_D1,   5),
      balance, equity, drawdown,
      (int)g_loss_streak,
      (int)g_win_streak,
      g_todayTradeCount
   );

   string url     = TI_API_URL + "/predict";
   string headers = "Content-Type: application/json\r\nX-API-Key: " + TI_API_KEY + "\r\n";
   char   post[], result[];
   string resHeaders;
   StringToCharArray(json, post, 0, StringLen(json));

   int httpCode = WebRequest("POST", url, headers, TI_Timeout_ms, post, result, resHeaders);
   if(httpCode < 0 || httpCode != 200)
   {
      Print("[TI] API unreachable (http=", httpCode, ") — fail-open, trade allowed");
      g_ti_status         = "⚠️ API offline — trades allowed";
      g_ti_predictStatus  = StringFormat("FAIL http=%d (err=%d)", httpCode, GetLastError());
      g_ti_predictTime    = TimeCurrent();
      return res;   // decision stays "ALLOW"
   }

   string resp = CharArrayToString(result);

   // Parse all fields including the new IDs
   res.decision     = TI_ParseStr(resp, "final_decision");
   res.snapshotId   = TI_ParseStr(resp, "snapshot_id");
   res.predictionId = TI_ParseStr(resp, "prediction_id");
   res.regime       = TI_ParseStr(resp, "regime");
   res.quality      = TI_ParseDbl(resp, "risk_quality_score");
   res.buyProb      = TI_ParseDbl(resp, "trader_buy_prob");
   res.sellProb     = TI_ParseDbl(resp, "trader_sell_prob");

   if(res.decision == "") res.decision = "ALLOW";

   // Update global display state
   g_ti_decision  = res.decision;
   g_ti_regime    = res.regime;
   g_ti_quality   = res.quality;
   g_ti_buy_prob  = res.buyProb;
   g_ti_sell_prob = res.sellProb;

   bool blocked = (res.decision == "BLOCK");
   if(blocked) g_ti_blocks++; else g_ti_allows++;

   g_ti_status = StringFormat("%s | %s | Q:%.0f%% | B:%.0f%% S:%.0f%%",
      res.decision, res.regime,
      res.quality  * 100,
      res.buyProb  * 100,
      res.sellProb * 100);

   g_ti_predictStatus = "OK";
   g_ti_predictTime   = TimeCurrent();

   Print(StringFormat(
      "[TI] [%s] %s → %s | Regime: %s | Quality: %.1f%% | Buy: %.1f%% Sell: %.1f%%"
      " | snap=%s | pred=%s",
      strategy, action, res.decision, res.regime,
      res.quality * 100, res.buyProb * 100, res.sellProb * 100,
      res.snapshotId, res.predictionId));

   return res;
}

// ── v5.18: extended with magic_number + open-time session ──
void TI_ReportTrade(ulong dealTicket, string symbol, string strategy,
                    string direction, double entryPrice, double exitPrice,
                    double pnlPips, double pnlUsd, string outcome, bool wasFlipped,
                    string snapshotId, string predictionId,
                    string regime, string session, double lotSize,
                    double maxDrawdownPips, datetime openedAt, long magicNumber)
{
   if(!TI_Enabled)
   {
      g_ti_reportStatus = "SKIPPED (AI disabled)";
      g_ti_reportTime   = TimeCurrent();
      return;
   }

   string json = StringFormat(
      "{\"ea_id\":\"%s\","
      "\"strategy\":\"%s\","
      "\"magic_number\":%d,"
      "\"mt5_ticket\":%d,\"symbol\":\"%s\","
      "\"direction\":\"%s\",\"entry_price\":%f,"
      "\"exit_price\":%f,\"pnl_pips\":%f,"
      "\"pnl_usd\":%f,\"outcome\":\"%s\","
      "\"was_flipped\":%s,"
      "\"snapshot_id\":\"%s\","
      "\"prediction_id\":\"%s\","
      "\"regime\":\"%s\","
      "\"session\":\"%s\","
      "\"lot_size\":%f,"
      "\"max_drawdown_pips\":%f,"
      "\"opened_at\":\"%s\"}",
      strategy,
      strategy,
      magicNumber,
      (int)dealTicket, symbol, direction,
      entryPrice, exitPrice, pnlPips, pnlUsd, outcome,
      wasFlipped ? "true" : "false",
      snapshotId,
      predictionId,
      regime,
      session,
      lotSize,
      maxDrawdownPips,
      TimeToString(openedAt, TIME_DATE|TIME_SECONDS)
   );

   string url     = TI_API_URL + "/trade/update";
   string headers = "Content-Type: application/json\r\nX-API-Key: " + TI_API_KEY + "\r\n";
   char   post[], result[];
   string resHeaders;
   StringToCharArray(json, post, 0, StringLen(json));

   int res = WebRequest("POST", url, headers, TI_Timeout_ms, post, result, resHeaders);
   g_ti_reportTime = TimeCurrent();
   if(res < 0 || res != 200)
   {
      g_ti_reportStatus = StringFormat("FAIL http=%d (err=%d)", res, GetLastError());
      Print("[TI] ⚠️ Failed to report trade #", dealTicket, " (http=", res, ")");
   }
   else
   {
      g_ti_reportStatus = "OK #" + IntegerToString((int)dealTicket);
      Print("[TI] ✅ Reported: [", strategy, "] #", dealTicket,
            " | ", outcome,
            " | ", DoubleToString(pnlPips, 1), " pips",
            " | $", DoubleToString(pnlUsd, 2),
            " | flipped=", wasFlipped ? "YES" : "NO",
            " | snap=", snapshotId != "" ? snapshotId : "none",
            " | pred=", predictionId != "" ? predictionId : "none",
            " | magic=", magicNumber,
            " | session=", session,
            " | opened=", openedAt > 0
                           ? TimeToString(openedAt, TIME_DATE|TIME_SECONDS)
                           : "unknown");
   }
}

// ============================================================
// OPEN TRADE — v5.17: stores openedAt alongside other meta
// ============================================================
bool OpenTrade(string signalID, string action, string strategy, double stopLoss)
{
   if(g_todayTradeCount >= 2000){ g_lastError="Max 2000 trades reached"; return false; }
   if(action!="BUY" && action!="SELL"){ g_lastError="Invalid action"; return false; }

   double lotSize    = CalcLotSize();
   bool   wasFlipped = false;

   // ── Trading Intelligence AI check ──────────────────────────────────────
   string snapshotId    = "";
   string predictionId  = "";
   string tiRegime      = "";
   bool   aiWantedBlock = false;   // v5.20: true if AI said BLOCK but override let it through

   if(TI_Enabled)
   {
      TIResult tiRes = TI_CheckSignal(action, strategy);   // v5.16: full struct

      // Capture IDs immediately — before any early return
      snapshotId   = tiRes.snapshotId;
      predictionId = tiRes.predictionId;
      tiRegime     = tiRes.regime;

      if(tiRes.decision == "BLOCK")
      {
         bool overrideActive = TI_OverrideBlock && (!TI_OverrideDemoOnly || IsDemoAccount());

         if(!overrideActive)
         {
            g_lastError = "TI AI: BLOCKED (" + tiRes.regime + " | Q:" +
                          DoubleToString(tiRes.quality * 100, 0) + "%)";
            Print("🚫 [TI] Trade BLOCKED | Regime: ", tiRes.regime,
                  " | Quality: ", DoubleToString(tiRes.quality * 100, 1), "%",
                  " | Strategy: ", strategy);
            return false;
         }

         // v5.20: override active — verdict is logged/reported, not enforced
         aiWantedBlock = true;
         g_ti_overrides++;
         Print("⚠️ [TI] BLOCK OVERRIDDEN — trading anyway (training mode) | Regime: ", tiRes.regime,
               " | Quality: ", DoubleToString(tiRes.quality * 100, 1), "%",
               " | Strategy: ", strategy);
      }

      if(TI_AllowFlip)
      {
         if(tiRes.decision == "FLIP_TO_BUY" && action == "SELL")
         {
            Print("🔄 [TI] FLIP: SELL → BUY | Regime: ", tiRes.regime, " | Strategy: ", strategy);
            action     = "BUY";
            wasFlipped = true;
         }
         else if(tiRes.decision == "FLIP_TO_SELL" && action == "BUY")
         {
            Print("🔄 [TI] FLIP: BUY → SELL | Regime: ", tiRes.regime, " | Strategy: ", strategy);
            action     = "SELL";
            wasFlipped = true;
         }
      }
   }
   // ── End TI check ────────────────────────────────────────────────────────

   ulong  tradeMagic  = GetMagicForStrategy(strategy);
   string encodedCode = EncodeStrategy(strategy);

   // _FL / _OV suffixes let OnTradeTransaction decode flip/override status at close time
   string flipTag     = wasFlipped    ? "_FL" : "";
   string overrideTag = aiWantedBlock ? "_OV" : "";
   string comment = "TF_" + encodedCode + "_" + StringSubstr(signalID, 0, 8) + flipTag + overrideTag;

   double executePrice = NormalizeDouble(
      (action=="BUY") ? SymbolInfoDouble(_Symbol,SYMBOL_ASK) : SymbolInfoDouble(_Symbol,SYMBOL_BID),
      _Digits);

   long   stopLevelPts = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   double minStop      = stopLevelPts * _Point;

   double sl;
   if(action=="BUY")  sl = NormalizeDouble(executePrice - FixedSLPoints, _Digits);
   else               sl = NormalizeDouble(executePrice + FixedSLPoints, _Digits);
   if(action=="BUY"  && (executePrice-sl) < minStop) sl = NormalizeDouble(executePrice - minStop, _Digits);
   if(action=="SELL" && (sl-executePrice) < minStop) sl = NormalizeDouble(executePrice + minStop, _Digits);

   string gateLabel = (PY_Enabled && !g_py_stale && g_py_use_fixed_lot)
      ? StringFormat("GATE=ACTIVE(fixed=%.2f)", g_py_fixed_lot)
      : StringFormat("GATE=OPEN(mult=%.4f)", PY_GetMultiplier());

   Print("║ [",strategy," → ",encodedCode,"] ",action,
         " @ ",DoubleToString(executePrice,_Digits),
         " | SL: ",DoubleToString(sl,_Digits),
         " | Lots: ",DoubleToString(lotSize,2),
         " | Magic: ",IntegerToString((int)tradeMagic),
         " | TI: ",g_ti_decision," Q:",DoubleToString(g_ti_quality*100,0),"%",
         " | Flipped: ", wasFlipped ? "YES" : "NO",
         " | Override: ", aiWantedBlock ? "YES" : "NO",
         " | PY health: ",DoubleToString(g_py_health,1),
         " | ",gateLabel,
         " | Comment: ",comment);

   g_trade.SetExpertMagicNumber(tradeMagic);
   g_trade.SetDeviationInPoints(Slippage);

   int attempts=3; bool success=false;
   while(attempts > 0)
   {
      executePrice = NormalizeDouble(
         (action=="BUY") ? SymbolInfoDouble(_Symbol,SYMBOL_ASK) : SymbolInfoDouble(_Symbol,SYMBOL_BID),
         _Digits);
      if(action=="BUY")  sl = NormalizeDouble(executePrice - FixedSLPoints, _Digits);
      else               sl = NormalizeDouble(executePrice + FixedSLPoints, _Digits);
      if(action=="BUY"  && (executePrice-sl) < minStop) sl = NormalizeDouble(executePrice - minStop, _Digits);
      if(action=="SELL" && (sl-executePrice) < minStop) sl = NormalizeDouble(executePrice + minStop, _Digits);

      if(action=="BUY") success = g_trade.Buy (lotSize, _Symbol, executePrice, sl, 0, comment);
      else              success = g_trade.Sell(lotSize, _Symbol, executePrice, sl, 0, comment);
      if(success && g_trade.ResultRetcode()==TRADE_RETCODE_DONE) break;
      uint retcode = g_trade.ResultRetcode();
      if(retcode!=TRADE_RETCODE_REQUOTE && retcode!=TRADE_RETCODE_PRICE_CHANGED && retcode!=TRADE_RETCODE_PRICE_OFF) break;
      Sleep(200); attempts--; success=false;
   }

   if(success && g_trade.ResultRetcode()==TRADE_RETCODE_DONE)
   {
      ulong openedTicket = g_trade.ResultOrder();
      g_todayTradeCount++;
      g_lastError = "";

      // ── v5.18: capture open timestamp AND open-time session ──
      string openSess = GetCurrentSession();
      StoreMeta(openedTicket, snapshotId, predictionId, tiRegime, lotSize, strategy,
                TimeCurrent(), openSess);
      if(DebugMode)
         Print("[TI] Stored meta for ticket #", openedTicket,
               " snap=", snapshotId, " pred=", predictionId,
               " openedAt=", TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS),
               " session=", openSess);

      Print("✅ #",openedTicket," | [",strategy,"] ",action,
            " @ ",DoubleToString(executePrice,_Digits),
            " | SL: ",DoubleToString(sl,_Digits),
            " | Lots: ",DoubleToString(lotSize,2),
            " | Magic: ",IntegerToString((int)tradeMagic),
            " | Tag: ",comment);
      if(SendNotifications)
         SendNotification("Unicore ["+strategy+"] "+action+" "+DoubleToString(lotSize,2)+" @ "+DoubleToString(executePrice,_Digits));
      return true;
   }
   g_lastError = "Trade Error: "+IntegerToString(g_trade.ResultRetcode())+" — "+g_trade.ResultRetcodeDescription();
   Print("❌ ", g_lastError); return false;
}

// ============================================================
// INIT / DEINIT / TIMER / TICK
// ============================================================
int OnInit()
{
   Print("══════════════════════════════════════════════════════════");
   Print("║ Unicore v5.17 INITIALIZED");
   Print("║ Symbol: ",_Symbol," | Account: #",(long)AccountInfoInteger(ACCOUNT_LOGIN));
   Print("║ SL: ",DoubleToString(FixedSLPoints,1)," pts | NO TP");
   Print("║ Exit: MANUAL ONLY — signals stack, no auto-close");
   Print("║ COMMENT MASKING: A1=UTBOT B2=TRAP C3=OB D4=FRAC E5=SNIP F6=FVG G7=AMT H8=LVN I9=LIQ J0=SE K1=MSV");
   Print("║ LOT SIZING: Python Bridge is sole authority");
   Print("║   Gate ACTIVE (health < 75) → fixed 0.01 lot");
   Print("║   Gate OPEN   (health >= 75) → ",DoubleToString(LotPerStep,2)," lots per $",DoubleToString(EquityStep,0)," equity × multiplier");
   Print("║ Python Lot Bridge: ", PY_Enabled ? "ENABLED ✅" : "DISABLED ❌");
   Print("║ ─────────────────────────────────────────────────────");
   Print("║ MAGIC NUMBER ROUTING (v5.14):");
   Print("║   77701=FRACTAL  77702=OB       77703=TRAP    77704=LIQUIDITY");
   Print("║   77705=AMT      77706=LVN      77707=SE      777701=FVG");
   Print("║   88801=UTBOT    88888=MSV");
   Print("║ MSV Gate: REMOVED — all signals execute freely");
   Print("║ ─────────────────────────────────────────────────────");
   Print("║ Trading Intelligence AI: ", TI_Enabled ? "ENABLED ✅" : "DISABLED ❌");
   if(TI_Enabled)
   {
      Print("║   API URL: ", TI_API_URL);
      Print("║   Flip mode: ", TI_AllowFlip ? "ENABLED (AI can reverse direction)" : "DISABLED (AI can only block)");
      Print("║   Fail-open: trades allowed when API is offline");
      Print("║   ea_id: per-strategy profiling ENABLED ✅");
      Print("║   snapshot_id + prediction_id round-trip: ENABLED ✅ (v5.16)");
      Print("║   opened_at round-trip: ENABLED ✅ (v5.17)");
      Print("║   Add to WebRequest URLs: ", TI_API_URL);
      Print("║   IMPORTANT for XAUUSD: set MAX_SPREAD_PIPS=50 in your .env file");
   }
   Print("══════════════════════════════════════════════════════════");

   PY_ReadLotParams();

   g_lastCheckTime = g_lastResetDate = TimeCurrent();
   g_attachTimeGMT = TimeGMT();
   LoadProcessedIDs();
   EventSetTimer(PollingInterval);
   Print("║ Attach time (GMT): ",TimeToString(g_attachTimeGMT,TIME_DATE|TIME_SECONDS));
   CheckForNewSignals();
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason){ EventKillTimer(); Comment(""); }

void OnTimer()
{
   ResetDailyCounter();
   UpdateChartDisplay();
   datetime now = TimeCurrent();
   if(now - g_lastCheckTime < PollingInterval) return;
   g_lastCheckTime = now;
   PY_ReadLotParams();
   CheckForNewSignals();
}

void OnTick(){}

// ============================================================
// TRADE CLOSE REPORTER — v5.17: also retrieves openedAt
// ============================================================
void OnTradeTransaction(
   const MqlTradeTransaction &trans,
   const MqlTradeRequest     &request,
   const MqlTradeResult      &result
)
{
   if(!TI_Enabled)
   {
      g_ti_patchStatus = "SKIPPED (AI disabled)";
      g_ti_patchTime   = TimeCurrent();
      return;
   }
   if(trans.type != TRADE_TRANSACTION_DEAL_ADD) return;

   ulong dealTicket = trans.deal;
   if(!HistoryDealSelect(dealTicket)) return;

   if(!IsUnicoreMagic((ulong)HistoryDealGetInteger(dealTicket, DEAL_MAGIC))) return;
   if(HistoryDealGetInteger(dealTicket, DEAL_ENTRY)  != DEAL_ENTRY_OUT) return;
   if(HistoryDealGetString (dealTicket, DEAL_SYMBOL) != _Symbol)        return;

   // ── Extract strategy and flip status from comment ────────────────────────
   string comment    = HistoryDealGetString(dealTicket, DEAL_COMMENT);
   string strategy   = "default";
   bool   wasFlipped = false;

   int startPos = StringFind(comment, "TF_");
   if(startPos >= 0)
   {
      startPos += 3;
      int endPos = StringFind(comment, "_", startPos);
      if(endPos > startPos)
      {
         string encodedCode = StringSubstr(comment, startPos, endPos - startPos);
         strategy = DecodeStrategy(encodedCode);
      }
   }
   wasFlipped = (StringFind(comment, "_FL") >= 0);

   // ── v5.16/v5.17: look up meta by position ticket ─────────────────────────
   ulong    posId        = (ulong)HistoryDealGetInteger(dealTicket, DEAL_POSITION_ID);
   string   snapshotId   = "";
   string   predictionId = "";
   string   metaRegime   = "";
   double   metaLotSize  = 0.0;
   datetime metaOpenedAt = 0;   // v5.17
   string   metaSession  = "";  // v5.18: open-time session

   TicketMeta meta;
   if(GetMeta(posId, meta))
   {
      snapshotId   = meta.snapshotId;
      predictionId = meta.predictionId;
      metaRegime   = meta.regime;
      metaLotSize  = meta.lotSize;
      metaOpenedAt = meta.openedAt;
      metaSession  = meta.openSession;  // v5.18
      if(meta.strategy != "default" && meta.strategy != "")
         strategy = meta.strategy;   // authoritative source
      RemoveMeta(posId);             // clean up — trade is closed
   }
   else
   {
      if(DebugMode)
         Print("[TI] No meta found for position #", posId, " — snapshot_id/opened_at/session will be empty");
   }
   // ─────────────────────────────────────────────────────────────────────────

   double profit    = HistoryDealGetDouble (dealTicket, DEAL_PROFIT)
                    + HistoryDealGetDouble (dealTicket, DEAL_SWAP)
                    + HistoryDealGetDouble (dealTicket, DEAL_COMMISSION);
   double exitPrice = HistoryDealGetDouble (dealTicket, DEAL_PRICE);
   long   dealType  = HistoryDealGetInteger(dealTicket, DEAL_TYPE);
   string direction = (dealType == DEAL_TYPE_BUY) ? "BUY" : "SELL";

   double entryPrice = exitPrice;
   if(HistorySelectByPosition(posId))
   {
      for(int i = 0; i < HistoryDealsTotal(); i++)
      {
         ulong d = HistoryDealGetTicket(i);
         if(d == 0) continue;
         if(HistoryDealGetInteger(d, DEAL_ENTRY) == DEAL_ENTRY_IN)
         {
            entryPrice = HistoryDealGetDouble(d, DEAL_PRICE);
            break;
         }
      }
   }

   double pnlPips = (direction == "BUY")
      ? (exitPrice - entryPrice) / (_Point * 10.0)
      : (entryPrice - exitPrice) / (_Point * 10.0);

   string outcome = (profit >  0.01) ? "WIN"
                  : (profit < -0.01) ? "LOSS"
                  : "BREAKEVEN";

   if(profit >  0.01){ g_win_streak++;  g_loss_streak = 0; }
   if(profit < -0.01){ g_loss_streak++; g_win_streak  = 0; }

   // v5.18: use session captured at open time; fall back to close-time only if meta was lost
   string session = (metaSession != "") ? metaSession : GetCurrentSession();

   // max_drawdown_pips: not tracked intra-trade by the EA, send 0 as placeholder
   double maxDrawdownPips = 0.0;

   // v5.18: resolve magic number from strategy name for the DB record
   long magicNumber = (long)GetMagicForStrategy(strategy);

   Print("[TI] Trade closed: #", dealTicket,
         " | [", strategy, "]",
         " | ", direction,
         " | ", outcome,
         " | Profit: $", DoubleToString(profit, 2),
         " | PnL: ", DoubleToString(pnlPips, 1), " pips",
         " | Flipped: ", wasFlipped ? "YES" : "NO",
         " | snap=", snapshotId != "" ? snapshotId : "none",
         " | pred=", predictionId != "" ? predictionId : "none",
         " | opened=", metaOpenedAt > 0
                        ? TimeToString(metaOpenedAt, TIME_DATE|TIME_SECONDS)
                        : "unknown");

   TI_ReportTrade(dealTicket, _Symbol, strategy, direction,
                  entryPrice, exitPrice, pnlPips, profit, outcome, wasFlipped,
                  snapshotId, predictionId, metaRegime, session,
                  metaLotSize, maxDrawdownPips, metaOpenedAt, magicNumber);  // v5.18
}

// ============================================================
// POLL SUPABASE
// ============================================================
void CheckForNewSignals()
{
   g_lastPollTime    = TimeCurrent();
   g_lastSignalStatus= "Polling...";
   datetime ago = TimeGMT() - (LookbackHours * 3600);
   MqlDateTime dt; TimeToStruct(ago, dt);
   string iso = StringFormat("%04d-%02d-%02dT%02d:%02d:%02d",
                             dt.year, dt.mon, dt.day, dt.hour, dt.min, dt.sec);
   string url = SupabaseURL+"/rest/v1/"+TableName+
                StringFormat("?status=eq.filled&created_at=gte.%s&order=created_at.desc&limit=20&apikey=%s",
                             iso, SupabaseKey);
   string headers = "Authorization: Bearer "+SupabaseKey+"\r\nContent-Type: application/json\r\n";
   char data[], result[]; string resHeaders; ResetLastError();
   int res = WebRequest("GET",url,headers,"",5000,data,0,result,resHeaders);
   if(res==-1){ int err=GetLastError(); g_lastSignalStatus="Connection error: "+IntegerToString(err); return; }
   string response = CharArrayToString(result);
   if(StringLen(response)<3){ g_lastSignalStatus="Waiting..."; return; }
   ProcessSignalsArray(response);
}

// ============================================================
// SPLIT JSON ARRAY
// ============================================================
int SplitJSONArray(string json, string &objects[], int maxObjects)
{
   StringTrimLeft(json); StringTrimRight(json);
   if(StringSubstr(json,0,1)=="[") json=StringSubstr(json,1,StringLen(json)-2);
   StringTrimLeft(json); StringTrimRight(json);
   if(StringLen(json)<5) return 0;
   int count=0, depth=0, objStart=-1;
   for(int i=0;i<StringLen(json);i++)
   {
      string ch = StringSubstr(json,i,1);
      if(ch=="{")
      {
         if(depth==0) objStart=i;
         depth++;
      }
      else if(ch=="}")
      {
         depth--;
         if(depth==0 && objStart>=0)
         {
            if(count >= maxObjects) break;
            ArrayResize(objects, count+1);
            objects[count] = StringSubstr(json,objStart,i-objStart+1);
            count++; objStart=-1;
         }
      }
   }
   return count;
}

// ============================================================
// PARSED SIGNAL
// ============================================================
struct ParsedSignal
{
   string id;
   string action;
   string original_action;
   string symbol;
   string strategy;
   double entry;
   double sl;
};

// ============================================================
// PROCESS SIGNALS
// ============================================================
void ProcessSignalsArray(string json)
{
   string objects[]; int count = SplitJSONArray(json, objects, 20);
   if(count==0){ g_lastSignalStatus="Waiting..."; return; }
   DebugPrint("Poll returned "+IntegerToString(count)+" filled signal(s)");

   ParsedSignal valid[]; int validCount=0;
   for(int idx=count-1;idx>=0;idx--)
   {
      string obj      = objects[idx];
      string id       = ExtractJSONValue(obj,"id");
      string action   = ExtractJSONValue(obj,"action");
      string symbol   = ExtractJSONValue(obj,"symbol");
      string strategy = ExtractJSONValue(obj,"strategy");
      string entryStr = ExtractJSONValue(obj,"entry_price");
      string slStr    = ExtractJSONValue(obj,"sl");
      string status   = ExtractJSONValue(obj,"status");

      if(strategy=="") strategy="UNKNOWN";
      strategy = NormalizeStrategy(strategy);
      StringToUpper(action);

      if(id=="")     continue;
      if(action=="") continue;
      if(IsProcessed(id))                                        continue;
      if(status!="filled")                                       continue;
      if(TradeOnlyMySymbol && StringFind(symbol,SignalSymbol)<0) continue;

      ArrayResize(valid, validCount+1);
      valid[validCount].id             = id;
      valid[validCount].original_action= action;
      valid[validCount].symbol         = symbol;
      valid[validCount].strategy       = strategy;
      valid[validCount].entry          = StringToDouble(entryStr);
      valid[validCount].sl             = StringToDouble(slStr);

      if(strategy=="TRAP")
      {
         valid[validCount].action = (action=="BUY") ? "SELL" : "BUY";
         Print("🔄 INVERT [TRAP] ",action," → ",valid[validCount].action);
      }
      else { valid[validCount].action = action; }
      validCount++;
   }
   if(validCount==0){ g_lastSignalStatus="No valid signals"; UpdateChartDisplay(); return; }

   ParsedSignal msvSigs[];    int msvCount=0;
   ParsedSignal nonMsvSigs[]; int nonMsvCount=0;
   for(int i=0;i<validCount;i++)
   {
      if(IsMSVStrategy(valid[i].strategy))
         { ArrayResize(msvSigs,    msvCount+1);    msvSigs[msvCount++]    = valid[i]; }
      else
         { ArrayResize(nonMsvSigs, nonMsvCount+1); nonMsvSigs[nonMsvCount++] = valid[i]; }
   }

   Print("══════════════════════════════════════════════════════════");
   if(msvCount>0)    Print("║ 🟢 MSV SIGNALS: ",msvCount,"  — executing first (primary)");
   if(nonMsvCount>0) Print("║ 🔵 NON-MSV SIGNALS: ",nonMsvCount," — executing freely (gate removed)");

   int mirrored = 0;

   for(int i=0;i<msvCount;i++)
   {
      g_lastSignalID       = msvSigs[i].id;
      g_lastSignalAction   = msvSigs[i].action;
      g_lastSignalStrategy = msvSigs[i].strategy;
      g_lastError          = "";

      string existingDir = GetMSVDirection();
      if(existingDir!="" && existingDir!=msvSigs[i].action)
      {
         int flushed = CloseOppositeDirectionTrades(msvSigs[i].action);
         Print("║ 🔄 MSV DIRECTION FLIP [",existingDir," → ",msvSigs[i].action,"]",
               " — closed ",flushed," opposite trade(s)");
      }

      Print("║ 🟢 MSV PRIMARY [",msvSigs[i].strategy,"] ",msvSigs[i].action," — executing now");
      if(AutoTrade)
      {
         OpenTrade(msvSigs[i].id, msvSigs[i].action, msvSigs[i].strategy, msvSigs[i].sl);
         MarkProcessed(msvSigs[i].id); SaveProcessedIDs();
         g_lastSignalStatus = (g_lastError=="")
            ? "✅ ["+msvSigs[i].strategy+"] "+msvSigs[i].action+" (primary)"
            : "❌ "+g_lastError;
         UpdateSignalStatus(msvSigs[i].id, g_lastError=="" ? "mirrored" : "error");
         if(g_lastError=="") mirrored++;
      }
      else { MarkProcessed(msvSigs[i].id); g_lastSignalStatus="Auto-trade OFF"; }
   }

   if(nonMsvCount > 0)
   {
      string msvDir = GetMSVDirection();
      if(msvDir != "")
      {
         string oppositeDir = (msvDir=="BUY") ? "SELL" : "BUY";
         int dirBlocked = 0;
         for(int i=0;i<nonMsvCount;i++)
         {
            if(nonMsvSigs[i].id=="" || nonMsvSigs[i].action!=oppositeDir) continue;
            Print("⛔ DIR-BLOCK [",nonMsvSigs[i].strategy,"] ",nonMsvSigs[i].action,
                  " — MSV direction is ",msvDir);
            MarkProcessed(nonMsvSigs[i].id);
            UpdateSignalStatus(nonMsvSigs[i].id,"dir_blocked");
            nonMsvSigs[i].id = "";
            dirBlocked++;
         }
         if(dirBlocked>0)
            Print("║ 🧭 Direction filter: blocked ",dirBlocked," signal(s) opposing MSV ",msvDir);
      }

      Print("║ LOT SIZE (this batch): ",DoubleToString(CalcLotSize(),2),
            " | PY health: ",DoubleToString(g_py_health,1),
            " | Gate: ", (g_py_use_fixed_lot ? "ACTIVE (fixed 0.01)" : "OPEN"),
            " | Equity: $",DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY),2));
      Print("══════════════════════════════════════════════════════════");

      g_lastVote = (msvDir != "") ? "MSV DIR: " + msvDir : "FREE";

      for(int i=0;i<nonMsvCount;i++)
      {
         if(nonMsvSigs[i].id=="") continue;
         g_lastSignalID       = nonMsvSigs[i].id;
         g_lastSignalAction   = nonMsvSigs[i].action;
         g_lastSignalStrategy = nonMsvSigs[i].strategy;
         g_lastError          = "";
         if(AutoTrade)
         {
            RouteSignal(nonMsvSigs[i].id, nonMsvSigs[i].action, nonMsvSigs[i].strategy,
                        nonMsvSigs[i].entry, nonMsvSigs[i].sl);
            MarkProcessed(nonMsvSigs[i].id); SaveProcessedIDs();
            g_lastSignalStatus = (g_lastError=="")
               ? "✅ ["+nonMsvSigs[i].strategy+"] "+nonMsvSigs[i].action+" executed"
               : "❌ "+g_lastError;
            UpdateSignalStatus(nonMsvSigs[i].id, g_lastError=="" ? "mirrored" : "error");
            if(g_lastError=="") mirrored++;
         }
         else { MarkProcessed(nonMsvSigs[i].id); g_lastSignalStatus="Auto-trade OFF"; }
      }
   }

   if(mirrored>0) Print("║ BATCH: Executed ",mirrored," of ",validCount," signal(s)");
   UpdateChartDisplay();
}

// ============================================================
// HELPERS
// ============================================================
void UpdateSignalStatus(string signalID, string newStatus)
{
   string url = SupabaseURL+"/rest/v1/"+TableName+"?id=eq."+signalID+"&apikey="+SupabaseKey;
   string headers = "Authorization: Bearer "+SupabaseKey+
                    "\r\nContent-Type: application/json\r\nPrefer: return=minimal\r\n";
   string body = "{\"status\":\""+newStatus+"\"}";
   char data[], result[]; string resHeaders;
   StringToCharArray(body, data, 0, StringLen(body));
   int res = WebRequest("PATCH",url,headers,"",5000,data,StringLen(body),result,resHeaders);

   g_ti_patchTime = TimeCurrent();
   // Supabase PATCH with Prefer: return=minimal replies 204 on success, not 200
   if(res < 0 || (res != 200 && res != 204))
   {
      g_ti_patchStatus = StringFormat("FAIL http=%d (err=%d)", res, GetLastError());
      Print("[TI] ⚠️ Failed to patch signal ", signalID, " → ", newStatus, " (http=", res, ")");
   }
   else
      g_ti_patchStatus = "OK → " + newStatus;
}

void ResetDailyCounter()
{
   MqlDateTime dt, ldt;
   TimeToStruct(TimeCurrent(), dt);
   TimeToStruct(g_lastResetDate, ldt);
   if(dt.day != ldt.day)
   {
      g_lastResetDate  = TimeCurrent();
      g_processedCount = 0;
      g_todayTradeCount= 0;
   }
}

string ExtractJSONValue(string json, string key)
{
   string sk = "\""+key+"\":";
   int sp = StringFind(json, sk);
   if(sp < 0) return "";
   sp += StringLen(sk);
   while(sp<StringLen(json) && (StringSubstr(json,sp,1)==" " || StringSubstr(json,sp,1)=="\t")) sp++;
   bool isStr = StringSubstr(json,sp,1)=="\"";
   if(isStr) sp++;
   int ep;
   if(isStr) { ep = StringFind(json,"\"",sp); }
   else
   {
      int cp = StringFind(json,",",sp);
      int bp = StringFind(json,"}",sp);
      if(cp<0 && bp<0) ep=StringLen(json);
      else if(cp<0)    ep=bp;
      else if(bp<0)    ep=cp;
      else             ep=MathMin(cp,bp);
   }
   if(ep < 0) return "";
   string v = StringSubstr(json,sp,ep-sp);
   StringTrimLeft(v); StringTrimRight(v);
   return v;
}

void LoadProcessedIDs()
{
   string fname = "UNICORE_Processed_"+_Symbol+".txt";
   int fh = FileOpen(fname, FILE_READ|FILE_TXT);
   if(fh==INVALID_HANDLE) return;
   g_processedCount = 0;
   while(!FileIsEnding(fh) && g_processedCount<MAX_PROCESSED)
   {
      string line = FileReadString(fh);
      StringTrimLeft(line); StringTrimRight(line);
      if(StringLen(line)>0) g_processedIDs[g_processedCount++]=line;
   }
   FileClose(fh);
}

void SaveProcessedIDs()
{
   string fname = "UNICORE_Processed_"+_Symbol+".txt";
   int fh = FileOpen(fname, FILE_WRITE|FILE_TXT);
   if(fh==INVALID_HANDLE) return;
   for(int i=0;i<g_processedCount;i++) FileWriteString(fh, g_processedIDs[i]+"\n");
   FileClose(fh);
}

void CalculateTradeStats()
{
   g_totalWins=0; g_totalWinProfit=0; g_totalLosses=0; g_totalLossAmount=0; g_netProfit=0;
   HistorySelect(0, TimeCurrent());
   for(int i=HistoryDealsTotal()-1;i>=0;i--)
   {
      ulong d = HistoryDealGetTicket(i); if(d==0) continue;
      if(!IsUnicoreMagic((ulong)HistoryDealGetInteger(d, DEAL_MAGIC))) continue;
      if(HistoryDealGetString(d, DEAL_SYMBOL) != _Symbol)              continue;
      if(HistoryDealGetInteger(d,DEAL_ENTRY)  != DEAL_ENTRY_OUT)       continue;
      double p = HistoryDealGetDouble(d,DEAL_PROFIT)
               + HistoryDealGetDouble(d,DEAL_SWAP)
               + HistoryDealGetDouble(d,DEAL_COMMISSION);
      if(p >= 0){ g_totalWins++;   g_totalWinProfit  += p; }
      else      { g_totalLosses++; g_totalLossAmount += p; }
   }
   g_netProfit = g_totalWinProfit + g_totalLossAmount;
}

// ============================================================
// CHART DISPLAY  (v5.17)
// ============================================================
void UpdateChartDisplay()
{
   CalculateTradeStats();

   int utbotOpen=0,trapOpen=0,obOpen=0,fractalOpen=0,sniperOpen=0,fvgOpen=0;
   int amtOpen=0,lvnOpen=0,liqOpen=0,seOpen=0,msvOpen=0;

   for(int i=PositionsTotal()-1;i>=0;i--)
   {
      ulong ticket=PositionGetTicket(i); if(ticket==0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if(!IsUnicoreMagic((ulong)PositionGetInteger(POSITION_MAGIC))) continue;
      string cmt = PositionGetString(POSITION_COMMENT);
      if     (StringFind(cmt,"TF_A1_")>=0) utbotOpen++;
      else if(StringFind(cmt,"TF_B2_")>=0) trapOpen++;
      else if(StringFind(cmt,"TF_C3_")>=0) obOpen++;
      else if(StringFind(cmt,"TF_D4_")>=0) fractalOpen++;
      else if(StringFind(cmt,"TF_E5_")>=0) sniperOpen++;
      else if(StringFind(cmt,"TF_F6_")>=0) fvgOpen++;
      else if(StringFind(cmt,"TF_G7_")>=0) amtOpen++;
      else if(StringFind(cmt,"TF_H8_")>=0) lvnOpen++;
      else if(StringFind(cmt,"TF_I9_")>=0) liqOpen++;
      else if(StringFind(cmt,"TF_J0_")>=0) seOpen++;
      else if(StringFind(cmt,"TF_K1_")>=0) msvOpen++;
   }
   int totalOpen = utbotOpen+trapOpen+obOpen+fractalOpen+sniperOpen+fvgOpen
                 + amtOpen+lvnOpen+liqOpen+seOpen+msvOpen;

   string msvDirLabel;
   string msvDir = GetMSVDirection();
   if(msvDir == "")   msvDirLabel = "No live MSV";
   else               msvDirLabel = msvDir + " (dir-filter active)";

   string lotGateRow;
   if(!PY_Enabled)
      lotGateRow = "║ Lot Gate: DISABLED (bridge off)";
   else if(g_py_stale)
      lotGateRow = "║ Lot Gate: ⚠️ UNKNOWN (stale file)";
   else if(g_py_use_fixed_lot)
      lotGateRow = StringFormat("║ Lot Gate: ⚠️ ACTIVE — fixed=%.2f lot  (health %.1f < 75)",
                                g_py_fixed_lot, g_py_health);
   else
      lotGateRow = StringFormat("║ Lot Gate: ✅ OPEN — scaling active  (health %.1f >= 75)",
                                g_py_health);

   string tiRow;
   if(!TI_Enabled)
      tiRow = "║ AI Filter: DISABLED";
   else
      tiRow = StringFormat(
         "║ AI Filter: %s | Blocks: %d | Allows: %d",
         g_ti_status, g_ti_blocks, g_ti_allows);

   // v5.19: per-stage pipeline status strings (never-fired shows "never")
   string predictTimeStr = (g_ti_predictTime==0) ? "never" : TimeToString(g_ti_predictTime,TIME_SECONDS);
   string patchTimeStr   = (g_ti_patchTime==0)   ? "never" : TimeToString(g_ti_patchTime,TIME_SECONDS);
   string reportTimeStr  = (g_ti_reportTime==0)  ? "never" : TimeToString(g_ti_reportTime,TIME_SECONDS);

   #define COL  "  ║  "
   #define SEP  "╠══════════════════════════╦══════════════════════════╣\n"
   #define TOP  "╔══════════════════════════╦══════════════════════════╗\n"
   #define BOT  "╚══════════════════════════╩══════════════════════════╝\n"
   #define MID  "╠══════════════════════════╩══════════════════════════╣\n"
   #define ENDF "\n"

   string py_live   = g_py_stale ? "⚠️ STALE" : "✅ LIVE";
   string py_lot    = DoubleToString(CalcLotSize(), 2);
   string py_mult   = DoubleToString(PY_GetMultiplier(), 4);
   string py_health = DoubleToString(g_py_health, 1);
   string py_base   = DoubleToString(CalcBaseLot(), 2);
   double equity    = AccountInfoDouble(ACCOUNT_EQUITY);
   string py_steps  = DoubleToString(MathFloor(equity / EquityStep), 0);
   string py_equity = DoubleToString(equity, 2);

   string d = "";

   d += TOP;
   d += "║   UNICORE v5.19 XAUUSD     " + COL + "  MSV=PRIMARY  Manual Exit  ║\n";
   d += SEP;

   d += "║ STATUS: " + g_lastSignalStatus + ENDF;
   if(g_lastError != "")
      d += "║ ERROR:  " + g_lastError + ENDF;
   d += "║ Last:   [" + g_lastSignalStrategy + "] " + g_lastSignalAction
      + COL + "Vote: " + g_lastVote + "\n";
   d += "║ MSV Dir: " + msvDirLabel + "  |  Gate: ⚪ REMOVED\n";
   d += "║ AI ea_id: per-strategy ✅ | snap+pred ✅ | opened_at ✅ (v5.17)\n";
   d += SEP;

   d += "╠══════════════════════╦═════════════════════════╣\n";
   d += "║  🐍 PYTHON BRIDGE    ║  " + py_live + "  (sole lot authority)  \n";
   d += "╠══════════════════════╬═════════════════════════╣\n";
   d += "║ Mode:    " + g_py_mode   + COL + " Mult:   " + py_mult + "x          \n";
   d += "║ Regime:  " + g_py_regime + "                                \n";
   d += "║ Health:  " + py_health   + "/100" + COL + " Equity: $" + py_equity + "  \n";
   d += "║ Steps:   " + py_steps + " × $" + DoubleToString(EquityStep,0)
      + " → base " + py_base + " lots              \n";
   d += "║ Final Lot: " + py_lot    + "  (" + py_base + " × " + py_mult + "x)         \n";
   d += lotGateRow + ENDF;
   d += "║ " + g_py_status + "                                         \n";
   d += SEP;

   d += "╠══════════════════════════════════════════════════╣\n";
   d += "║  🤖 TRADING INTELLIGENCE AI (v5.19)               \n";
   d += tiRow + ENDF;
   d += "║ Predict: " + g_ti_predictStatus + COL + predictTimeStr + "\n";
   d += "║ Patch:   " + g_ti_patchStatus   + COL + patchTimeStr   + "\n";
   d += "║ Report:  " + g_ti_reportStatus  + COL + reportTimeStr  + "\n";
   d += SEP;

   d += "╠══════════════════════════════════════════════════╣\n";
   d += "║  🔢 MAGIC ROUTING                                 \n";
   d += "║  77701=FRAC  77702=OB   77703=TRAP  77704=LIQ    \n";
   d += "║  77705=AMT   77706=LVN  77707=SE    777701=FVG   \n";
   d += "║  88801=UTBOT 88888=MSV                           \n";
   d += SEP;

   d += "║  STRATEGY       Open" + COL + "  STRATEGY       Open  ║\n";
   d += "║  K1 MSV ★       " + IntegerToString(msvOpen)
      + COL + "  A1 UTBOT        " + IntegerToString(utbotOpen) + "      ║\n";
   d += "║  B2 TRAP(INV)   " + IntegerToString(trapOpen)
      + COL + "  C3 OB           " + IntegerToString(obOpen)    + "      ║\n";
   d += "║  D4 FRACTAL     " + IntegerToString(fractalOpen)
      + COL + "  E5 SNIPER       " + IntegerToString(sniperOpen)+ "      ║\n";
   d += "║  F6 FVG         " + IntegerToString(fvgOpen)
      + COL + "  G7 AMT          " + IntegerToString(amtOpen)   + "      ║\n";
   d += "║  H8 LVN         " + IntegerToString(lvnOpen)
      + COL + "  I9 LIQUIDITY    " + IntegerToString(liqOpen)   + "      ║\n";
   d += "║  J0 SMARTENTRY  " + IntegerToString(seOpen)
      + COL + "  TOTAL OPEN:     " + IntegerToString(totalOpen) + "      ║\n";
   d += SEP;

   d += "║  W:" + IntegerToString(g_totalWins)
      + "  +$" + DoubleToString(g_totalWinProfit,2)
      + COL
      + "  L:" + IntegerToString(g_totalLosses)
      + "  -$" + DoubleToString(MathAbs(g_totalLossAmount),2) + "  ║\n";
   d += "║  Net P&L: $" + DoubleToString(g_netProfit,2)
      + COL
      + "  Today: " + IntegerToString(g_todayTradeCount) + "/2000    ║\n";
   d += SEP;

   d += "║  SL:" + DoubleToString(FixedSLPoints,1) + "pt  NoTP  Lookback:" + IntegerToString(LookbackHours)
      + "h" + COL + "  Poll: " + TimeToString(g_lastPollTime,TIME_SECONDS) + "     ║\n";
   d += MID;

   d += "║  🔐 MASK: A1=UTBOT B2=TRAP C3=OB D4=FRAC E5=SNIP F6=FVG G7=AMT H8=LVN I9=LIQ J0=SE K1=MSV\n";
   d += BOT;

   Comment(d);
}
//+------------------------------------------------------------------+