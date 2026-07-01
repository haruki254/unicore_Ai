//+------------------------------------------------------------------+
//|          Trading Intelligence EA — MT5 Expert Advisor             |
//|          Connects to Python AI backend via HTTP                    |
//|          Version 1.0                                               |
//+------------------------------------------------------------------+
#property copyright "Trading Intelligence System"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Indicators\Indicators.mqh>

//── Input Parameters ────────────────────────────────────────────
input string   API_URL        = "http://127.0.0.1:8000";   // Python API URL
input string   API_KEY        = "@Youtube 2017";            // API Secret Key
input double   LOT_SIZE       = 0.01;                       // Lot size
input double   STOP_LOSS_PIPS = 20.0;                       // Stop Loss in pips
input double   TAKE_PROFIT_PIPS = 40.0;                     // Take Profit in pips
input int      CANDLES_M5     = 100;                        // M5 candles to send
input int      CANDLES_M15    = 50;                         // M15 candles to send
input int      CANDLES_H1     = 20;                         // H1 candles to send
input int      CANDLES_H4     = 10;                         // H4 candles to send
input int      CANDLES_D1     = 5;                          // D1 candles to send
input bool     ALLOW_FLIP     = true;                       // Allow AI to flip direction
input bool     LOG_ALL        = true;                       // Log every signal
input int      MAX_SPREAD_POINTS = 30;                      // Max spread in points
input int      MAGIC_NUMBER   = 20240101;                   // EA Magic Number
input string   EA_ID          = "default";                  // EA identifier (FRACTAL/OB/TRAP/etc.)

//── Global objects ───────────────────────────────────────────────
CTrade         Trade;
CPositionInfo  Position;

//── State variables ──────────────────────────────────────────────
string   g_symbol        = Symbol();
int      g_digits        = (int)SymbolInfoInteger(g_symbol, SYMBOL_DIGITS);
double   g_point         = SymbolInfoDouble(g_symbol, SYMBOL_POINT);
bool     g_trade_pending = false;
datetime g_last_signal   = 0;
int      g_trades_today  = 0;
double   g_loss_streak   = 0;
double   g_win_streak    = 0;

//── Per-trade context (stored from /predict response, sent on /trade/update) ──
string   g_snapshot_id   = "";   // snapshot_id returned by /predict
string   g_prediction_id = "";   // prediction_id returned by /predict
string   g_regime        = "";   // regime returned by /predict
string   g_session       = "";   // session returned by /predict
string   g_ea_id         = "";   // resolved from EA_ID input param
double   g_entry_price   = 0.0;  // entry price of the open trade
string   g_direction     = "";   // direction of the open trade

//+------------------------------------------------------------------+
//| Expert initialization                                             |
//+------------------------------------------------------------------+
int OnInit()
{
    Trade.SetExpertMagicNumber(MAGIC_NUMBER);
    Trade.SetDeviationInPoints(10);
    Trade.SetTypeFilling(ORDER_FILLING_FOK);

    // Resolve ea_id: prefer EA_ID input, else derive from MAGIC_NUMBER
    if(EA_ID != "" && EA_ID != "default") {
        g_ea_id = EA_ID;
    } else {
        switch(MAGIC_NUMBER) {
            case 77701:  g_ea_id = "FRACTAL";     break;
            case 77702:  g_ea_id = "OB";           break;
            case 77703:  g_ea_id = "TRAP";         break;
            case 77704:  g_ea_id = "LIQUIDITY";    break;
            case 77705:  g_ea_id = "AMT";          break;
            case 77706:  g_ea_id = "LVN";          break;
            case 77707:  g_ea_id = "SMARTENTRY";   break;
            case 777701: g_ea_id = "FVG";          break;
            case 88801:  g_ea_id = "UTBOT";        break;
            case 88888:  g_ea_id = "MSV";          break;
            default:     g_ea_id = "default";      break;
        }
    }

    Print("===========================================");
    Print("  Trading Intelligence EA v1.0 Starting");
    Print("  Symbol:  ", g_symbol);
    Print("  API:     ", API_URL);
    Print("  EA ID:   ", g_ea_id);
    Print("  Magic:   ", MAGIC_NUMBER);
    Print("===========================================");

    // Verify API connectivity
    string health = HttpGet("/health");
    if(StringFind(health, "healthy") < 0) {
        Alert("⚠️ Cannot reach Trading Intelligence API at ", API_URL);
        Print("Health check failed: ", health);
        return INIT_FAILED;
    }
    Print("✅ API connected: ", health);
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    Print("Trading Intelligence EA stopped. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Expert tick handler                                               |
//+------------------------------------------------------------------+
void OnTick()
{
    // Only act on new bar
    if(!IsNewBar()) return;

    // Check spread
    double spread = SymbolInfoInteger(g_symbol, SYMBOL_SPREAD);
    if(spread > MAX_SPREAD_POINTS) {
        if(LOG_ALL) Print("⏭ Skip: spread too wide (", spread, " pts)");
        return;
    }

    // Check if we already have an open trade
    if(HasOpenTrade()) {
        if(LOG_ALL) Print("⏭ Skip: trade already open");
        return;
    }
}

//+------------------------------------------------------------------+
//| Called externally (or from another EA/script) with a signal       |
//+------------------------------------------------------------------+
void ProcessSignal(string eaSignal)
{
    eaSignal = StringToUpper(eaSignal);
    if(eaSignal != "BUY" && eaSignal != "SELL") {
        Print("Invalid EA signal: ", eaSignal);
        return;
    }

    Print("──────────────────────────────────────────");
    Print("📡 EA Signal: ", eaSignal);

    // Build JSON request
    string json = BuildPredictionRequest(eaSignal);

    // Send to Python API
    string response = HttpPost("/predict", json);
    if(response == "") {
        Print("❌ API request failed — skipping trade");
        return;
    }

    if(LOG_ALL) Print("API Response: ", response);

    // Parse response
    string finalDecision  = ParseJsonString(response, "final_decision");
    string tradeDirection = ParseJsonString(response, "trade_direction");
    double buyProb        = ParseJsonDouble(response, "trader_buy_prob");
    double sellProb       = ParseJsonDouble(response, "trader_sell_prob");
    double riskQuality    = ParseJsonDouble(response, "risk_quality_score");
    string regime         = ParseJsonString(response, "regime");
    bool   isFlip         = (ParseJsonString(response, "is_flip") == "true");
    bool   isBlocked      = (ParseJsonString(response, "is_blocked") == "true");
    string predId         = ParseJsonString(response, "prediction_id");
    string blockReasons   = ParseJsonString(response, "block_reasons");
    double inferenceMs    = ParseJsonDouble(response, "inference_ms");
    string snapshotId     = ParseJsonString(response, "snapshot_id");
    string sessionStr     = ParseJsonString(response, "session");

    // ── Store context globals for use when trade closes ──────────
    g_snapshot_id   = snapshotId;
    g_prediction_id = predId;
    g_regime        = regime;
    g_session       = sessionStr;

    Print("┌─────────────────────────────────────────");
    Print("│  EA Signal:    ", eaSignal);
    Print("│  BUY Prob:     ", DoubleToString(buyProb  * 100, 1), "%");
    Print("│  SELL Prob:    ", DoubleToString(sellProb * 100, 1), "%");
    Print("│  Risk Quality: ", DoubleToString(riskQuality * 100, 1), "%");
    Print("│  Regime:       ", regime);
    Print("│  Flip:         ", isFlip ? "YES" : "NO");
    Print("│  DECISION:  ▶  ", finalDecision);
    if(isBlocked)
        Print("│  Block Reasons:", blockReasons);
    Print("│  Latency:      ", inferenceMs, "ms");
    Print("└─────────────────────────────────────────");

    // ── Execute decision ────────────────────────────────────────
    if(isBlocked || finalDecision == "BLOCK") {
        Print("🚫 BLOCKED by Risk Manager AI");
        return;
    }

    if(!ALLOW_FLIP && isFlip) {
        Print("🔄 FLIP detected but ALLOW_FLIP=false — skipping");
        return;
    }

    if(tradeDirection == "BUY" || finalDecision == "ALLOW_BUY" || finalDecision == "FLIP_TO_BUY") {
        ExecuteTrade(ORDER_TYPE_BUY, predId, eaSignal, isFlip);
    }
    else if(tradeDirection == "SELL" || finalDecision == "ALLOW_SELL" || finalDecision == "FLIP_TO_SELL") {
        ExecuteTrade(ORDER_TYPE_SELL, predId, eaSignal, isFlip);
    }
    else {
        Print("⚠️ Unknown final decision: ", finalDecision);
    }
}

//+------------------------------------------------------------------+
//| Execute a trade order                                             |
//+------------------------------------------------------------------+
void ExecuteTrade(
    ENUM_ORDER_TYPE orderType,
    string predictionId,
    string originalSignal,
    bool   isFlip
) {
    double ask    = SymbolInfoDouble(g_symbol, SYMBOL_ASK);
    double bid    = SymbolInfoDouble(g_symbol, SYMBOL_BID);
    double price  = (orderType == ORDER_TYPE_BUY) ? ask : bid;
    double pipVal = g_point * 10.0;

    double sl, tp;
    if(orderType == ORDER_TYPE_BUY) {
        sl = price - STOP_LOSS_PIPS    * pipVal;
        tp = price + TAKE_PROFIT_PIPS  * pipVal;
    } else {
        sl = price + STOP_LOSS_PIPS    * pipVal;
        tp = price - TAKE_PROFIT_PIPS  * pipVal;
    }

    sl = NormalizeDouble(sl, g_digits);
    tp = NormalizeDouble(tp, g_digits);

    string dirStr  = (orderType == ORDER_TYPE_BUY) ? "BUY" : "SELL";
    string flipStr = isFlip ? " [FLIP]" : "";
    string comment = "TI_" + predictionId + flipStr;

    // ── Store direction and entry price for the close payload ────
    g_direction   = dirStr;
    g_entry_price = price;

    Print("📤 Opening ", dirStr, " @ ", price,
          " SL=", sl, " TP=", tp,
          " Lot=", LOT_SIZE, flipStr);

    bool sent = false;
    if(orderType == ORDER_TYPE_BUY) {
        sent = Trade.Buy(LOT_SIZE, g_symbol, 0, sl, tp, comment);
    } else {
        sent = Trade.Sell(LOT_SIZE, g_symbol, 0, sl, tp, comment);
    }

    if(sent) {
        ulong ticket = Trade.ResultOrder();
        Print("✅ Trade opened: ticket #", ticket);
        g_trades_today++;
    } else {
        Print("❌ Trade failed: ", Trade.ResultRetcodeDescription());
    }
}

//+------------------------------------------------------------------+
//| Called when a trade is closed (from OnTradeTransaction)           |
//+------------------------------------------------------------------+
void OnTradeTransaction(
    const MqlTradeTransaction &trans,
    const MqlTradeRequest     &request,
    const MqlTradeResult      &result
) {
    if(trans.type != TRADE_TRANSACTION_DEAL_ADD) return;

    ulong dealTicket = trans.deal;
    if(!HistoryDealSelect(dealTicket)) return;

    long magic = HistoryDealGetInteger(dealTicket, DEAL_MAGIC);
    if(magic != MAGIC_NUMBER) return;

    long entryType = HistoryDealGetInteger(dealTicket, DEAL_ENTRY);
    if(entryType != DEAL_ENTRY_OUT) return;

    long   posId      = HistoryDealGetInteger(dealTicket, DEAL_POSITION_ID);
    double exitPrice  = HistoryDealGetDouble(dealTicket, DEAL_PRICE);
    double profit     = HistoryDealGetDouble(dealTicket, DEAL_PROFIT);
    long   dealType   = HistoryDealGetInteger(dealTicket, DEAL_TYPE); // type of the CLOSING deal

    // Find the matching entry deal for this position to get the open price
    double entryPrice = 0;
    if(HistorySelectByPosition(posId)) {
        int total = HistoryDealsTotal();
        for(int i = 0; i < total; i++) {
            ulong tk = HistoryDealGetTicket(i);
            if(HistoryDealGetInteger(tk, DEAL_ENTRY) == DEAL_ENTRY_IN) {
                entryPrice = HistoryDealGetDouble(tk, DEAL_PRICE);
                break;
            }
        }
    }

    double pnlPips = 0;
    string outcome = "BREAKEVEN";

    if(profit > 0.01) {
        outcome = "WIN";
        g_win_streak++;
        g_loss_streak = 0;
    } else if(profit < -0.01) {
        outcome = "LOSS";
        g_loss_streak++;
        g_win_streak = 0;
    }

    // The CLOSING deal type is opposite of the position's original direction:
    // if the position was BUY, the closing deal is a SELL, and vice versa.
    bool wasBuy = (dealType == DEAL_TYPE_SELL); // closing sell => position was buy
    double rawPips = wasBuy
                    ? (exitPrice - entryPrice) / (g_point * 10)
                    : (entryPrice - exitPrice) / (g_point * 10);
    pnlPips = rawPips;

    Print("💰 Trade closed: #", dealTicket,
          " | ", outcome,
          " | ", DoubleToString(pnlPips,1), " pips");

    // ── Report to Python API — includes full context for AI learning ──
    string json = StringFormat(
        "{\"mt5_ticket\":%d,\"symbol\":\"%s\","
        "\"direction\":\"%s\",\"entry_price\":%f,"
        "\"exit_price\":%f,\"pnl_pips\":%f,"
        "\"pnl_usd\":%f,\"outcome\":\"%s\","
        "\"was_flipped\":false,"
        "\"ea_id\":\"%s\","
        "\"snapshot_id\":\"%s\","
        "\"prediction_id\":\"%s\","
        "\"regime\":\"%s\","
        "\"session\":\"%s\"}",
        (int)dealTicket,
        g_symbol,
        wasBuy ? "BUY" : "SELL",
        entryPrice,
        exitPrice,
        pnlPips,
        profit,
        outcome,
        g_ea_id,
        g_snapshot_id,
        g_prediction_id,
        g_regime,
        g_session
    );

    string resp = HttpPost("/trade/update", json);
    if(resp == "")
        Print("⚠️ Failed to report trade close to API");
    else {
        // Clear context globals after successful report
        g_snapshot_id   = "";
        g_prediction_id = "";
        g_regime        = "";
        g_session       = "";
        g_direction     = "";
        g_entry_price   = 0.0;
    }
}

//+------------------------------------------------------------------+
//| Build the JSON prediction request                                 |
//+------------------------------------------------------------------+
string BuildPredictionRequest(string eaSignal)
{
    double spread = SymbolInfoInteger(g_symbol, SYMBOL_SPREAD) * g_point;
    double spreadPips = spread / (g_point * 10.0);

    string m5Candles  = BuildCandleArray(PERIOD_M5,  CANDLES_M5);
    string m15Candles = BuildCandleArray(PERIOD_M15, CANDLES_M15);
    string h1Candles  = BuildCandleArray(PERIOD_H1,  CANDLES_H1);
    string h4Candles  = BuildCandleArray(PERIOD_H4,  CANDLES_H4);
    string d1Candles  = BuildCandleArray(PERIOD_D1,  CANDLES_D1);

    // Account state for risk context
    double balance   = AccountInfoDouble(ACCOUNT_BALANCE);
    double equity    = AccountInfoDouble(ACCOUNT_EQUITY);
    double drawdown  = (balance > 0) ? (balance - equity) / balance : 0;

    string json = StringFormat(
        "{"
        "\"symbol\":\"%s\","
        "\"ea_id\":\"%s\","
        "\"ea_signal\":\"%s\","
        "\"price\":%f,"
        "\"spread_pips\":%f,"
        "\"candles_m5\":%s,"
        "\"candles_m15\":%s,"
        "\"candles_h1\":%s,"
        "\"candles_h4\":%s,"
        "\"candles_d1\":%s,"
        "\"risk_context\":{"
        "  \"account_balance\":%f,"
        "  \"account_equity\":%f,"
        "  \"account_drawdown_pct\":%f,"
        "  \"recent_loss_streak\":%d,"
        "  \"recent_win_streak\":%d,"
        "  \"trades_today\":%d"
        "}"
        "}",
        g_symbol,
        g_ea_id,
        eaSignal,
        SymbolInfoDouble(g_symbol, SYMBOL_BID),
        spreadPips,
        m5Candles,
        m15Candles,
        h1Candles,
        h4Candles,
        d1Candles,
        balance,
        equity,
        drawdown,
        (int)g_loss_streak,
        (int)g_win_streak,
        g_trades_today
    );
    return json;
}

//+------------------------------------------------------------------+
//| Build candle JSON array for a given timeframe                     |
//+------------------------------------------------------------------+
string BuildCandleArray(ENUM_TIMEFRAMES tf, int count)
{
    MqlRates rates[];
    int copied = CopyRates(g_symbol, tf, 1, count, rates);
    if(copied <= 0) return "[]";

    string arr = "[";
    for(int i = 0; i < copied; i++) {
        if(i > 0) arr += ",";
        arr += StringFormat(
            "{\"open\":%f,\"high\":%f,\"low\":%f,\"close\":%f,\"volume\":%d}",
            rates[i].open,
            rates[i].high,
            rates[i].low,
            rates[i].close,
            (int)rates[i].tick_volume
        );
    }
    arr += "]";
    return arr;
}

//+------------------------------------------------------------------+
//| HTTP GET request                                                  |
//+------------------------------------------------------------------+
string HttpGet(string endpoint)
{
    string url     = API_URL + endpoint;
    string headers = "X-API-Key: " + API_KEY + "\r\n";
    char   post[], result[];
    string resultHeaders;
    int    timeout = 5000;

    int res = WebRequest("GET", url, headers, timeout, post, result, resultHeaders);
    if(res < 0 || res != 200) {
        Print("HttpGet error: ", res, " URL: ", url);
        return "";
    }
    return CharArrayToString(result);
}

//+------------------------------------------------------------------+
//| HTTP POST request                                                 |
//+------------------------------------------------------------------+
string HttpPost(string endpoint, string body)
{
    string url     = API_URL + endpoint;
    string headers = "Content-Type: application/json\r\nX-API-Key: " + API_KEY + "\r\n";
    char   post[], result[];
    string resultHeaders;
    int    timeout = 10000;

    StringToCharArray(body, post, 0, StringLen(body));

    int res = WebRequest("POST", url, headers, timeout, post, result, resultHeaders);
    if(res < 0 || res != 200) {
        Print("HttpPost error: ", res, " endpoint: ", endpoint);
        Print("Body: ", StringSubstr(body, 0, 200));
        return "";
    }
    return CharArrayToString(result);
}

//+------------------------------------------------------------------+
//| Parse a string value from JSON                                    |
//+------------------------------------------------------------------+
string ParseJsonString(string json, string key)
{
    string search = "\"" + key + "\":\"";
    int start = StringFind(json, search);
    if(start < 0) {
        // Try without quotes (bool/null)
        search = "\"" + key + "\":";
        start  = StringFind(json, search);
        if(start < 0) return "";
        int s2 = start + StringLen(search);
        int e2 = StringFind(json, ",", s2);
        if(e2 < 0) e2 = StringFind(json, "}", s2);
        if(e2 < 0) return "";
        return StringSubstr(json, s2, e2 - s2);
    }
    int s = start + StringLen(search);
    int e = StringFind(json, "\"", s);
    if(e < 0) return "";
    return StringSubstr(json, s, e - s);
}

//+------------------------------------------------------------------+
//| Parse a double value from JSON                                    |
//+------------------------------------------------------------------+
double ParseJsonDouble(string json, string key)
{
    string search = "\"" + key + "\":";
    int start = StringFind(json, search);
    if(start < 0) return 0.0;
    int s = start + StringLen(search);
    int e = StringFind(json, ",", s);
    if(e < 0) e = StringFind(json, "}", s);
    if(e < 0) return 0.0;
    return StringToDouble(StringSubstr(json, s, e - s));
}

//+------------------------------------------------------------------+
//| Check if a new bar has started                                    |
//+------------------------------------------------------------------+
bool IsNewBar()
{
    static datetime lastBar = 0;
    datetime current = iTime(g_symbol, PERIOD_M5, 0);
    if(current != lastBar) {
        lastBar = current;
        return true;
    }
    return false;
}

//+------------------------------------------------------------------+
//| Check if EA has an open trade                                     |
//+------------------------------------------------------------------+
bool HasOpenTrade()
{
    for(int i = 0; i < PositionsTotal(); i++) {
        if(Position.SelectByIndex(i)) {
            if(Position.Magic() == MAGIC_NUMBER &&
               Position.Symbol() == g_symbol)
                return true;
        }
    }
    return false;
}