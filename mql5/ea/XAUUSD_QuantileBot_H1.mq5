//+------------------------------------------------------------------+
//| XAUUSD_QuantileBot_H1.mq5                                        |
//|                                                                    |
//| Does NOT compute features or run any model itself. Every H1 bar   |
//| close it exports recent H1/H4/D1 bars to MQL5/Files/, then reads  |
//| a signal file written back by mql5/inference_service/live_inference.py |
//| (running as a separate, continuously-running Python process on    |
//| this same machine) and acts on it. See that file for why this     |
//| file-bridge exists instead of an ONNX/native-MQL5 model — this EA |
//| stays intentionally simple: exporting bars, reading a signal, and |
//| order/risk management are the only things it does.                |
//|                                                                    |
//| NOT COMPILE-VERIFIED — written without access to MetaEditor.      |
//| Before ANY live/demo use:                                         |
//|   1. Compile in MetaEditor, fix whatever the compiler flags.      |
//|   2. Run with the Python service OFF first — confirm the CSV bar  |
//|      exports land in MQL5/Files/ and match the expected columns.  |
//|   3. Start the Python service, confirm it reads those files and   |
//|      writes xauusd_h1_signal.csv without errors.                  |
//|   4. Only then let the EA place real orders — start on the demo   |
//|      account, tiny size, watched closely for a few real signals.  |
//+------------------------------------------------------------------+
#property copyright ""
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

input group "=== Symbol & risk ==="
input string InpSymbol            = "XAUUSD";   // EXACT broker symbol — verify in Market Watch first, may need a suffix
input double InpRiskPct           = 0.05;       // 5% fixed risk per trade — deliberate, do not soften
input double InpSLAtrMult         = 1.5;        // must match SL_ATR_MULT in live_inference.py

input group "=== Barriers ==="
input int    InpMaxHorizonBars    = 24;         // vertical barrier: force-close if still open after this many H1 bars

input group "=== File bridge ==="
input int    InpBarsToExport      = 60;         // H1 bars written each cycle — the Python service only needs a handful of recent ones
input int    InpH4BarsToExport    = 60;
input int    InpD1BarsToExport    = 60;
input int    InpSignalMaxAgeMin   = 90;         // reject a signal whose computed_at is older than this — Python service down/stalled

input group "=== Misc ==="
input int    InpMagicNumber       = 20260819;
input int    InpPollSeconds       = 15;

CTrade trade;

datetime g_last_h1_bar_time = 0;

//+------------------------------------------------------------------+
int OnInit()
  {
   trade.SetExpertMagicNumber(InpMagicNumber);
   if(!SymbolSelect(InpSymbol, true))
     {
      Print("ERROR: could not select symbol '", InpSymbol, "' — check the exact name in Market Watch.");
      return INIT_FAILED;
     }
   EventSetTimer(InpPollSeconds);
   Print("XAUUSD_QuantileBot_H1 initialized. Symbol=", InpSymbol, " Risk=", InpRiskPct * 100, "%");
   return INIT_SUCCEEDED;
  }

void OnDeinit(const int reason)
  {
   EventKillTimer();
  }

void OnTick()
  {
   CheckNewBarAndAct();
  }

void OnTimer()
  {
   CheckNewBarAndAct();
   ManageOpenPosition(); // checked every poll too, not just on a new bar, so the vertical barrier can't overshoot by much
  }

//+------------------------------------------------------------------+
//| Fires once per new H1 bar: export data, manage any open position,|
//| then look for a fresh entry signal.                               |
//+------------------------------------------------------------------+
void CheckNewBarAndAct()
  {
   datetime current_forming_bar = iTime(InpSymbol, PERIOD_H1, 0);
   if(current_forming_bar == g_last_h1_bar_time)
      return;
   g_last_h1_bar_time = current_forming_bar;

   datetime closed_bar_time = iTime(InpSymbol, PERIOD_H1, 1);
   Print("New H1 bar. Closed bar time: ", TimeToString(closed_bar_time, TIME_DATE | TIME_MINUTES));

   ExportBars(PERIOD_H1, InpBarsToExport, "xauusd_h1_bars.csv");
   ExportBars(PERIOD_H4, InpH4BarsToExport, "xauusd_h4_bars.csv");
   ExportBars(PERIOD_D1, InpD1BarsToExport, "xauusd_d1_bars.csv");

   ManageOpenPosition();
   TryReadSignalAndEnter(closed_bar_time);
  }

//+------------------------------------------------------------------+
//| Writes (time,open,high,low,close,spread_price) for the last N    |
//| CLOSED bars of the given timeframe — index 1..N, never 0 (the     |
//| still-forming current bar), matching how live_inference.py treats |
//| higher-timeframe context as only-fully-closed-bars.               |
//+------------------------------------------------------------------+
void ExportBars(ENUM_TIMEFRAMES tf, int count, string filename)
  {
   MqlRates rates[];
   int copied = CopyRates(InpSymbol, tf, 1, count, rates); // start at shift 1: skip the forming bar
   if(copied <= 0)
     {
      Print("WARNING: CopyRates returned ", copied, " bars for ", EnumToString(tf), " — export skipped this cycle.");
      return;
     }

   double point = SymbolInfoDouble(InpSymbol, SYMBOL_POINT);
   int fh = FileOpen(filename, FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
   if(fh == INVALID_HANDLE)
     {
      Print("ERROR: could not open ", filename, " for writing, error ", GetLastError());
      return;
     }

   FileWrite(fh, "time", "open", "high", "low", "close", "spread_price");
   for(int i = 0; i < copied; i++)
     {
      double spread_price = rates[i].spread * point;
      FileWrite(fh, TimeToString(rates[i].time, TIME_DATE | TIME_SECONDS),
                DoubleToString(rates[i].open, _Digits), DoubleToString(rates[i].high, _Digits),
                DoubleToString(rates[i].low, _Digits), DoubleToString(rates[i].close, _Digits),
                DoubleToString(spread_price, _Digits));
     }
   FileClose(fh);
  }

//+------------------------------------------------------------------+
//| Vertical barrier: force-close a position that's been open longer  |
//| than InpMaxHorizonBars H1 bars, at market — mirrors the backtest's|
//| triple-barrier timeout, which never leaves a trade open forever.  |
//+------------------------------------------------------------------+
void ManageOpenPosition()
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;
      if(!PositionSelectByTicket(ticket))
         continue;
      if(PositionGetString(POSITION_SYMBOL) != InpSymbol || PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
         continue;

      datetime entry_time = (datetime)PositionGetInteger(POSITION_TIME);
      double hours_open = (double)(TimeCurrent() - entry_time) / 3600.0;
      if(hours_open >= InpMaxHorizonBars)
        {
         Print("Vertical barrier reached (", DoubleToString(hours_open, 1), "h open) — closing ticket ", ticket, " at market.");
         trade.PositionClose(ticket);
        }
     }
  }

//+------------------------------------------------------------------+
//| Returns true if we already have an open position on this symbol   |
//| under our magic number — idempotency guard against duplicate      |
//| entries (e.g. if the EA restarts and re-processes the same bar).  |
//+------------------------------------------------------------------+
bool HasOpenPosition()
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;
      if(!PositionSelectByTicket(ticket))
         continue;
      if(PositionGetString(POSITION_SYMBOL) == InpSymbol && PositionGetInteger(POSITION_MAGIC) == InpMagicNumber)
         return true;
     }
   return false;
  }

//+------------------------------------------------------------------+
//| Reads xauusd_h1_signal.csv, validates it's fresh and for the bar  |
//| we just closed, and enters if it says BUY.                        |
//+------------------------------------------------------------------+
void TryReadSignalAndEnter(datetime closed_bar_time)
  {
   if(HasOpenPosition())
      return; // one position at a time, per the backtested design

   int fh = FileOpen("xauusd_h1_signal.csv", FILE_READ | FILE_CSV | FILE_ANSI, ',');
   if(fh == INVALID_HANDLE)
     {
      Print("No signal file yet (Python service may not have run this bar) — skipping entry this bar.");
      return;
     }

   // header row — read and discard 7 fields
   for(int i = 0; i < 7 && !FileIsEnding(fh); i++)
      FileReadString(fh);

   if(FileIsEnding(fh))
     {
      FileClose(fh);
      Print("Signal file has no data row yet — skipping.");
      return;
     }

   long   bar_time_unix    = (long)StringToInteger(FileReadString(fh));
   string action           = FileReadString(fh);
   double sl_price         = StringToDouble(FileReadString(fh));
   double tp_price         = StringToDouble(FileReadString(fh));
   double meta_proba       = StringToDouble(FileReadString(fh));
   double pred_tp_return   = StringToDouble(FileReadString(fh));
   long   computed_at_unix = (long)StringToInteger(FileReadString(fh));
   FileClose(fh);

   datetime signal_bar_time = (datetime)bar_time_unix;
   if(signal_bar_time != closed_bar_time)
     {
      Print("Signal is for a different bar (signal=", TimeToString(signal_bar_time),
            " expected=", TimeToString(closed_bar_time), ") — Python service is behind, skipping entry.");
      return;
     }

   double age_minutes = (double)(TimeCurrent() - (datetime)computed_at_unix) / 60.0;
   if(age_minutes > InpSignalMaxAgeMin)
     {
      Print("Signal is stale (", DoubleToString(age_minutes, 1), " min old) — Python service may be down, skipping entry.");
      return;
     }

   Print("Signal: action=", action, " meta_proba=", DoubleToString(meta_proba, 3),
         " pred_tp_return=", DoubleToString(pred_tp_return, 5));

   if(action != "BUY")
      return;

   EnterLong(sl_price, tp_price);
  }

//+------------------------------------------------------------------+
//| Sizes the position for InpRiskPct of current equity given the SL  |
//| distance, then sends a market buy with SL/TP embedded — no        |
//| separate polling loop manages the exit, the broker does.          |
//+------------------------------------------------------------------+
void EnterLong(double sl_price, double tp_price)
  {
   MqlTick tick;
   if(!SymbolInfoTick(InpSymbol, tick))
     {
      Print("ERROR: could not get current tick for ", InpSymbol);
      return;
     }
   double entry_price = tick.ask;

   double sl_distance = entry_price - sl_price;
   if(sl_distance <= 0)
     {
      Print("ERROR: non-positive SL distance (entry=", entry_price, " sl=", sl_price, ") — refusing to size a trade on this.");
      return;
     }

   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double risk_amount = equity * InpRiskPct;

   double tick_value = SymbolInfoDouble(InpSymbol, SYMBOL_TRADE_TICK_VALUE);
   double tick_size  = SymbolInfoDouble(InpSymbol, SYMBOL_TRADE_TICK_SIZE);
   if(tick_size <= 0)
     {
      Print("ERROR: invalid tick size for ", InpSymbol);
      return;
     }
   double value_per_price_unit_per_lot = tick_value / tick_size;

   double raw_lots = risk_amount / (sl_distance * value_per_price_unit_per_lot);

   double vol_min  = SymbolInfoDouble(InpSymbol, SYMBOL_VOLUME_MIN);
   double vol_max  = SymbolInfoDouble(InpSymbol, SYMBOL_VOLUME_MAX);
   double vol_step = SymbolInfoDouble(InpSymbol, SYMBOL_VOLUME_STEP);
   double lots = MathFloor(raw_lots / vol_step) * vol_step;
   lots = MathMax(vol_min, MathMin(vol_max, lots));

   if(lots < vol_min)
     {
      Print("Computed lot size (", DoubleToString(raw_lots, 4), ") rounds below the broker minimum (",
            vol_min, ") — refusing to oversize the trade by trading the minimum anyway. Skipping.");
      return;
     }

   Print("Entering LONG: equity=", DoubleToString(equity, 2), " risk_amount=", DoubleToString(risk_amount, 2),
         " sl_distance=", DoubleToString(sl_distance, 3), " lots=", DoubleToString(lots, 2),
         " sl=", DoubleToString(sl_price, 3), " tp=", DoubleToString(tp_price, 3));

   if(!trade.Buy(lots, InpSymbol, 0.0, sl_price, tp_price, "quantile_h1"))
      Print("ERROR: order send failed, retcode=", trade.ResultRetcode(), " ", trade.ResultRetcodeDescription());
  }
//+------------------------------------------------------------------+
