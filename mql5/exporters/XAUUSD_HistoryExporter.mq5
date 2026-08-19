//+------------------------------------------------------------------+
//| XAUUSD_HistoryExporter.mq5                                       |
//| Run once (or periodically) as a Script inside the MT5 terminal.  |
//| Exports M5 OHLC bars with per-bar bid/ask/spread stats derived   |
//| from tick data, to a CSV in MQL5/Files. No Python<->MT5 live     |
//| connection is used anywhere in this project — this script is    |
//| the entire data path from broker to disk.                       |
//+------------------------------------------------------------------+
#property script_show_inputs

input string   InpSymbol      = "XAUUSD";     // exact broker symbol name — verify in the log on first run
input datetime InpStartDate   = D'2020.01.01 00:00';
input datetime InpEndDate     = D'2026.08.17 00:00';
input int      InpChunkDays   = 30;           // bounds memory: ticks are pulled per chunk, not for the whole range at once
input string   InpOutFileName = "XAUUSD_M5_export.csv";

//+------------------------------------------------------------------+
void PrintSymbolCandidates()
  {
   Print("Symbols containing 'XAU' available on this account (verify InpSymbol matches one of these exactly):");
   int total = SymbolsTotal(false);
   for(int i = 0; i < total; i++)
     {
      string name = SymbolName(i, false);
      if(StringFind(name, "XAU") >= 0)
         Print("  ", name);
     }
  }

//+------------------------------------------------------------------+
int OnStart()
  {
   PrintSymbolCandidates();

   if(!SymbolSelect(InpSymbol, true))
     {
      Print("ERROR: could not select symbol '", InpSymbol, "'. Check the exact name against the list above.");
      return -1;
     }

   double point = SymbolInfoDouble(InpSymbol, SYMBOL_POINT);
   int    fh = FileOpen(InpOutFileName, FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
   if(fh == INVALID_HANDLE)
     {
      Print("ERROR: could not open output file '", InpOutFileName, "', error ", GetLastError());
      return -1;
     }

   FileWrite(fh, "time", "open", "high", "low", "close", "tick_volume", "broker_spread_points",
             "real_volume", "bid_open", "ask_open", "bid_close", "ask_close",
             "spread_mean_points", "spread_min_points", "spread_max_points", "tick_count");

   long chunk_seconds = (long)InpChunkDays * 86400;
   int  bar_seconds    = PeriodSeconds(PERIOD_M5);
   long total_bars_written = 0;

   for(datetime chunk_start = InpStartDate; chunk_start < InpEndDate; chunk_start += (int)chunk_seconds)
     {
      datetime chunk_end = chunk_start + (int)chunk_seconds;
      if(chunk_end > InpEndDate)
         chunk_end = InpEndDate;

      MqlRates rates[];
      int rates_count = CopyRates(InpSymbol, PERIOD_M5, chunk_start, chunk_end, rates);
      if(rates_count <= 0)
        {
         Print("No rates for chunk ", TimeToString(chunk_start), " - ", TimeToString(chunk_end), ", skipping.");
         continue;
        }

      MqlTick ticks[];
      // COPY_TICKS_INFO = bid/ask ticks only, no trade/volume ticks — that's all we need for spread reconstruction.
      int ticks_count = CopyTicksRange(InpSymbol, ticks, COPY_TICKS_INFO,
                                        (ulong)chunk_start * 1000, (ulong)chunk_end * 1000);
      if(ticks_count <= 0)
         Print("WARNING: no ticks for chunk ", TimeToString(chunk_start), " - ", TimeToString(chunk_end),
               " — spread stats for this chunk's bars will be empty.");

      int tick_ptr = 0; // ticks[] is chronologically ordered by CopyTicksRange — single forward sweep is valid

      for(int i = 0; i < rates_count; i++)
        {
         datetime bar_start = rates[i].time;
         datetime bar_end   = bar_start + bar_seconds;

         double bid_open = 0, ask_open = 0, bid_close = 0, ask_close = 0;
         double spread_sum = 0, spread_min = DBL_MAX, spread_max = -DBL_MAX;
         int    tick_count_in_bar = 0;

         while(tick_ptr < ticks_count && ticks[tick_ptr].time < bar_start)
            tick_ptr++; // catch up to this bar's window (handles chunk boundary drift)

         int j = tick_ptr;
         while(j < ticks_count && ticks[j].time < bar_end)
           {
            double bid = ticks[j].bid;
            double ask = ticks[j].ask;
            if(bid > 0 && ask > 0)
              {
               double spread_points = (ask - bid) / point;
               if(tick_count_in_bar == 0)
                 {
                  bid_open = bid;
                  ask_open = ask;
                 }
               bid_close = bid;
               ask_close = ask;
               spread_sum += spread_points;
               if(spread_points < spread_min) spread_min = spread_points;
               if(spread_points > spread_max) spread_max = spread_points;
               tick_count_in_bar++;
              }
            j++;
           }

         double spread_mean = (tick_count_in_bar > 0) ? spread_sum / tick_count_in_bar : 0;
         if(tick_count_in_bar == 0)
           {
            spread_min = 0;
            spread_max = 0;
           }

         FileWrite(fh, TimeToString(rates[i].time, TIME_DATE | TIME_SECONDS),
                   DoubleToString(rates[i].open, _Digits), DoubleToString(rates[i].high, _Digits),
                   DoubleToString(rates[i].low, _Digits), DoubleToString(rates[i].close, _Digits),
                   (long)rates[i].tick_volume, rates[i].spread, (long)rates[i].real_volume,
                   DoubleToString(bid_open, _Digits), DoubleToString(ask_open, _Digits),
                   DoubleToString(bid_close, _Digits), DoubleToString(ask_close, _Digits),
                   DoubleToString(spread_mean, 2), DoubleToString(spread_min, 2), DoubleToString(spread_max, 2),
                   tick_count_in_bar);

         total_bars_written++;
         tick_ptr = j; // advance past this bar's ticks so the next bar's sweep doesn't re-scan them
        }

      Print("Chunk ", TimeToString(chunk_start), " - ", TimeToString(chunk_end), ": wrote ", rates_count, " bars.");
     }

   FileClose(fh);
   Print("Done. Total bars written: ", total_bars_written, ". File: MQL5/Files/", InpOutFileName);
   return 0;
  }
