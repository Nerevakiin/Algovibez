#!/usr/bin/env python3
"""
Very small EMA-cross algorithmic-trading skeleton.

Usage examples
--------------
# back-test
python ema_cross_bot.py --ticker AAPL --from 2018-01-01 --to 2023-01-01 --plot

# paper-trade tomorrow's open (Alpaca)
export APCA_API_KEY_ID=*****
export APCA_API_SECRET_KEY=*****
python ema_cross_bot.py --ticker AAPL --live
"""
import argparse
import datetime as dt
import os
import backtrader as bt
import yfinance as yf
import alpaca_trade_api as tradeapi

# ---------- strategy logic ---------------------------------------------------
class EMACross(bt.Strategy):
    params = dict(fast=20, slow=50)

    def __init__(self):
        self.fast_ema = bt.ind.EMA(period=self.p.fast)
        self.slow_ema = bt.ind.EMA(period=self.p.slow)
        self.crossup = bt.ind.CrossUp(self.fast_ema, self.slow_ema)
        self.crossdn = bt.ind.CrossDown(self.fast_ema, self.slow_ema)

    def next(self):
        if self.crossup:
            self.order_target_percent(target=0.95)  # 100% long (leave 5% cash)
        elif self.crossdn:
            self.order_target_percent(target=0.0)   # flat

# ---------- helpers ----------------------------------------------------------
def backtest(ticker, start, end, plot=False):
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100_000)
    cerebro.broker.setcommission(commission=0.001)  # 0.1% per side

    data = yf.download(ticker, start=start, end=end, auto_adjust=True)
    feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(feed)
    cerebro.addstrategy(EMACross)
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")

    print("Starting portfolio value: %.2f" % cerebro.broker.getvalue())
    results = cerebro.run()
    print("Final portfolio value  : %.2f" % cerebro.broker.getvalue())

    strat = results[0]
    print("CAGR : %.2f %%" % (strat.analyzers.returns.get_analysis()["rnorm100"]))
    print("MaxDD: %.2f %%" % (strat.analyzers.dd.get_analysis()["max"]["drawdown"]))

    if plot:
        cerebro.plot(style="candlestick")

# ---------- live / paper -----------------------------------------------------
class AlpacaPaperTrader:
    def __init__(self, key, secret, base_url="https://paper-api.alpaca.markets"):
        self.api = tradeapi.REST(key, secret, base_url, api_version="v2")

    def run(self, ticker):
        # fetch last 50+ days of daily data
        end = dt.datetime.now()
        start = end - dt.timedelta(days=80)
        data = yf.download(ticker, start=start, end=end, auto_adjust=True)

        # compute latest EMA values
        fast = 20
        slow = 50
        ema_fast = data["Close"].ewm(span=fast).mean()[-1]
        ema_slow = data["Close"].ewm(span=slow).mean()[-1]
        prev_fast = data["Close"].ewm(span=fast).mean()[-2]
        prev_slow = data["Close"].ewm(span=slow).mean()[-2]

        # determine signal
        signal = None
        if prev_fast <= prev_slow and ema_fast > ema_slow:
            signal = "BUY"
        elif prev_fast >= prev_slow and ema_fast < ema_slow:
            signal = "SELL"

        if signal is None:
            print("No cross today – doing nothing.")
            return

        # place market-on-open order
        account = self.api.get_account()
        cash = float(account.cash)
        last_price = data["Close"][-1]
        if signal == "BUY":
            qty = int(cash * 0.95 / last_price)
            if qty > 0:
                self.api.submit_order(
                    symbol=ticker,
                    qty=qty,
                    side="buy",
                    type="market",
                    time_in_force="opg",
                )
                print(f"Submitted BUY order for {qty} shares at ~{last_price}")
        else:  # SELL
            positions = self.api.list_positions()
            for pos in positions:
                if pos.symbol == ticker:
                    self.api.submit_order(
                        symbol=ticker,
                        qty=pos.qty,
                        side="sell",
                        type="market",
                        time_in_force="opg",
                    )
                    print(f"Submitted SELL order for {pos.qty} shares")

# ---------- main -------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--from", dest="start", type=lambda s: dt.datetime.strptime(s, "%Y-%m-%d"))
    parser.add_argument("--to", dest="end", type=lambda s: dt.datetime.strptime(s, "%Y-%m-%d"))
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--live", action="store_true", help="run once in live/paper mode")
    args = parser.parse_args()

    if args.live:
        key = os.getenv("APCA_API_KEY_ID")
        secret = os.getenv("APCA_API_SECRET_KEY")
        if not key or not secret:
            raise RuntimeError("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY env vars")
        AlpacaPaperTrader(key, secret).run(args.ticker.upper())
    else:
        if not args.start or not args.end:
            parser.error("--from and --to required for back-test")
        backtest(args.ticker.upper(), args.start, args.end, args.plot)