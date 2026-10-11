# Nifty monthly RSI scanner

Scans every NSE-listed stock (EQ/BE series, ~2,500) for **monthly RSI(14) above 70** on the current month's candle, and adds each hit's weekly RSI(14).

- **Dashboard:** GitHub Pages serves `docs/`.
- **Updates:** `.github/workflows/scan.yml` runs every weekday at 16:30 IST, and on demand from Actions → *Monthly RSI scan* → *Run workflow*.
- **Run locally:** `pip install -r requirements.txt && python scanner.py` (see `python scanner.py --help`).

RSI is Wilder's (same as TradingView's `ta.rsi`). Prices from Yahoo Finance, index lists from NSE. Not investment advice.
