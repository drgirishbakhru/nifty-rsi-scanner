#!/usr/bin/env python3
"""
Monthly RSI(14) > 70 scanner for NSE stocks.

Universes (pulled live from NSE archives):
  total   - Nifty Total Market (~750 stocks: Nifty 500 + Microcap 250)  [default]
  nifty500, midsmall400, smallcap250, microcap250
  all     - every NSE-listed equity in EQ/BE series (~2,300+)

RSI is Wilder's RSI on monthly closes (same as TradingView's ta.rsi), using the
current (still-forming) monthly candle as the latest bar.

Usage:
  .venv/bin/python monthly_rsi_scanner.py                  # Total Market, RSI > 70
  .venv/bin/python monthly_rsi_scanner.py --universe all
  .venv/bin/python monthly_rsi_scanner.py --fresh          # only fresh crosses (prev month <= 70)
  .venv/bin/python monthly_rsi_scanner.py --level 60 --length 14
"""
import argparse
import io
import re
import sys
from datetime import datetime

import pandas as pd
import requests
import yfinance as yf

NSE = "https://archives.nseindia.com/content"
UNIVERSES = {
    "total": f"{NSE}/indices/ind_niftytotalmarket_list.csv",
    "nifty500": f"{NSE}/indices/ind_nifty500list.csv",
    "midsmall400": f"{NSE}/indices/ind_niftymidsmallcap400list.csv",
    "smallcap250": f"{NSE}/indices/ind_niftysmallcap250list.csv",
    "microcap250": f"{NSE}/indices/ind_niftymicrocap250_list.csv",
    "all": f"{NSE}/equities/EQUITY_L.csv",
}
HEADERS = {"User-Agent": "Mozilla/5.0"}


def read_list(name):
    r = requests.get(UNIVERSES[name], headers=HEADERS, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.strip().upper() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].str.strip()
    return df


def load_universe(name):
    df = read_list(name)
    if name == "all":
        df = df[df["SERIES"].isin(["EQ", "BE"])]
        return dict(zip(df["SYMBOL"], df["NAME OF COMPANY"]))
    return dict(zip(df["SYMBOL"], df["COMPANY NAME"]))


def rsi_wilder(close, length=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)
    return rsi.where(avg_loss != 0, 100.0)


def fetch_monthly(symbols, chunk=100, interval="1mo", period="max", suffix=".NS"):
    """Download bars in chunks (monthly, full history by default); returns {symbol: close Series}."""
    out = {}
    for i in range(0, len(symbols), chunk):
        batch = symbols[i:i + chunk]
        tickers = [f"{s}{suffix}" for s in batch]
        print(f"  downloading {i + 1}-{i + len(batch)} of {len(symbols)}...", file=sys.stderr)
        data = yf.download(tickers, period=period, interval=interval, group_by="ticker",
                           auto_adjust=False, progress=False, threads=True)
        for s, t in zip(batch, tickers):
            try:
                close = data[t]["Close"].dropna() if len(tickers) > 1 else data["Close"].dropna()
            except KeyError:
                continue
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            if len(close):
                out[s] = close
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", choices=UNIVERSES, default="total")
    ap.add_argument("--length", type=int, default=14)
    ap.add_argument("--level", type=float, default=70)
    ap.add_argument("--fresh", action="store_true", help="only stocks whose previous month RSI was <= level")
    ap.add_argument("--min-bars", type=int, default=24, help="skip stocks with fewer monthly bars")
    ap.add_argument("--dashboard", metavar="DIR",
                    help="scan all NSE stocks (+ BSE-only stocks) and write rows.json + summary.json for the dashboard")
    ap.add_argument("--bse-min-cr", type=float, default=1000,
                    help="dashboard: minimum market cap in crore for BSE-only stocks (default 1000)")
    args = ap.parse_args()
    if args.dashboard:
        args.universe, args.fresh = "all", False

    names = load_universe(args.universe)
    symbols = sorted(names)
    print(f"Universe '{args.universe}': {len(symbols)} symbols", file=sys.stderr)

    closes = fetch_monthly(symbols)
    rows, skipped = scan(closes, names, args)

    missing = len(symbols) - len(closes)
    print(f"Fetched {len(closes)} | no data {missing} | too short {skipped}", file=sys.stderr)

    if args.dashboard:
        if len(closes) < 0.8 * len(symbols):
            sys.exit(f"Only {len(closes)} of {len(symbols)} symbols downloaded; keeping the previous results.")
        write_dashboard(args, rows, len(symbols), len(closes), skipped)
        return
    if not rows:
        print(f"No stocks with monthly RSI({args.length}) > {args.level}.")
        return
    res = pd.DataFrame(rows).sort_values(["FreshCross", "RSI_M"], ascending=[False, False])
    pd.set_option("display.width", 200, "display.max_rows", None, "display.max_colwidth", 40)
    print(res.to_string(index=False))
    print(f"\n{len(res)} stocks | {(res.FreshCross == 'YES').sum()} fresh crosses this month")

    fname = f"monthly_rsi_{args.universe}_{datetime.now():%Y%m%d}.csv"
    res.to_csv(fname, index=False)
    print(f"Saved -> {fname}")


def scan(closes, names, args):
    """Monthly RSI hits from {symbol: close Series}; returns (rows, number skipped as too short)."""
    rows, skipped = [], 0
    for s, close in closes.items():
        if len(close) < max(args.min_bars, args.length + 2):
            skipped += 1
            continue
        rsi = rsi_wilder(close, args.length)
        cur, prev = rsi.iloc[-1], rsi.iloc[-2]
        if pd.isna(cur) or cur <= args.level:
            continue
        fresh = pd.notna(prev) and prev <= args.level
        if args.fresh and not fresh:
            continue
        rows.append({
            "Symbol": s,
            "Company": names.get(s, ""),
            "Close": round(float(close.iloc[-1]), 2),
            "RSI_M": round(float(cur), 2),
            "RSI_M_prev": round(float(prev), 2),
            "FreshCross": "YES" if fresh else "",
            "MonthChg%": round(float(close.iloc[-1] / close.iloc[-2] - 1) * 100, 2),
        })
    return rows, skipped


def yahoo_screen(exchange, min_mcap):
    """Every Yahoo screener quote on an exchange ('BSE', 'NSI') with market cap >= min_mcap (INR)."""
    from yfinance import EquityQuery as Q
    q = Q("and", [Q("eq", ["exchange", exchange]), Q("gte", ["intradaymarketcap", min_mcap])])
    out, offset = [], 0
    while True:
        r = yf.screen(q, size=250, offset=offset, sortField="intradaymarketcap", sortAsc=False)
        out += r.get("quotes", [])
        offset += 250
        if not r.get("quotes") or offset >= r.get("total", 0):
            return out


def _norm(name):
    return re.sub(r"[^a-z0-9]", "", re.sub(r"\b(limited|ltd)\b", "", str(name).lower()))


def bse_only_universe(min_cr):
    """BSE stocks with market cap >= min_cr crore that are not listed on NSE: {symbol: (name, mcap_cr)}."""
    nse = read_list("all")
    nse_syms = set(nse["SYMBOL"])
    nse_names = set(nse["NAME OF COMPANY"].map(_norm))
    out = {}
    for x in yahoo_screen("BSE", min_cr * 1e7):
        sym = x["symbol"].removesuffix(".BO")
        name = x.get("longName") or x.get("shortName") or sym
        if sym in nse_syms or _norm(name) in nse_names:
            continue
        if re.search(r"\b(reit|invit|trust|fund|etf)\b", name, re.I):
            continue  # REITs/InvITs/ETFs, not stocks
        out[sym] = (name, round(x.get("marketCap", 0) / 1e7))
    return out


def write_dashboard(args, rows, universe_n, fetched_n, skipped_n):
    """Tag each hit with its Nifty index membership and write JSON for the dashboard."""
    import json
    import os
    tm = read_list("total")
    industry = dict(zip(tm["SYMBOL"], tm["INDUSTRY"]))
    n500 = set(read_list("nifty500")["SYMBOL"])
    micro = set(read_list("microcap250")["SYMBOL"])

    # BSE-only stocks above the market-cap floor. A failure here keeps the NSE results.
    bse_rows, bse_n, bse_fetched, bse_skipped, bse_ok = [], 0, 0, 0, True
    try:
        bse = bse_only_universe(args.bse_min_cr)
        bse_n = len(bse)
        print(f"BSE-only universe (>= {args.bse_min_cr:.0f} cr): {bse_n} symbols", file=sys.stderr)
        bse_closes = fetch_monthly(sorted(bse), suffix=".BO")
        bse_fetched = len(bse_closes)
        bse_rows, bse_skipped = scan(bse_closes, {k: v[0] for k, v in bse.items()}, args)
    except Exception as e:  # noqa: BLE001
        print(f"BSE scan failed: {e}", file=sys.stderr)
        bse, bse_ok = {}, False

    nse_mcap = {}
    try:
        nse_mcap = {x["symbol"].removesuffix(".NS"): round(x.get("marketCap", 0) / 1e7)
                    for x in yahoo_screen("NSI", 1)}
    except Exception as e:  # noqa: BLE001
        print(f"NSE market caps unavailable: {e}", file=sys.stderr)

    print("Fetching weekly bars for the hits...", file=sys.stderr)
    weekly = {("NSE", k): v for k, v in fetch_monthly([r["Symbol"] for r in rows], interval="1wk", period="10y").items()}
    if bse_rows:
        weekly.update({("BSE", k): v for k, v in fetch_monthly([r["Symbol"] for r in bse_rows], interval="1wk",
                                                               period="10y", suffix=".BO").items()})
    out = []
    for exch, r in [("NSE", r) for r in rows] + [("BSE", r) for r in bse_rows]:
        s = r["Symbol"]
        wc = weekly.get((exch, s))
        w = rsi_wilder(wc, args.length) if wc is not None and len(wc) > args.length + 1 else None
        rsi_w = round(float(w.iloc[-1]), 2) if w is not None and pd.notna(w.iloc[-1]) else None
        if exch == "BSE":
            group, mcap = "BSE only", bse[s][1]
        else:
            group = "Nifty 500" if s in n500 else "Microcap 250" if s in micro else "Outside Total Market"
            mcap = nse_mcap.get(s)
        out.append({
            "symbol": s, "exchange": exch, "tv": f"{exch}:{s}", "company": r["Company"],
            "industry": industry.get(s, "") if exch == "NSE" else "",
            "index": group, "mcap_cr": mcap, "close": r["Close"], "rsi": r["RSI_M"], "rsi_prev": r["RSI_M_prev"],
            "fresh": bool(r["FreshCross"]), "month_chg_pct": r["MonthChg%"],
            "rsi_w": rsi_w, "weekly_above": rsi_w is not None and rsi_w > args.level,
        })
    out.sort(key=lambda r: -r["rsi"])
    now = datetime.now().astimezone()
    summary = [{
        "scanned_at": now.isoformat(timespec="seconds"),
        "month": now.strftime("%Y-%m"),
        "rsi_length": args.length, "level": args.level, "min_bars": args.min_bars,
        "universe": universe_n, "fetched": fetched_n, "too_short": skipped_n,
        "total_market_size": len(tm),
        "bse_ok": bse_ok, "bse_min_cr": args.bse_min_cr, "bse_universe": bse_n,
        "bse_fetched": bse_fetched, "bse_too_short": bse_skipped,
    }]
    os.makedirs(args.dashboard, exist_ok=True)
    for fname, data in (("rows.json", out), ("summary.json", summary)):
        with open(os.path.join(args.dashboard, fname), "w") as f:
            json.dump(data, f, indent=1)
    print(f"{len(out)} hits -> {args.dashboard}/rows.json, summary.json")


if __name__ == "__main__":
    main()
