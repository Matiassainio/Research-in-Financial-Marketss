"""
Downloads all missing input data for gpr_extension.py into this folder.

    pip install yfinance pandas openpyxl xlrd requests
    python fetch_data.py

Creates:
  futures.xlsx                 (sheets "prices" and "volume")  <- yfinance
  DAAA.csv, DBAA.csv, T10Y2Y.csv                               <- FRED
  data_gpr_daily_recent.xls                                    <- Caldara & Iacoviello

NOTE (write into the Data section): Yahoo continuous front-month futures start
~2000 (not 1985) and roll at expiry. In a full study the data would come from
Datastream/Bloomberg with open-interest based rolling, as in Baur & Smales (2020).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

HERE = Path(__file__).resolve().parent
START, END = "1985-01-01", "2026-01-01"   # END is exclusive in yfinance -> data to 31.12.2025

TICKERS = {
    "GOLD": "GC=F",
    "SILVER": "SI=F",
    "PLATINUM": "PL=F",
    "PALLADIUM": "PA=F",
    "COPPER": "HG=F",
    "SP500": "ES=F",        # E-mini S&P 500 futures (from 2000)
    "USDX": "DX-Y.NYB",     # ICE US Dollar Index
    "TNOTE10": "ZN=F",      # 10-Year T-Note futures
    "CHF": "CHF=X",         # quoted CHF per USD -> inverted below to USD per CHF
    "JPY": "JPY=X",         # quoted JPY per USD -> inverted below to USD per JPY
}
INVERT = {"CHF", "JPY"}     # so that a price increase = the safe-haven currency appreciates

FRED = ["DAAA", "DBAA", "T10Y2Y"]
GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch_yahoo():
    prices, volume = {}, {}
    for name, tic in TICKERS.items():
        print(f"yfinance {name:10s} {tic}")
        d = yf.download(tic, start=START, end=END, interval="1d",
                        auto_adjust=False, progress=False, threads=False)
        if d.empty:
            print("   -> NO DATA")
            continue
        if isinstance(d.columns, pd.MultiIndex):          # newer yfinance versions
            d.columns = d.columns.get_level_values(0)
        p = d["Close"].astype(float)
        v = d["Volume"].astype(float) if "Volume" in d else pd.Series(np.nan, index=d.index)

        # Clean stale quotes: Yahoo repeats the previous price on days without
        # trades (bad for PL=F / PA=F in early years). Drop zero-volume days for
        # contracts that normally report volume.
        # Only drop a zero-volume day if the price is ALSO unchanged (pure stale
        # quote). Dropping all zero-volume days removed too many real prices.
        if (v > 0).mean() > 0.5:
            stale = ((v == 0) | v.isna()) & (p == p.shift(1))
            p = p.where(~stale)
        p = p.where(p > 0)
        if name in INVERT:
            p = 1.0 / p
        prices[name], volume[name] = p, v
        r = 100 * np.log(p.dropna()).diff()
        print(f"   {p.first_valid_index().date()} .. {p.last_valid_index().date()}  "
              f"N={p.notna().sum()}  zero-ret share={(r == 0).mean():.3f}  max|r|={r.abs().max():.1f}%")

    P = pd.DataFrame(prices)
    V = pd.DataFrame(volume)
    P.index.name = V.index.name = "Date"
    P.index = pd.to_datetime(P.index).tz_localize(None)
    V.index = pd.to_datetime(V.index).tz_localize(None)
    out = HERE / "futures.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        P.reset_index().to_excel(xw, sheet_name="prices", index=False)
        V.reset_index().to_excel(xw, sheet_name="volume", index=False)
    print("Saved", out)


def fetch_fred():
    import time
    for s in FRED:
        if (HERE / f"{s}.csv").exists():
            print(f"{s}.csv already exists, skipped")
            continue
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}&cosd=1984-01-01&coed=2025-12-31"
        for attempt in range(1, 4):
            try:
                r = requests.get(url, headers=HEADERS, timeout=180)
                r.raise_for_status()
                (HERE / f"{s}.csv").write_bytes(r.content)
                print("Saved", s + ".csv")
                break
            except Exception as e:
                print(f"FRED {s} attempt {attempt} failed ({type(e).__name__})")
                time.sleep(5)
        else:
            print(f"  -> download manually: https://fred.stlouisfed.org/series/{s}  (Download -> CSV, save as {s}.csv)")


def fetch_gpr():
    try:
        r = requests.get(GPR_URL, headers=HEADERS, timeout=120)
        r.raise_for_status()
        (HERE / "data_gpr_daily_recent.xls").write_bytes(r.content)
        print("Saved data_gpr_daily_recent.xls")
    except Exception as e:
        print(f"GPR daily failed ({e}). Download manually from https://www.matteoiacoviello.com/gpr.htm")


if __name__ == "__main__":
    import sys
    if "--fred-gpr-only" not in sys.argv:
        fetch_yahoo()
    fetch_fred()
    fetch_gpr()
    print("\nDone. Next: python gpr_extension.py")
