"""
Baur & Smales (2020) "Hedging geopolitical risk with precious metals" (JBF 117)
-- replication & sample extension to 31.12.2025

Run:  python gpr_extension.py            (reads files from the same folder)
      python gpr_extension.py --synthetic (pipeline test with fake data)

Output: results_gpr_extension.xlsx (one sheet per table)

------------------------------------------------------------------------------
INPUT FILES (put in the same folder as this script)
------------------------------------------------------------------------------
1) data_gpr_daily_recent.xls   Daily GPR, Caldara & Iacoviello
                               (matteoiacoviello.com/gpr.htm -> "Daily GPR")
                               needs columns: date/DAY, GPRD, GPRD_THREAT, GPRD_ACT
2) data_gpr_export.xls         Monthly GPR (GPR, GPRT, GPRA)  -> monthly analysis
3) DAAA.csv, DBAA.csv, T10Y2Y.csv   FRED downloads (fred.stlouisfed.org -> Download CSV)
4) futures.xlsx                From LSEG Workspace / Datastream. Sheet "prices":
                               column "Date" + one column per asset (settlement price):
                                   GOLD, SILVER, PLATINUM, PALLADIUM, COPPER, SP500, USDX, TNOTE10
                               Optional sheets "volume" and "oi" with the same column
                               names -> Volume and Open Int. controls (as in the paper).

NOTES ON DESIGN CHOICES (write these into the methodology section)
- The GPR index was revised in Caldara & Iacoviello (2022, AER). The old 2018
  daily series is not updated anymore, so the WHOLE sample 1985-2025 is run on
  the new index. Sample "orig" (1985-01 .. 2018-12) re-estimates the paper's
  period with the new index -> shows whether the original results survive the
  change of measure; "post" (2019-01 .. 2025-12) is the true out-of-sample test.
- dGPR_t = 100 * [ln GPR_t - ln GPR_{t-1}], computed on each asset's trading-day
  grid. Following the paper, return(t) is regressed on dGPR(t+1) (news is
  published the day after the event). LEAD = 1 below.
- Standard errors: the paper uses Hodrick (1992). For one-period (non-overlapping)
  daily returns Hodrick 1B collapses to the White heteroskedasticity-consistent
  estimator, so COV = "HC1" is the faithful choice. Newey-West is available as a
  robustness option (COV = "HAC").
- Credit spread = Baa - Aaa (FRED DBAA - DAAA). The paper writes "Aaa - Baa";
  the sign convention only flips the control's coefficient.
- Hi_GPR thresholds (top decile of level / change) are computed within each
  estimation sample; the paper's 166.8 / 56.4 are on the old index scale.
- EGARCH(1,1) with dGPR and controls in both mean and variance equation
  (paper eq. 5A/5B), QML (Bollerslev-Wooldridge) robust standard errors.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tools.numdiff import approx_fprime, approx_hess
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
OUTFILE = HERE / "results_gpr_extension.xlsx"

ASSETS = ["GOLD", "SILVER", "PLATINUM", "PALLADIUM", "COPPER", "SP500", "USDX", "TNOTE10",
          "CHF", "JPY"]          # CHF, JPY = USD per 1 CHF / per 1 JPY (up = currency appreciates)
PRECIOUS = ["GOLD", "SILVER", "PLATINUM", "PALLADIUM"]

SAMPLES = {
    "orig": ("2000-01-01", "2018-12-31"),   # pre-period (B&S: 1985-10/2018; Yahoo futures start 2000)
    "post": ("2019-01-01", "2025-12-31"),   # out-of-sample extension
    "full": ("2000-01-01", "2025-12-31"),
}
# (1) structural break dates: end of B&S sample, and start of the "realised
#     geopolitical risk" era (Russia-Ukraine build-up / invasion 2022)
BREAK_DATES = {"2019": "2019-01-01", "2022": "2022-01-01"}
# (2) threat vs act sub-period
WAR_SAMPLE = ("2022-01-01", "2025-12-31")
# (3) safe-haven horse race: gold vs other safe-haven candidates
HORSE_RACE = [("GOLD", "CHF"), ("GOLD", "JPY"), ("GOLD", "TNOTE10"), ("GOLD", "SILVER")]
# (4) quantile regression
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

LEAD = 1            # regress R_t on dGPR_{t+LEAD}
COV = "HC1"         # "HC1" (= Hodrick 1992 at horizon 1) or "HAC"
HAC_LAGS = 5
HI_Q = 0.90         # top decile
# Hi_GPR thresholds are taken from the whole daily GPR history 1985-2025 (set in
# main), so that "extreme" means the same GPR level in every sample and asset.
HI_THR = {"level": None, "change": None}
RUN_EGARCH = True
EGARCH_SAMPLES = ["orig", "post", "full"]


# ----------------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------------
def _find(name_options):
    for n in name_options:
        p = HERE / n
        if p.exists():
            return p
    return None


def load_gpr_daily() -> pd.DataFrame:
    p = _find(["data_gpr_daily_recent.xls", "data_gpr_daily_recent.xlsx", "data_gpr_daily.xls"])
    if p is None:
        sys.exit("Missing data_gpr_daily_recent.xls (Daily GPR) in " + str(HERE))
    df = pd.read_excel(p)
    cols = {c.lower(): c for c in df.columns}
    if "date" in cols:
        d = pd.to_datetime(df[cols["date"]])
    elif "day" in cols:
        d = pd.to_datetime(df[cols["day"]].astype(int).astype(str), format="%Y%m%d")
    else:
        raise ValueError("No date column (date / DAY) in daily GPR file")
    out = pd.DataFrame({
        "GPR": df[cols["gprd"]].values,
        "GPR_THREAT": df[cols["gprd_threat"]].values,
        "GPR_ACT": df[cols["gprd_act"]].values,
    }, index=d).sort_index()
    out.index.name = "Date"
    return out.astype(float)


def load_gpr_monthly() -> pd.DataFrame | None:
    p = _find(["data_gpr_export.xls", "data_gpr_export.xlsx"])
    if p is None:
        return None
    df = pd.read_excel(p)
    df["month"] = pd.to_datetime(df["month"])
    df = df.set_index("month")[["GPR", "GPRT", "GPRA"]].dropna()
    return df.astype(float)


def load_fred(series: str) -> pd.Series:
    p = _find([f"{series}.csv", f"{series.lower()}.csv"])
    if p is None:
        sys.exit(f"Missing {series}.csv (FRED download) in {HERE}")
    df = pd.read_csv(p)
    dcol = df.columns[0]                       # DATE or observation_date
    s = pd.to_numeric(df[series if series in df.columns else df.columns[1]], errors="coerce")
    s.index = pd.to_datetime(df[dcol])
    return s.rename(series).dropna()


def load_futures():
    p = _find(["futures.xlsx", "futures.xls"])
    if p is None:
        sys.exit("Missing futures.xlsx (sheet 'prices': Date + asset columns) in " + str(HERE))
    xl = pd.ExcelFile(p)

    def _sheet(name):
        if name not in xl.sheet_names:
            return None
        d = pd.read_excel(xl, name)
        d["Date"] = pd.to_datetime(d["Date"])
        return d.set_index("Date").sort_index().apply(pd.to_numeric, errors="coerce")

    return _sheet("prices"), _sheet("volume"), _sheet("oi")


# Data cleaning for Yahoo continuous futures.
# GOLD, SP500, USDX, TNOTE10 were checked and are clean -> not filtered, so that
# genuine crash days (2008, March 2020) are never removed.
# (a) Hampel filter: a price is a bad print if its log level deviates from the
#     centred 7-day rolling median by more than max(K x robust daily std, FLOOR).
#     Real level shifts are kept because the median moves with them.
# (b) Yahoo PL=F / PA=F contain stale stretches (old contract prices) in 2007-2009,
#     e.g. platinum shown at ~$1750 when the market was ~$1000. These cannot be
#     fixed point-wise, so platinum and palladium start in 2010.
CLEAN_ASSETS = ["SILVER", "PLATINUM", "PALLADIUM", "COPPER", "CHF", "JPY"]
HAMPEL_K = 8
HAMPEL_FLOOR = {"CHF": 0.06, "JPY": 0.06}      # others: 0.08 (8 %)
ASSET_START = {"PLATINUM": "2010-01-01", "PALLADIUM": "2010-01-01"}


def clean_prices(prices, vol=None):
    prices = prices.copy()
    log = []
    for a, start in ASSET_START.items():
        if a in prices.columns:
            n = int(prices.loc[:pd.Timestamp(start) - pd.Timedelta(days=1), a].notna().sum())
            prices.loc[:pd.Timestamp(start) - pd.Timedelta(days=1), a] = np.nan
            log.append((a, pd.NaT, f"sample starts {start} (stale Yahoo data before), {n} obs dropped"))
    for a in [c for c in CLEAN_ASSETS if c in prices.columns]:
        p = prices[a].dropna()
        p = p[p > 0]
        lp = np.log(p)
        r = lp.diff()
        sd = 1.4826 * (r - r.median()).abs().median()
        med = lp.rolling(7, center=True, min_periods=4).median()
        thr = max(HAMPEL_K * sd, HAMPEL_FLOOR.get(a, 0.08))
        bad = (lp - med).abs() > thr
        log += [(a, d, f"Hampel: |log p - median7| > {thr:.3f}") for d in p.index[bad]]
        prices[a] = p[~bad].reindex(prices.index)
    out = pd.DataFrame(log, columns=["Asset", "Date", "Reason"])
    if len(out):
        print("Cleaning:", out.groupby("Asset").size().to_dict())
    return prices, out


def build_controls() -> pd.DataFrame:
    aaa, baa, tp = load_fred("DAAA"), load_fred("DBAA"), load_fred("T10Y2Y")
    c = pd.concat([aaa, baa, tp], axis=1)
    c["CREDIT_SPREAD"] = c["DBAA"] - c["DAAA"]
    c["TERM_PREM"] = c["T10Y2Y"]
    return c[["CREDIT_SPREAD", "TERM_PREM"]]


# ----------------------------------------------------------------------------
# PANEL CONSTRUCTION (one frame per asset on its own trading-day grid)
# ----------------------------------------------------------------------------
def asset_frame(asset, prices, vol, oi, gpr, ctrl) -> pd.DataFrame:
    p = prices[asset].dropna()
    p = p[p > 0]
    df = pd.DataFrame({"P": p})
    df["R"] = 100 * np.log(df["P"]).diff()

    g = gpr.reindex(df.index)                  # GPR on the same calendar day
    for col, name in [("GPR", "dGPR"), ("GPR_THREAT", "dTHREAT"), ("GPR_ACT", "dACT")]:
        lg = np.log(g[col].where(g[col] > 0))
        df[name] = (100 * lg.diff()).shift(-LEAD)          # change at t+LEAD
    df["GPR_LEVEL"] = g["GPR"].shift(-LEAD)
    for k in (-1, 0, 1, 2):                                  # for lead-lag table
        lg = np.log(g["GPR"].where(g["GPR"] > 0))
        df[f"dGPR_k{k:+d}"] = (100 * lg.diff()).shift(-k)

    c = ctrl.reindex(df.index).ffill(limit=3)
    df["dCREDIT"] = c["CREDIT_SPREAD"].diff()
    df["dTERM"] = c["TERM_PREM"].diff()

    if vol is not None and asset in vol:
        v = vol[asset].reindex(df.index)
        df["VOLUME"] = np.log(v.where(v > 0))
    if oi is not None and asset in oi:
        o = oi[asset].reindex(df.index)
        df["dOI"] = 100 * np.log(o.where(o > 0)).diff()
    return df.replace([np.inf, -np.inf], np.nan)


USE_VOLUME_OI = False   # Yahoo volume is 0 on many days -> would drop observations.
                        # Set True with Datastream/Bloomberg volume & open interest.


def controls_for(df):
    base = ["dCREDIT", "dTERM"]
    if not USE_VOLUME_OI:
        return base
    return base + [c for c in ("VOLUME", "dOI") if c in df.columns and df[c].notna().sum() > 100]


def cut(df, sample):
    a, b = SAMPLES[sample]
    return df.loc[a:b]


# ----------------------------------------------------------------------------
# OLS HELPER
# ----------------------------------------------------------------------------
def ols(df, y, xs):
    d = df[[y] + xs].dropna()
    if len(d) < 30:
        return None
    X = sm.add_constant(d[xs])
    m = sm.OLS(d[y], X)
    if COV == "HAC":
        r = m.fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    else:
        r = m.fit(cov_type=COV)
    return r


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def column(r, keep):
    """Coefficient (SE) column for table output."""
    out = {}
    if r is None:
        return out
    for v in keep:
        if v in r.params.index:
            out[v] = f"{r.params[v]:.4f}{stars(r.pvalues[v])}"
            out[v + " (se)"] = f"({r.bse[v]:.4f})"
    out["Controls"] = "YES" if "dCREDIT" in r.params.index else "NO"
    out["Adj. R2"] = f"{r.rsquared_adj:.4f}"
    out["F-stat"] = f"{r.fvalue:.2f}" if np.isfinite(r.fvalue) else ""
    out["DW"] = f"{durbin_watson(r.resid):.3f}"
    out["N"] = int(r.nobs)
    return out


def tidy(cols: dict, row_order: list) -> pd.DataFrame:
    t = pd.DataFrame(cols)
    order = []
    for v in row_order:
        order += [v, v + " (se)"]
    order += ["Controls", "Adj. R2", "F-stat", "DW", "N"]
    t = t.reindex([r for r in order if r in t.index])
    t.index = ["" if i.endswith(" (se)") else i for i in t.index]
    return t


# ----------------------------------------------------------------------------
# TABLES
# ----------------------------------------------------------------------------
def table1_summary(frames, sample):
    rows = {}
    first = next(iter(frames.values()))
    g = cut(first, sample)["dGPR"].dropna()
    for name, s in [("dGPR", g)] + [(a, cut(f, sample)["R"].dropna()) for a, f in frames.items()]:
        try:
            adf_p = adfuller(s.values, regression="ct", autolag="AIC")[1]
        except Exception:
            adf_p = np.nan
        rows[name] = {"Mean": s.mean(), "Std": s.std(), "Min": s.min(), "Max": s.max(),
                      "Skew": s.skew(), "Kurt": s.kurt(), "ADF p (c+t)": adf_p, "N": len(s)}
    return pd.DataFrame(rows).T


def table1_corr(frames, sample):
    first = next(iter(frames.values()))
    d = {"dGPR": cut(first, sample)["dGPR"]}
    for a, f in frames.items():
        d[a] = cut(f, sample)["R"]
    return pd.DataFrame(d).corr()          # pairwise (assets have different start dates)


def table3_baseline(frames, sample):
    cols = {}
    for a, f in frames.items():
        d = cut(f, sample)
        cols[(a, "2A")] = column(ols(d, "R", ["dGPR"]), ["const", "dGPR"])
        cols[(a, "2B")] = column(ols(d, "R", ["dGPR"] + controls_for(d)), ["const", "dGPR"] + controls_for(d))
    return tidy(cols, ["const", "dGPR", "dCREDIT", "dTERM", "VOLUME", "dOI"])


def table4_threat_act(frames, sample):
    cols = {}
    for a, f in frames.items():
        d = cut(f, sample)
        xs = ["dTHREAT", "dACT"]
        cols[(a, "uni")] = column(ols(d, "R", xs), ["const"] + xs)
        cols[(a, "ctrl")] = column(ols(d, "R", xs + controls_for(d)), ["const"] + xs)
    return tidy(cols, ["const", "dTHREAT", "dACT"])


def table5_extreme(frames, sample):
    """Eq. (4). Panel A: Hi = top decile GPR level, interaction.
    Panel B: restricted to Hi level. Panel C: Hi = top decile dGPR. Panel D: restricted."""
    panels = {}
    for pan, basis in [("A", "level"), ("C", "change")]:
        cols, cols_r = {}, {}
        for a, f in frames.items():
            d = cut(f, sample).copy()
            base = d["GPR_LEVEL"] if basis == "level" else d["dGPR"]
            thr = HI_THR[basis] if HI_THR[basis] is not None else base.quantile(HI_Q)
            d["Hi"] = (base > thr).astype(float).where(base.notna())
            d["dGPRxHi"] = d["dGPR"] * d["Hi"]
            cols[(a, "uni")] = column(ols(d, "R", ["dGPRxHi"]), ["const", "dGPRxHi"])
            xs = ["dGPR", "Hi", "dGPRxHi"] + controls_for(d)
            cols[(a, "full")] = column(ols(d, "R", xs), ["const", "dGPR", "Hi", "dGPRxHi"])
            cols[(a, "uni")]["Threshold"] = f"{thr:.2f}"
            dr = d[d["Hi"] == 1]
            cols_r[(a, "uni")] = column(ols(dr, "R", ["dGPR"]), ["const", "dGPR"])
            cols_r[(a, "ctrl")] = column(ols(dr, "R", ["dGPR"] + controls_for(dr)), ["const", "dGPR"])
        t = tidy(cols, ["const", "dGPR", "Hi", "dGPRxHi"])
        thr_row = pd.DataFrame(cols).loc[["Threshold"]] if "Threshold" in pd.DataFrame(cols).index else None
        if thr_row is not None:
            t = pd.concat([t, thr_row])
        panels[pan] = t
        panels[chr(ord(pan) + 1)] = tidy(cols_r, ["const", "dGPR"])
    return panels


def table7_leadlag(frames, sample):
    cols = {}
    ks = ["dGPR_k-1", "dGPR_k+0", "dGPR_k+1", "dGPR_k+2"]
    for a, f in frames.items():
        d = cut(f, sample)
        cols[(a, "ctrl")] = column(ols(d, "R", ks + controls_for(d)), ["const"] + ks)
    return tidy(cols, ["const"] + ks)


def table_break(frames, break_date):
    """(1)+(2) Full sample with Post dummy: R = b0 + b1 dGPR + b2 POST + b3 dGPR*POST + controls.
    b3 = change in the GPR sensitivity after break_date. TA_ctrl: same for threats vs acts."""
    cols = {}
    for a, f in frames.items():
        d = cut(f, "full").copy()
        d["POST"] = (d.index >= break_date).astype(float)
        d["dGPRxPOST"] = d["dGPR"] * d["POST"]
        xs = ["dGPR", "POST", "dGPRxPOST"]
        cols[(a, "uni")] = column(ols(d, "R", xs), ["const"] + xs)
        cols[(a, "ctrl")] = column(ols(d, "R", xs + controls_for(d)), ["const"] + xs)
        # threat/act version
        d["dTHREATxPOST"] = d["dTHREAT"] * d["POST"]
        d["dACTxPOST"] = d["dACT"] * d["POST"]
        xs2 = ["dTHREAT", "dACT", "POST", "dTHREATxPOST", "dACTxPOST"]
        cols[(a, "TA_ctrl")] = column(ols(d, "R", xs2 + controls_for(d)), ["const"] + xs2)
    return tidy(cols, ["const", "dGPR", "POST", "dGPRxPOST", "dTHREAT", "dACT", "dTHREATxPOST", "dACTxPOST"])


def table_threat_act_war(frames):
    """(2) Threats vs acts in 2022-2025 only, incl. Wald test H0: b_threat = b_act."""
    cols = {}
    a0, b0 = WAR_SAMPLE
    for a, f in frames.items():
        d = f.loc[a0:b0]
        for spec, xs in [("uni", ["dTHREAT", "dACT"]), ("ctrl", ["dTHREAT", "dACT"] + controls_for(d))]:
            r = ols(d, "R", xs)
            c = column(r, ["const", "dTHREAT", "dACT"])
            if r is not None:
                w = r.wald_test("dTHREAT = dACT", scalar=True)
                c["Wald p (THREAT=ACT)"] = f"{float(w.pvalue):.3f}"
            cols[(a, spec)] = c
    t = tidy(cols, ["const", "dTHREAT", "dACT"])
    extra = pd.DataFrame(cols).reindex(["Wald p (THREAT=ACT)"])
    return pd.concat([t, extra])


def table_horse_race(frames, sample):
    """(3) Stacked regression on common dates for each pair (A, B):
    R_it = a + b dGPR_t + c D_B + d (D_B x dGPR_t) + controls, SE clustered by date.
    d = (GPR sensitivity of B) - (GPR sensitivity of A); d < 0 => A is the better hedge.
    Also run on the top-decile-GPR-level days (safe-haven version)."""
    cols = {}
    for A, B in HORSE_RACE:
        if A not in frames or B not in frames:
            continue
        fa, fb = cut(frames[A], sample), cut(frames[B], sample)
        common = fa.index.intersection(fb.index)
        for label, mask in [("all days", None), ("top-decile GPR", "hi")]:
            parts = []
            for dummy, f in [(0.0, fa), (1.0, fb)]:
                p = f.loc[common, ["R", "dGPR", "GPR_LEVEL", "dCREDIT", "dTERM"]].copy()
                p["D_B"] = dummy
                p["date_id"] = np.arange(len(p))
                parts.append(p)
            s = pd.concat(parts)
            if mask == "hi":
                thr = HI_THR["level"] if HI_THR["level"] is not None else s["GPR_LEVEL"].quantile(HI_Q)
                s = s[s["GPR_LEVEL"] > thr]
            s["D_BxdGPR"] = s["D_B"] * s["dGPR"]
            s["D_BxdCREDIT"] = s["D_B"] * s["dCREDIT"]     # controls may differ by asset
            s["D_BxdTERM"] = s["D_B"] * s["dTERM"]
            xs = ["dGPR", "D_B", "D_BxdGPR", "dCREDIT", "dTERM", "D_BxdCREDIT", "D_BxdTERM"]
            s = s[["R", "date_id"] + xs].dropna()
            if len(s) < 60:
                continue
            r = sm.OLS(s["R"], sm.add_constant(s[xs])).fit(
                cov_type="cluster", cov_kwds={"groups": s["date_id"]})
            c = column(r, ["dGPR", "D_BxdGPR"])
            c["B sensitivity (b+d)"] = f"{r.params['dGPR'] + r.params['D_BxdGPR']:.4f}"
            cols[(f"{A} vs {B}", label)] = c
    t = tidy(cols, ["dGPR", "D_BxdGPR"])
    extra = pd.DataFrame(cols).reindex(["B sensitivity (b+d)"])
    return pd.concat([t, extra])


def table_quantile(frames, sample):
    """(4) Quantile regression of returns on dGPR (+ controls) across the return
    distribution. Lower quantiles = the asset's own bad days: a positive dGPR
    coefficient there means GPR shocks cushion the left tail (safe-haven-type)."""
    rows = {}
    for a, f in frames.items():
        d = cut(f, sample)
        xs = ["dGPR", "dCREDIT", "dTERM"]
        dd = d[["R"] + xs].dropna()
        if len(dd) < 200:
            continue
        X = sm.add_constant(dd[xs])
        coef, se = {}, {}
        for q in QUANTILES:
            try:
                r = sm.QuantReg(dd["R"], X).fit(q=q, max_iter=5000)
                coef[f"q{q:.2f}"] = f"{r.params['dGPR']:.4f}{stars(r.pvalues['dGPR'])}"
                se[f"q{q:.2f}"] = f"({r.bse['dGPR']:.4f})"
            except Exception:
                coef[f"q{q:.2f}"], se[f"q{q:.2f}"] = "n/a", ""
        o = ols(dd, "R", xs)
        coef["OLS"] = f"{o.params['dGPR']:.4f}{stars(o.pvalues['dGPR'])}"
        se["OLS"] = f"({o.bse['dGPR']:.4f})"
        rows[a] = coef
        rows[a + " (se)"] = se
    t = pd.DataFrame(rows).T
    t.index = ["" if i.endswith(" (se)") else i for i in t.index]
    return t


# ----------------------------------------------------------------------------
# EGARCH(1,1) with exogenous regressors in mean and variance (eq. 5A / 5B)
# ----------------------------------------------------------------------------
try:
    from numba import njit
except ImportError:  # works without numba, just slower
    def njit(f=None, **kw):
        return f if f is not None else (lambda g: g)

EZ = np.sqrt(2 / np.pi)


@njit(cache=False)
def _egarch_ll(theta, y, Xm, Xv, h0):
    km = Xm.shape[1]
    kv = Xv.shape[1]
    b = theta[:km]
    lc, l1, gE, l2 = theta[km], theta[km + 1], theta[km + 2], theta[km + 3]
    lv = theta[km + 4:km + 4 + kv]
    n = y.shape[0]
    ll = np.empty(n)
    logh = h0
    zprev = 0.0
    for t in range(n):
        mu = 0.0
        for j in range(km):
            mu += Xm[t, j] * b[j]
        if t > 0:
            xv = 0.0
            for j in range(kv):
                xv += Xv[t, j] * lv[j]
            logh = lc + l1 * (abs(zprev) - EZ) + gE * zprev + l2 * logh + xv
            if logh > 30.0:
                logh = 30.0
            elif logh < -30.0:
                logh = -30.0
        h = np.exp(logh)
        e = y[t] - mu
        ll[t] = -0.5 * (np.log(2 * np.pi) + logh + e * e / h)
        zprev = e / np.sqrt(h)
    return ll


def egarch_fit(d: pd.DataFrame, xs_mean, xs_var):
    d = d[["R"] + sorted(set(xs_mean) | set(xs_var))].dropna()
    y = d["R"].values.astype(float)
    Xm = np.column_stack([np.ones(len(d))] + [d[x].values for x in xs_mean]).astype(float)
    Xv = np.column_stack([d[x].values for x in xs_var]).astype(float)
    h0 = float(np.log(np.var(y)))
    names = ["c"] + [f"mean:{x}" for x in xs_mean] + ["lambda_c", "lambda1 |z|", "gammaE z", "lambda2 logh"] + [f"var:{x}" for x in xs_var]

    b_ols = np.linalg.lstsq(Xm, y, rcond=None)[0]
    theta0 = np.concatenate([b_ols, [0.02 * h0, 0.1, 0.0, 0.97], np.zeros(Xv.shape[1])])
    theta0[Xm.shape[1]] = (1 - 0.97) * h0

    nll = lambda th: -_egarch_ll(th, y, Xm, Xv, h0).sum()
    res = minimize(nll, theta0, method="L-BFGS-B",
                   bounds=[(None, None)] * Xm.shape[1] + [(None, None), (None, None), (None, None), (-0.999, 0.999)] + [(None, None)] * Xv.shape[1],
                   options={"maxiter": 2000})
    th = res.x
    # QML robust covariance: H^-1 G'G H^-1
    try:
        H = approx_hess(th, lambda t: _egarch_ll(t, y, Xm, Xv, h0).sum())
        G = approx_fprime(th, lambda t: _egarch_ll(t, y, Xm, Xv, h0))
        Hi = np.linalg.inv(H)
        V = Hi @ (G.T @ G) @ Hi
        se = np.sqrt(np.clip(np.diag(V), 0, None))
    except Exception:
        se = np.full_like(th, np.nan)
    z = th / se
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(np.abs(z)))
    out = {}
    for n_, b_, s_, p_ in zip(names, th, se, p):
        out[n_] = f"{b_:.4f}{stars(p_) if np.isfinite(p_) else ''}"
        out[n_ + " (se)"] = f"({s_:.4f})"
    out["LogL"] = f"{-res.fun:.1f}"
    out["Converged"] = str(res.success)
    out["N"] = len(y)
    return out, names


def table6_egarch(frames, sample):
    cols, names = {}, None
    for a, f in frames.items():
        d = cut(f, sample)
        xs = ["dGPR", "dCREDIT", "dTERM"]
        print(f"    EGARCH {sample} {a} ...", flush=True)
        try:
            cols[a], names = egarch_fit(d, xs, xs)
        except Exception as e:
            print(f"      failed: {e}")
    if not cols:
        return pd.DataFrame()
    t = pd.DataFrame(cols)
    order = []
    for n_ in names:
        order += [n_, n_ + " (se)"]
    order += ["LogL", "Converged", "N"]
    t = t.reindex(order)
    t.index = ["" if i.endswith(" (se)") else i for i in t.index]
    return t


# ----------------------------------------------------------------------------
# MONTHLY ANALYSIS (Section 4.1 / Table 8): Gold & SP500 vs %GPR and GPRSHOCK
# ----------------------------------------------------------------------------
def table8_monthly(prices, gm, sample):
    if gm is None:
        return None
    a, b = SAMPLES[sample]
    g = gm.loc[a:b].copy()
    g["pctGPR"] = 100 * np.log(g["GPR"]).diff()
    # GPRSHOCK = residual of AR(1) on GPR level (Caldara & Iacoviello 2018)
    ar = sm.OLS(g["GPR"].iloc[1:].values, sm.add_constant(g["GPR"].shift(1).iloc[1:].values)).fit()
    g["GPRSHOCK"] = np.r_[np.nan, ar.resid]

    cols = {}
    for asset in [x for x in ("GOLD", "SP500") if x in prices]:
        p = prices[asset].dropna().loc[a:b]
        variants = {
            "month-end": p.resample("MS").last(),
            "average": p.resample("MS").mean(),
            "day 15": p[p.index.day <= 15].resample("MS").last(),
            "day 10": p[p.index.day <= 10].resample("MS").last(),
        }
        for vname, pm in variants.items():
            r = (100 * np.log(pm).diff()).rename("R")
            d = pd.concat([r, g[["pctGPR", "GPRSHOCK"]]], axis=1)
            for x in ("pctGPR", "GPRSHOCK"):
                cols[(asset, vname, x)] = column(ols(d, "R", [x]), ["const", x])
    return tidy(cols, ["const", "pctGPR", "GPRSHOCK"])


# ----------------------------------------------------------------------------
# SYNTHETIC DATA (only to test that the pipeline runs)
# ----------------------------------------------------------------------------
def make_synthetic():
    rng = np.random.default_rng(0)
    cal = pd.date_range("1984-12-01", "2025-12-31", freq="D")
    lg = np.cumsum(rng.normal(0, 0.05, len(cal))) * 0.1 + np.log(100) + rng.normal(0, 0.3, len(cal))
    gpr = pd.DataFrame({"GPR": np.exp(lg)}, index=cal)
    gpr["GPR_THREAT"] = gpr["GPR"] * np.exp(rng.normal(0, 0.2, len(cal)))
    gpr["GPR_ACT"] = gpr["GPR"] * np.exp(rng.normal(0, 0.2, len(cal)))
    bd = pd.bdate_range("1985-01-01", "2025-12-31")
    dg = 100 * np.log(gpr["GPR"].reindex(bd)).diff().shift(-1).fillna(0).values
    prices = {}
    for i, a in enumerate(ASSETS):
        beta = {"GOLD": 0.002, "SILVER": 0.002, "SP500": -0.002, "CHF": 0.001}.get(a, 0.0)
        r = beta * dg + rng.standard_t(5, len(bd)) * 0.8
        prices[a] = 100 * np.exp(np.cumsum(r) / 100)
    prices = pd.DataFrame(prices, index=bd)
    vol = pd.DataFrame(rng.integers(1000, 100000, prices.shape), index=bd, columns=ASSETS).astype(float)
    oi = pd.DataFrame(rng.integers(10000, 500000, prices.shape), index=bd, columns=ASSETS).astype(float)
    ctrl = pd.DataFrame({"CREDIT_SPREAD": 1 + np.cumsum(rng.normal(0, 0.01, len(bd))) * 0.1,
                         "TERM_PREM": 1 + np.cumsum(rng.normal(0, 0.02, len(bd))) * 0.1}, index=bd)
    gm = gpr.resample("MS").mean().rename(columns={"GPR_THREAT": "GPRT", "GPR_ACT": "GPRA"})
    return gpr, prices, vol, oi, ctrl, gm


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    global OUTFILE, RUN_EGARCH
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="test run with fake data")
    ap.add_argument("--no-egarch", action="store_true")
    args = ap.parse_args()
    if args.no_egarch:
        RUN_EGARCH = False

    if args.synthetic:
        gpr, prices, vol, oi, ctrl, gm = make_synthetic()
        OUTFILE = HERE / "results_SYNTHETIC_TEST.xlsx"
    else:
        gpr = load_gpr_daily()
        gm = load_gpr_monthly()
        prices, vol, oi = load_futures()
        ctrl = build_controls()
    prices, clean_log = clean_prices(prices, vol)

    assets = [a for a in ASSETS if a in prices.columns]
    missing = [a for a in ASSETS if a not in prices.columns]
    if missing:
        print("Note: assets not in futures file, skipped:", missing)
    print(f"GPR daily: {gpr.index.min().date()} .. {gpr.index.max().date()}")
    frames = {a: asset_frame(a, prices, vol, oi, gpr, ctrl) for a in assets}

    g_all = gpr.loc["1985-01-01":"2025-12-31", "GPR"]     # whole GPR history
    HI_THR["level"] = float(g_all.quantile(HI_Q))
    HI_THR["change"] = float((100 * np.log(g_all.where(g_all > 0)).diff()).quantile(HI_Q))
    print(f"Hi_GPR thresholds (1985-2025): level > {HI_THR['level']:.1f}, change > {HI_THR['change']:.1f}%")

    coverage = pd.DataFrame({a: {"first": f["R"].first_valid_index().date(),
                                 "last": f["R"].last_valid_index().date(),
                                 **{f"N {s}": int(cut(f, s)["R"].notna().sum()) for s in SAMPLES}}
                             for a, f in frames.items()}).T

    sheets = {}
    for s in SAMPLES:
        print(f"Sample {s} {SAMPLES[s]}")
        sheets[f"T1_summary_{s}"] = table1_summary(frames, s)
        sheets[f"T1_corr_{s}"] = table1_corr(frames, s)
        sheets[f"T3_baseline_{s}"] = table3_baseline(frames, s)
        sheets[f"T4_threat_act_{s}"] = table4_threat_act(frames, s)
        for pan, t in table5_extreme(frames, s).items():
            sheets[f"T5{pan}_extreme_{s}"] = t
        sheets[f"T7_leadlag_{s}"] = table7_leadlag(frames, s)
        m = table8_monthly(prices, gm, s)
        if m is not None:
            sheets[f"T8_monthly_{s}"] = m
        sheets[f"X3_horserace_{s}"] = table_horse_race(frames, s)
        print("    quantile regressions ...", flush=True)
        sheets[f"X4_quantile_{s}"] = table_quantile(frames, s)
        if RUN_EGARCH and s in EGARCH_SAMPLES:
            sheets[f"T6_egarch_{s}"] = table6_egarch(frames, s)
    for k, bd in BREAK_DATES.items():
        sheets[f"X1_break_{k}"] = table_break(frames, bd)
    sheets["X2_threat_act_2022_25"] = table_threat_act_war(frames)

    readme = pd.DataFrame({"Info": [
        "Baur & Smales (2020) replication + extension to 2025-12-31",
        f"Samples: {SAMPLES}",
        f"Return(t) regressed on dGPR(t+{LEAD}); dGPR = 100*dln(GPR), new Caldara-Iacoviello (2022) daily index",
        f"Std errors: {COV}" + (f" ({HAC_LAGS} lags)" if COV == "HAC" else " (= Hodrick 1992 at 1-period horizon)"),
        "Coefficient with stars, standard error in parentheses on the row below. *** 1%, ** 5%, * 10%",
        "T5A/T5C: Hi dummy = top decile of GPR level / dGPR over the whole 1985-2025 GPR history (fixed threshold); T5B/T5D: sample restricted to Hi=1",
        "NOTE: Yahoo futures start ~2000 (PLATINUM/PALLADIUM 2010), so 'orig' is effectively 2000-2018 for most assets - see DATA_coverage",
        "NOTE: daily GPR changes are very noisy (sd ~44%/day, AR(1) ~ -0.37); GPR_ACT has zero days (mostly 2020-21) -> those log changes are missing",
        f"X1_break_*: full sample, interaction with POST dummy ({BREAK_DATES}); dGPRxPOST = change in sensitivity; TA_ctrl = threats/acts version",
        f"X2: threats vs acts in {WAR_SAMPLE}, Wald p tests H0: b_threat = b_act",
        "X3: horse race, stacked regression on common dates, SE clustered by date; D_BxdGPR = sens(B) - sens(A); 'top-decile GPR' = safe-haven days",
        f"X4: quantile regression of R on dGPR + dCREDIT + dTERM, quantiles {QUANTILES}; OLS column for comparison",
        "CHF and JPY are USD per unit of the currency, so a positive coefficient = currency appreciates when GPR rises",
        "T8: monthly, pctGPR = 100*dln(monthly GPR), GPRSHOCK = AR(1) residual of GPR",
        "SYNTHETIC DATA - NOT REAL RESULTS" if args.synthetic else "",
    ]})
    with pd.ExcelWriter(OUTFILE, engine="openpyxl") as xw:
        readme.to_excel(xw, sheet_name="README", index=False)
        clean_log.to_excel(xw, sheet_name="DATA_cleaning_log", index=False)
        coverage.to_excel(xw, sheet_name="DATA_coverage")
        for name, t in sheets.items():
            t.to_excel(xw, sheet_name=name[:31])
    print("Saved:", OUTFILE)


if __name__ == "__main__":
    main()
