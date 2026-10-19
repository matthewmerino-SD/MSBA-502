"""
502 Project: Can Simple Models Forecast Stock Risk as Well as Machine Learning?
===============================================================================
Forecasts next month's volatility for ~500 S&P 500 stocks (2011-2026) and compares
simple models (OLS, Ridge) with complex ones (Random Forest, Gradient Boosting,
ARIMAX), using 12 price-history variables vs. the full set of 39 variables.

Data (downloaded automatically, no data files needed):
    Yahoo Finance (yfinance) ... daily prices, volume, stock splits
    SEC EDGAR API .............. company financial statements
    FRED ....................... VIX, T-bill rate, yield curve, credit spread
    Wikipedia .................. S&P 500 company list and sectors

Setup:
    pip install yfinance pandas numpy scikit-learn statsmodels requests lxml joblib
    Replace the SEC_USER_AGENT placeholder below with your own name and email (required by the SEC).
Run:
    python "502 Project.py"
The first run takes ~15-25 minutes (downloads are cached in ./data). Results print to the
console and are saved to ./results, including the full output in results/output.txt.
"""
import io
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================
SEC_USER_AGENT = "John Cena johncena@gmail.com"   # placeholder: replace with your name and email
MAX_TICKERS = None                  # None = full S&P 500; e.g. 60 for a quick trial
PRICE_START = "2009-01-01"          # extra history so 12-month features exist by 2011
PANEL_START = "2011-01-31"
VALID_START = "2016-12-31"          # tuning: fit on rows before this, score up to TRAIN_END
TRAIN_END = "2018-12-31"            # models trained on rows before this date
CASE_TICKERS = ["AAPL", "NVDA", "TSLA", "JPM", "XOM", "JNJ", "KO"]
RF_TREES = 200
HGB_ITER = 400
N_JOBS = -1
SEED = 42
REFRESH_PRICES = False              # True = re-download prices/FRED/universe
REBUILD_PANEL = False               # True = rebuild panel from cached raw data

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 200)
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

T0 = time.time()


class _Tee:
    """Write console output to both the screen and results/output.txt."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, x):
        for st in self.streams:
            st.write(x)

    def flush(self):
        for st in self.streams:
            st.flush()


def section(title):
    print(f"\n{'=' * 100}\n{title}    [{(time.time() - T0) / 60:.1f} min]\n{'=' * 100}", flush=True)


def save(df, name):
    df.to_csv(RESULTS_DIR / f"{name}.csv")


def cached(path, fn, refresh=False):
    if path.exists() and not refresh:
        return pd.read_pickle(path)
    obj = fn()
    pd.to_pickle(obj, path)
    return obj


# =============================================================================
# 1. DATA SOURCES
# =============================================================================
def get_universe():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (academic research script)"}, timeout=30).text
    t = pd.read_html(io.StringIO(html))[0]
    t = t.rename(columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector", "CIK": "cik"})
    t["ticker"] = t["ticker"].astype(str).str.replace(".", "-", regex=False)
    t["cik"] = pd.to_numeric(t["cik"], errors="coerce")
    t = t.dropna(subset=["cik"]).astype({"cik": int})
    t = t.drop_duplicates("cik")                      # one share class per company (e.g. GOOGL/GOOG)
    return t[["ticker", "name", "sector", "cik"]].reset_index(drop=True)


def download_prices(tickers):
    import yfinance as yf
    frames = []
    for i in range(0, len(tickers), 100):
        chunk = tickers[i:i + 100]
        print(f"  downloading prices {i + 1}-{i + len(chunk)} of {len(tickers)}", flush=True)
        frames.append(yf.download(chunk, start=PRICE_START, auto_adjust=False, actions=True,
                                  progress=False, threads=True, group_by="column"))
    px = pd.concat(frames, axis=1)
    px.index = pd.to_datetime(px.index).tz_localize(None)
    out = {"adj": px["Adj Close"], "close": px["Close"], "volume": px["Volume"]}
    out["splits"] = px["Stock Splits"] if "Stock Splits" in px.columns.get_level_values(0) else out["adj"] * 0
    return {k: v.sort_index(axis=1) for k, v in out.items()}


FRED_SERIES = {"vix": "VIXCLS", "tbill": "DTB3", "curve": "T10Y2Y", "credit": "BAA10Y"}


def fetch_fred():
    out = {}
    for k, sid in FRED_SERIES.items():
        s = pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}",
                        index_col=0, parse_dates=True, na_values=".").iloc[:, 0]
        out[k] = pd.to_numeric(s, errors="coerce")
    return pd.DataFrame(out)


def fetch_companyfacts(cik):
    if not SEC_USER_AGENT or "@" not in SEC_USER_AGENT:
        raise SystemExit("Set SEC_USER_AGENT at the top of the script to your name and email (required by the SEC).")
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
    for attempt in range(3):
        try:
            r = requests.get(url, headers={"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"},
                             timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(2 * (attempt + 1))
    return None


# name: (XBRL tags in priority order, unit, kind). "annual" = 12-month flow, "instant" = balance sheet
CONCEPTS = {
    "revenue": (["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                 "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet"], "USD", "annual"),
    "net_income": (["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"], "USD", "annual"),
    "op_cash_flow": (["NetCashProvidedByUsedInOperatingActivities",
                      "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"], "USD", "annual"),
    "assets": (["Assets"], "USD", "instant"),
    "liabilities": (["Liabilities"], "USD", "instant"),
    "equity": (["StockholdersEquity",
                "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"], "USD", "instant"),
    "cash": (["CashAndCashEquivalentsAtCarryingValue",
              "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", "Cash"], "USD", "instant"),
    "shares": (["dei:EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding"], "shares", "instant"),
}
ANNUAL_ITEMS = {"revenue", "net_income", "op_cash_flow", "revenue_growth"}
FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A", "10-KT"}


def extract_company(js):
    """Keep only what we need: (period end, value, date filed) per item, first-reported version."""
    facts = js.get("facts", {}) if js else {}
    out, filed_all = {}, set()
    for name, (tags, unit, kind) in CONCEPTS.items():
        rows = []
        for pri, tag in enumerate(tags):
            ns, tg = tag.split(":") if ":" in tag else ("us-gaap", tag)
            for it in facts.get(ns, {}).get(tg, {}).get("units", {}).get(unit, []):
                if it.get("form") not in FORMS or it.get("val") is None:
                    continue
                rows.append((it["end"], it.get("start"), float(it["val"]), it["filed"], pri, it["form"]))
        if not rows:
            continue
        d = pd.DataFrame(rows, columns=["end", "start", "val", "filed", "pri", "form"])
        d["end"], d["filed"] = pd.to_datetime(d["end"]), pd.to_datetime(d["filed"])
        filed_all.update(d.loc[d["form"].str.startswith(("10-K", "10-Q")), "filed"].tolist())
        if kind == "annual":
            days = (d["end"] - pd.to_datetime(d["start"])).dt.days
            d = d[(days >= 350) & (days <= 380)]
        # first-reported value for each period (ignores later restatements -> no look-ahead)
        d = d.sort_values(["end", "pri", "filed"]).drop_duplicates("end", keep="first")
        if len(d):
            out[name] = d[["end", "val", "filed"]].reset_index(drop=True)
    if "revenue" in out:
        rv = out["revenue"].sort_values("end").reset_index(drop=True)
        prev = rv["val"].shift(1)
        g = (rv["val"] / prev - 1).where(rv["end"].diff().dt.days.between(330, 400) & (prev > 0))
        out["revenue_growth"] = pd.DataFrame({"end": rv["end"], "val": g, "filed": rv["filed"]}).dropna()
    return {"facts": out, "filed": sorted(filed_all)}


def load_edgar(universe):
    edir = DATA_DIR / "edgar"
    edir.mkdir(exist_ok=True)
    comps, n_new = {}, 0
    for _, row in universe.iterrows():
        p = edir / f"{row.cik}.pkl"
        if p.exists():
            comp = pd.read_pickle(p)
        else:
            js = fetch_companyfacts(row.cik)
            time.sleep(0.12)                                   # SEC limit: 10 requests/second
            comp = extract_company(js)
            pd.to_pickle(comp, p)
            n_new += 1
            if n_new % 50 == 0:
                print(f"  SEC filings downloaded: {n_new}", flush=True)
        comps[row.ticker] = comp
    return comps


# =============================================================================
# 2. FEATURE CONSTRUCTION
# =============================================================================
def split_factor_after(splits):
    """For each day: product of split ratios AFTER that day. Converts as-reported share
    counts to the same split-adjusted basis as Yahoo's Close price."""
    s = splits.replace(0, 1.0).fillna(1.0)
    return s[::-1].cumprod()[::-1] / s


def complete_month_mask(idx_daily, monthly_index):
    last = idx_daily[-1]
    if (last + pd.offsets.BMonthEnd(0) - last).days > 3:      # current month not finished
        return monthly_index < (last + pd.offsets.MonthEnd(0))
    return np.ones(len(monthly_index), bool)


def price_features(adj, close, volume, spy_adj):
    r = adj.pct_change(fill_method=None)
    m = spy_adj.pct_change(fill_method=None)
    ann = np.sqrt(252)
    f = {
        "vol_1w": r.rolling(5, min_periods=4).std() * ann,
        "vol_1m": r.rolling(21, min_periods=15).std() * ann,
        "vol_3m": r.rolling(63, min_periods=45).std() * ann,
        "vol_12m": r.rolling(252, min_periods=180).std() * ann,
        "mom_12_1": adj.shift(21) / adj.shift(252) - 1,
        "ma_gap_200": adj / adj.rolling(200, min_periods=150).mean() - 1,
        "dd_52w": adj / adj.rolling(252, min_periods=180).max() - 1,
        "log_dollar_vol": np.log((close * volume).rolling(21, min_periods=15).mean().where(lambda x: x > 0)),
    }
    w, mp = 252, 180
    e_rm = r.mul(m, axis=0).rolling(w, min_periods=mp).mean()
    e_r = r.rolling(w, min_periods=mp).mean()
    e_m = m.rolling(w, min_periods=mp).mean()
    var_m = m.rolling(w, min_periods=mp).var(ddof=0)
    beta = (e_rm - e_r.mul(e_m, axis=0)).div(var_m, axis=0)
    idio = (r.rolling(w, min_periods=mp).var(ddof=0) - beta.pow(2).mul(var_m, axis=0)).clip(lower=0)
    f["beta_12m"] = beta
    f["idio_vol_12m"] = np.sqrt(idio) * ann

    mo = {k: v.resample("ME").last() for k, v in f.items()}
    mo["ret_1m"] = (1 + r).resample("ME").prod(min_count=10) - 1
    mo["max_ret_1m"] = r.resample("ME").max()
    rv = (r.resample("ME").std() * ann).where(r.resample("ME").count() >= 15)
    mo["close_me"] = close.resample("ME").last()
    keep = complete_month_mask(adj.index, rv.index)
    mo = {k: v.loc[keep] for k, v in mo.items()}
    rv, ret = rv.loc[keep], mo["ret_1m"]
    mo["next_vol"] = rv.shift(-1)
    mo["next_ret"] = ret.shift(-1).where(rv.shift(-1).notna())
    return mo


def fundamentals_monthly(comp, month_ends, split_after):
    res = pd.DataFrame(index=month_ends)
    left = pd.DataFrame({"date": pd.DatetimeIndex(month_ends).astype("datetime64[ns]")})
    for name, d in comp["facts"].items():
        d = d.dropna(subset=["end", "filed", "val"]).sort_values("filed")
        if d.empty:
            continue
        vals = d["val"].to_numpy(float)
        if name == "shares":
            fac = split_after.asof(pd.DatetimeIndex(d["end"])).fillna(split_after.iloc[0]).to_numpy(float)
            vals = vals * fac
            # drop share counts more than 3x away from the company's recent typical count (filing errors)
            ref = pd.Series(vals).rolling(5, min_periods=1).median().shift(1).bfill().to_numpy()
            ok = np.abs(np.log(np.maximum(vals, 1) / np.maximum(ref, 1))) < np.log(3)
            d, vals = d[ok], vals[ok]
            if d.empty:
                continue
        right = pd.DataFrame({"filed": d["filed"].astype("datetime64[ns]").to_numpy(), "val": vals,
                              "end": d["end"].astype("datetime64[ns]").to_numpy()})
        mm = pd.merge_asof(left, right, left_on="date", right_on="filed", direction="backward")
        max_age = 550 if name in ANNUAL_ITEMS else 280
        res[name] = mm["val"].where((mm["date"] - mm["end"]).dt.days <= max_age).to_numpy()
    filed_m = {pd.Timestamp(x).to_period("M") for x in comp["filed"]}
    # Earnings-season flag, point-in-time: did the company file a 10-Q/10-K in next month's
    # calendar month one year ago? (Reporting schedules repeat every year.)
    res["earn_next"] = [float((d.to_period("M") + 1 - 12) in filed_m) for d in month_ends]
    return res


def wide_to_long(frames):
    parts = []
    for k, w in frames.items():
        s = (w.rename_axis(index="date", columns="ticker").reset_index()
              .melt(id_vars="date", var_name="ticker", value_name=k).set_index(["date", "ticker"])[k])
        parts.append(s)
    return pd.concat(parts, axis=1)


def build_panel():
    section("1. DATA COLLECTION")
    uni = cached(DATA_DIR / "universe.pkl", get_universe, REFRESH_PRICES)
    if MAX_TICKERS:
        uni = uni.iloc[:MAX_TICKERS]
    tickers = uni["ticker"].tolist()
    px = cached(DATA_DIR / f"prices_{len(tickers)}.pkl", lambda: download_prices(tickers + ["SPY"]), REFRESH_PRICES)
    fred = cached(DATA_DIR / "fred.pkl", fetch_fred, REFRESH_PRICES)
    comps = load_edgar(uni)

    adj, close, volume, splits = px["adj"], px["close"], px["volume"], px["splits"]
    spy = adj["SPY"]
    stock_cols = [t for t in tickers if t in adj.columns and adj[t].notna().sum() > 500]
    print(f"  Universe: {len(uni)} companies | usable price histories: {len(stock_cols)}")
    adj, close, volume, splits = (x[stock_cols] for x in (adj, close, volume, splits))

    section("2. BUILDING THE STOCK-MONTH PANEL")
    mo = price_features(adj, close, volume, spy)
    panel = wide_to_long(mo)
    month_ends = mo["vol_1m"].index

    fund_parts = []
    for t in stock_cols:
        sa = split_factor_after(splits[t])
        fm = fundamentals_monthly(comps.get(t, {"facts": {}, "filed": []}), month_ends, sa)
        fm["ticker"] = t
        fund_parts.append(fm.rename_axis("date").reset_index())
    fund = pd.concat(fund_parts).set_index(["date", "ticker"])
    panel = panel.join(fund, how="left")
    for c in list(CONCEPTS) + ["revenue_growth", "earn_next"]:
        if c not in panel.columns:
            panel[c] = np.nan

    fr = fred.sort_index().ffill(limit=5).resample("ME").last()
    r_spy = spy.pct_change(fill_method=None)
    mkt = pd.DataFrame(index=month_ends)
    for k in FRED_SERIES:
        mkt[k] = fr[k].reindex(month_ends)
    mkt["spy_vol_1m"] = (r_spy.rolling(21, min_periods=15).std() * np.sqrt(252)).resample("ME").last().reindex(month_ends)
    mkt["rf_next"] = mkt["tbill"] / 100 / 12
    spy_m = ((1 + r_spy).resample("ME").prod(min_count=10) - 1).reindex(month_ends)
    mkt["spy_next_ret"] = spy_m.shift(-1)
    panel = panel.join(mkt, on="date")
    panel = panel.join(uni.set_index("ticker")[["name", "sector"]], on="ticker")
    return panel


VOL_COLS = ["vol_1w", "vol_1m", "vol_3m", "vol_12m", "idio_vol_12m", "spy_vol_1m", "vix"]
WINSOR_COLS = ["log_vol_1w", "log_vol_1m", "log_vol_3m", "log_vol_12m", "log_idio_vol_12m", "beta_12m",
               "ret_1m", "mom_12_1", "ma_gap_200", "dd_52w", "max_ret_1m", "log_dollar_vol",
               "log_mcap", "ep", "bm", "sp", "leverage", "roa", "cash_assets", "ocf_assets", "sales_growth"]
FUND_RAW = ["log_mcap", "ep", "bm", "sp", "leverage", "roa", "cash_assets", "ocf_assets", "sales_growth"]


def prepare(panel):
    df = panel.copy()
    mcap = df["close_me"] * df["shares"]
    mcap = mcap.where(mcap > 0)
    df["mcap_bn"] = mcap / 1e9
    df["log_mcap"] = np.log(mcap)
    df["ep"] = df["net_income"] / mcap                       # earnings yield = 1 / (P/E)
    df["bm"] = df["equity"] / mcap                           # book-to-market
    df["sp"] = df["revenue"] / mcap                          # sales-to-price
    lev = df["liabilities"] / df["assets"]
    df["leverage"] = lev.fillna(1 - df["equity"] / df["assets"])
    df["roa"] = df["net_income"] / df["assets"]
    df["cash_assets"] = df["cash"] / df["assets"]
    df["ocf_assets"] = df["op_cash_flow"] / df["assets"]
    df["sales_growth"] = df["revenue_growth"]
    df["loss"] = (df["net_income"] < 0).astype(float)
    for c in VOL_COLS:
        df["log_" + c] = np.log(df[c].where(df[c] > 0))
    df["pe_ratio"] = (mcap / df["net_income"]).where(df["net_income"] > 0)   # for display only

    # Rate LEVELS jumped from ~0% (2011-2018 training data) to ~5% (2023); models can't extrapolate that.
    # Use 12-month CHANGES instead (raw levels kept for the ablation test).
    sv = df.groupby(level="date")["spy_vol_1m"].first().sort_index()
    df = df.join(pd.DataFrame({"spy_vol_3m": np.sqrt((sv ** 2).rolling(3).mean()),
                               "spy_vol_12m": np.sqrt((sv ** 2).rolling(12).mean()),
                               "spy_next_vol": sv.shift(-1)}), on="date")
    mk = df.groupby(level="date")[["tbill", "curve"]].first().sort_index()
    chg = mk.diff(12).rename(columns={"tbill": "tbill_chg_12m", "curve": "curve_chg_12m"})
    df = df.join(chg, on="date")
    df = df[(df.index.get_level_values("date") >= PANEL_START)]
    df = df.dropna(subset=["log_vol_1m", "log_vol_12m", "beta_12m", "close_me"])
    coverage = df[FUND_RAW].notna().mean()
    df["fund_missing"] = df[["log_mcap", "ep", "leverage"]].isna().any(axis=1).astype(float)

    g = df.groupby(level="date")
    lo = g[WINSOR_COLS].transform(lambda x: x.quantile(0.01))
    hi = g[WINSOR_COLS].transform(lambda x: x.quantile(0.99))
    df[WINSOR_COLS] = df[WINSOR_COLS].clip(lo, hi, axis=None)
    df[WINSOR_COLS] = df[WINSOR_COLS].fillna(df.groupby(level="date")[WINSOR_COLS].transform("median"))
    df[WINSOR_COLS] = df[WINSOR_COLS].fillna(df[WINSOR_COLS].median())
    for c in ["log_vix", "log_spy_vol_1m", "tbill", "curve", "tbill_chg_12m", "curve_chg_12m", "credit", "earn_next"]:
        df[c] = df[c].groupby(level="date").transform(lambda x: x.ffill().bfill()).fillna(df[c].median())
    sec = pd.get_dummies(df["sector"].fillna("Unknown"), prefix="sec", drop_first=True, dtype=float)
    sec.columns = [c.replace(" ", "_") for c in sec.columns]
    df = pd.concat([df, sec], axis=1)
    df["y"] = np.log(df["next_vol"].where(df["next_vol"] > 0))
    return df, coverage


PRICE_F = ["log_vol_1w", "log_vol_1m", "log_vol_3m", "log_vol_12m", "log_idio_vol_12m", "beta_12m",
           "ret_1m", "mom_12_1", "ma_gap_200", "dd_52w", "max_ret_1m", "log_dollar_vol"]
MARKET_F = ["log_vix", "tbill_chg_12m", "curve_chg_12m", "credit", "log_spy_vol_1m"]
FUND_F = ["log_mcap", "ep", "bm", "sp", "leverage", "roa", "cash_assets", "ocf_assets", "sales_growth",
          "loss", "fund_missing"]
EVENT_F = ["earn_next"]
ARIMAX_EXOG = ["log_vol_1m", "log_vol_3m", "log_vix", "credit", "earn_next", "ma_gap_200"]
ARIMAX_EXOG_PRICE = ["log_vol_1m", "log_vol_3m", "ma_gap_200", "dd_52w"]


# =============================================================================
# 3. MODELS
# =============================================================================
GRIDS = {
    "Ridge": [{"alpha": a} for a in [1, 10, 100, 1000, 10000, 100000]],
    "Random forest": [{"max_depth": d, "min_samples_leaf": l} for d in [8, 14] for l in [50, 200]],
    "Gradient boosting": [{"learning_rate": lr, "max_leaf_nodes": n} for lr in [0.03, 0.1] for n in [15, 31]],
}


def make_model(name, p=None):
    p = p or {}
    if name == "OLS":
        return make_pipeline(StandardScaler(), LinearRegression())
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(**p))
    if name == "Random forest":
        return RandomForestRegressor(n_estimators=RF_TREES, max_features=0.33, max_samples=0.5,
                                     n_jobs=N_JOBS, random_state=SEED, **p)
    if name == "Gradient boosting":
        return HistGradientBoostingRegressor(max_iter=HGB_ITER, min_samples_leaf=100, l2_regularization=1.0,
                                             early_stopping=False, random_state=SEED, **p)
    raise ValueError(name)


def _arimax_one(tk, s, train_end, exog):
    import warnings as w
    w.filterwarnings("ignore")
    try:
        s = s.sort_index()
        full = pd.date_range(s.index.min(), s.index.max(), freq="ME")
        s = s.reindex(full)
        X = s[exog].ffill().bfill()
        y = s["y"].to_numpy(float)
        n = int((s.index < train_end).sum())
        if np.isfinite(y[:n]).sum() < 48 or n >= len(s):
            return None
        mu, sd = X.iloc[:n].mean(), X.iloc[:n].std().replace(0, 1).fillna(1)
        Xs = ((X - mu) / sd).fillna(0).to_numpy(float)
        res = SARIMAX(y[:n], exog=Xs[:n], order=(1, 0, 0), trend="c").fit(disp=False, maxiter=200)
        pred = res.apply(y, exog=Xs).predict(start=n, end=len(y) - 1)
        return tk, pd.Series(np.asarray(pred), index=s.index[n:])
    except Exception:
        return None


def oos_r2(y, p, base):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - base) ** 2)


def rank_corr(y, p, dates):
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p), "d": np.asarray(dates)})
    return d.groupby("d")[["y", "p"]].apply(lambda g: g["y"].rank().corr(g["p"].rank())).mean()


def metrics(y, p, base, naive, dates):
    return {"R2 (log vol)": oos_r2(y, p, base),
            "Beats naive (% less error)": 100 * (1 - np.mean((y - p) ** 2) / np.mean((y - naive) ** 2)),
            "Avg error (vol pts)": 100 * np.mean(np.abs(np.exp(y) - np.exp(p))),
            "Rank corr (which stocks)": rank_corr(y, p, dates)}


def perf_stats(r, rf):
    curve = (1 + r).cumprod()
    ex = r - rf
    return {"Ann. return %": 100 * (curve.iloc[-1] ** (12 / len(r)) - 1),
            "Ann. vol %": 100 * r.std() * np.sqrt(12),
            "Sharpe": ex.mean() / ex.std() * np.sqrt(12),
            "Max drawdown %": 100 * (curve / curve.cummax() - 1).min()}


# =============================================================================
# 4. MAIN
# =============================================================================
def main():
    panel_path = DATA_DIR / f"panel_{MAX_TICKERS or 'all'}_v2.pkl"
    panel = cached(panel_path, build_panel, REBUILD_PANEL or REFRESH_PRICES)
    df, coverage = prepare(panel)
    sec_f = [c for c in df.columns if c.startswith("sec_")]
    FULL = PRICE_F + MARKET_F + FUND_F + EVENT_F + sec_f

    dates = df.index.get_level_values("date")
    lab = df["y"].notna()
    tr, te = lab & (dates < TRAIN_END), lab & (dates >= TRAIN_END)
    fit_m, val_m = lab & (dates < VALID_START), lab & (dates >= VALID_START) & (dates < TRAIN_END)
    Xtr, ytr, Xte, yte = df.loc[tr, FULL], df.loc[tr, "y"], df.loc[te, FULL], df.loc[te, "y"]

    # ------------------------------------------------------------------ summary
    section("3. DATA SUMMARY")
    print(f"Stock-month rows (labeled): {lab.sum():,} | companies: {df.index.get_level_values('ticker').nunique()} | "
          f"months: {dates[lab].nunique()} ({dates[lab].min():%Y-%m} to {dates[lab].max():%Y-%m})")
    print(f"Train rows: {tr.sum():,} ({dates[tr].min():%Y-%m} to {dates[tr].max():%Y-%m}) | "
          f"Test rows: {te.sum():,} ({dates[te].min():%Y-%m} to {dates[te].max():%Y-%m})")
    print(f"Features: {len(FULL)} (price {len(PRICE_F)}, market {len(MARKET_F)}, fundamentals {len(FUND_F)}, "
          f"earnings 1, sector dummies {len(sec_f)})")
    print("\nFundamental data coverage (share of stock-months with SEC data, before filling gaps):")
    print((coverage * 100).round(1).to_string())
    print("\nTarget, next-month volatility (%):")
    print((df.loc[lab, "next_vol"] * 100).describe().round(1).to_string())
    snap = df.loc[lab, ["mcap_bn", "pe_ratio", "leverage", "beta_12m", "vol_1m", "next_vol"]].describe().T
    print("\nVariable snapshot (median and middle 50%):\n", snap[["25%", "50%", "75%"]].round(3))

    # ------------------------------------------------------------------ what drives risk
    section("4. WHAT DRIVES STOCK RISK? Avg next-month volatility (%) by quintile, sorted each month")
    chars = {"Market cap": "log_mcap", "Leverage (liab/assets)": "leverage", "Earnings yield (1/PE)": "ep",
             "Book-to-market": "bm", "Profitability (ROA)": "roa", "Sales growth": "sales_growth",
             "Beta": "beta_12m", "Past 1m volatility": "log_vol_1m", "Momentum 12-1": "mom_12_1"}
    d_all = df.loc[lab].copy()
    fund_ok = d_all["fund_missing"] == 0
    rows = {}
    for label, c in chars.items():
        sub = d_all if c in PRICE_F else d_all[fund_ok]
        q = sub.groupby(level="date")[c].transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False))
        tab = (sub["next_vol"] * 100).groupby([sub.index.get_level_values("date"), q]).mean().unstack().mean()
        # same sort but WITHIN each sector (removes industry effects, e.g. banks = high leverage, low vol)
        gk = [sub.index.get_level_values("date"), sub["sector"]]
        qs = sub.groupby(gk)[c].transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False)
                                          if len(x) >= 5 else pd.Series(np.nan, index=x.index))
        ts = (sub["next_vol"] * 100).groupby([sub.index.get_level_values("date"), qs]).mean().unstack().mean()
        rows[label] = list(tab.values) + [tab.iloc[-1] - tab.iloc[0], ts.iloc[-1] - ts.iloc[0]]
    qt = pd.DataFrame(rows, index=["Q1 (low)", "Q2", "Q3", "Q4", "Q5 (high)", "Q5 - Q1", "Q5 - Q1 within sector"]).T
    print(qt.round(1))
    save(qt, "quintiles")

    by_sec = d_all.groupby("sector")["next_vol"].agg(["mean", "count"])
    by_sec["mean"] *= 100
    print("\nAvg next-month volatility by sector (%):\n", by_sec.sort_values("mean", ascending=False).round(1))

    e1 = d_all.groupby([d_all.index.get_level_values("date"), "earn_next"])["next_vol"].mean().unstack()
    if {0.0, 1.0} <= set(e1.columns):
        ratio = (e1[1.0] / e1[0.0] - 1).dropna()
        print(f"\nEarnings-season months: avg volatility {100 * e1[1.0].mean():.1f}% vs {100 * e1[0.0].mean():.1f}% otherwise; "
              f"within the same month, stocks entering earnings season are {100 * ratio.mean():+.1f}% more volatile "
              f"({len(ratio)} months compared)")

    vix_m = d_all.groupby(level="date")["vix"].first()
    hi_vix = vix_m >= vix_m.quantile(2 / 3)
    lo_vix = vix_m <= vix_m.quantile(1 / 3)
    sub = d_all[fund_ok]
    inter = {}
    for label, c in [("Leverage", "leverage"), ("Market cap", "log_mcap"), ("Profitability", "roa")]:
        q = sub.groupby(level="date")[c].transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False))
        t = (sub["next_vol"] * 100).groupby([sub.index.get_level_values("date"), q]).mean().unstack()
        sp = t[4] - t[0]
        inter[label] = {"Q5-Q1 spread, calm markets (low VIX)": sp[lo_vix].mean(),
                        "Q5-Q1 spread, stressed markets (high VIX)": sp[hi_vix].mean()}
    print("\nDo fundamentals matter more when markets are stressed? (vol pts):\n", pd.DataFrame(inter).T.round(1))

    # ------------------------------------------------------------------ returns vs volatility
    section("5. WHY VOLATILITY, NOT RETURNS? Same features, same models, two targets (test period)")
    rv_tab = {}
    for tgt in ["next_ret", "y"]:
        ytr_t, yte_t = df.loc[tr, tgt], df.loc[te, tgt]
        ok_tr, ok_te = ytr_t.notna(), yte_t.notna()
        for nm in ["Ridge", "Gradient boosting"]:
            mdl = make_model(nm, {"alpha": 10} if nm == "Ridge" else {"learning_rate": 0.03, "max_leaf_nodes": 15})
            mdl.fit(Xtr[ok_tr], ytr_t[ok_tr])
            p = mdl.predict(Xte[ok_te])
            rv_tab[(("Next-month RETURN" if tgt == "next_ret" else "Next-month VOLATILITY (log)"), nm)] = {
                "Out-of-sample R2 %": 100 * oos_r2(yte_t[ok_te], p, ytr_t[ok_tr].mean()),
                "Rank corr (which stocks)": rank_corr(yte_t[ok_te], p, dates[te][ok_te.to_numpy()])}
    print(pd.DataFrame(rv_tab).T.round(3))

    # ------------------------------------------------------------------ tuning
    section("6. MODEL TRAINING (hyperparameters tuned on 2017-2018 validation data)")
    MODEL_NAMES = ["OLS", "Ridge", "Random forest", "Gradient boosting", "ARIMAX"]

    def train_all(feats, ar_exog, tag):
        print(f"\n  --- {tag} ---")
        bst = {}
        for nm, grid in GRIDS.items():
            scores = []
            for p in grid:
                mdl = make_model(nm, p).fit(df.loc[fit_m, feats], df.loc[fit_m, "y"])
                scores.append(np.mean((df.loc[val_m, "y"] - mdl.predict(df.loc[val_m, feats])) ** 2))
            bst[nm] = grid[int(np.argmin(scores))]
        print("  best params: " + "; ".join(f"{k} {v}" for k, v in bst.items()))
        prd, mods = {}, {}
        for nm in MODEL_NAMES[:-1]:
            mods[nm] = make_model(nm, bst.get(nm)).fit(df.loc[tr, feats], ytr)
            prd[nm] = pd.Series(mods[nm].predict(df.loc[te, feats]), index=Xte.index)
        t1 = time.time()
        jobs = Parallel(n_jobs=N_JOBS)(delayed(_arimax_one)(tk, g.droplevel("ticker")[["y"] + ar_exog],
                                                            pd.Timestamp(TRAIN_END), ar_exog)
                                        for tk, g in df.groupby(level="ticker"))
        ar = [pd.DataFrame({"date": s_.index, "ticker": tk, "p": s_.values}) for j in jobs if j for tk, s_ in [j]]
        ar = pd.concat(ar).set_index(["date", "ticker"])["p"] if ar else pd.Series(dtype=float)
        prd["ARIMAX"] = ar.reindex(Xte.index)
        print(f"  ARIMAX fitted per stock ({ar.index.get_level_values('ticker').nunique()} stocks, "
              f"exog: {', '.join(ar_exog)}) in {time.time() - t1:.0f}s")
        return bst, mods, pd.DataFrame(prd)

    best12, models12, P12 = train_all(PRICE_F, ARIMAX_EXOG_PRICE, f"12 PRICE-HISTORY VARIABLES: {', '.join(PRICE_F)}")
    best, models, P = train_all(FULL, ARIMAX_EXOG, f"ALL {len(FULL)} VARIABLES (price + market + fundamentals + earnings + sector)")

    # ------------------------------------------------------------------ main results
    section("7. HEADLINE: HOW WELL DOES EACH MODEL PREDICT NEXT MONTH'S VOLATILITY? (test 2019-today)")
    common = P.notna().all(axis=1) & P12.notna().all(axis=1) & np.isfinite(yte)
    print(f"Evaluated on {common.sum():,} stock-months where every model has a forecast "
          f"({100 * common.mean():.1f}% of test rows)")
    yc, Pc, P12c = yte[common], P[common], P12[common]
    dc = yc.index.get_level_values("date")
    base = ytr.mean()
    naive = np.log(df.loc[yc.index, "vol_1m"].clip(lower=1e-4))
    vix_fit = LinearRegression().fit(df.loc[tr, ["log_vix"]], ytr)
    p_vix = pd.Series(vix_fit.predict(df.loc[yc.index, ["log_vix"]]), index=yc.index)
    rel = np.log(df.loc[yc.index, "vol_12m"] / df.loc[yc.index, "spy_vol_12m"])
    p_vixrel = (df.loc[yc.index, "log_vix"] - np.log(100) + rel).fillna(naive)
    bl = {"Baseline: historical average": np.full(len(yc), base),
          "Baseline: last month's volatility": naive,
          "Baseline: VIX alone (fitted)": p_vix,
          "Baseline: VIX x stock's relative risk": p_vixrel}
    rows = {}
    for k, p in bl.items():
        m_ = metrics(yc, p, base, naive, dc)
        rows[k] = {("12 price vars", c): m_[c] for c in m_} | {("All vars", c): m_[c] for c in m_}
    for nm in MODEL_NAMES:
        m12, mall = metrics(yc, P12c[nm], base, naive, dc), metrics(yc, Pc[nm], base, naive, dc)
        rows[nm] = {("12 price vars", c): m12[c] for c in m12} | {("All vars", c): mall[c] for c in mall}
    head = pd.DataFrame(rows).T
    head.columns = pd.MultiIndex.from_tuples(head.columns)
    head.loc[["Baseline: historical average", "Baseline: VIX alone (fitted)"],
             [("12 price vars", "Rank corr (which stocks)"), ("All vars", "Rank corr (which stocks)")]] = np.nan
    order = ["R2 (log vol)", "Beats naive (% less error)", "Avg error (vol pts)", "Rank corr (which stocks)"]
    for c in order:
        print(f"\n{c}:")
        print(head.xs(c, axis=1, level=1).round(3).to_string())
    print("\n(Baselines don't use model variables, so both columns are identical for them. "
          "VIX alone gives every stock the same forecast in a month, so it cannot rank stocks.)")
    save(head, "headline_models")

    yr = {}
    years = (dc + pd.offsets.MonthEnd(1)).year       # year of the month being forecast
    for y_ in sorted(set(years)):
        m_ = years == y_
        yr[y_] = {nm: 100 * oos_r2(yc[m_], Pc.loc[m_, nm], base) for nm in Pc.columns}
        yr[y_]["Ridge (12 vars)"] = 100 * oos_r2(yc[m_], P12c.loc[m_, "Ridge"], base)
        yr[y_]["RF (12 vars)"] = 100 * oos_r2(yc[m_], P12c.loc[m_, "Random forest"], base)
        yr[y_]["Naive"] = 100 * oos_r2(yc[m_], naive[m_], base)
        yr[y_]["VIX alone"] = 100 * oos_r2(yc[m_], p_vix[m_], base)
    yr = pd.DataFrame(yr).T
    print("\nR2 (%) by year being forecast (all-variable models unless noted):\n", yr.round(1))
    save(yr, "r2_by_year")

    # ------------------------------------------------------------------ market level vs VIX
    section("7b. CAN WE BEAT THE VIX? Forecasting the S&P 500's (SPY) next-month volatility, test 2019-today")
    mk = df.groupby(level="date")[["log_vix", "spy_vol_1m", "spy_vol_3m", "spy_vol_12m", "spy_next_vol",
                                   "credit", "tbill_chg_12m", "curve_chg_12m"]].first().sort_index()
    for c in ["spy_vol_1m", "spy_vol_3m", "spy_vol_12m"]:
        mk["log_" + c] = np.log(mk[c])
    mk["y"] = np.log(mk["spy_next_vol"])
    mk = mk.dropna(subset=["y", "log_spy_vol_12m", "log_vix"])
    mtr, mte = mk[mk.index < TRAIN_END], mk[mk.index >= TRAIN_END]
    har = ["log_spy_vol_1m", "log_spy_vol_3m", "log_spy_vol_12m"]
    allm = har + ["log_vix", "credit", "tbill_chg_12m", "curve_chg_12m"]
    mp = {"VIX as-is (no model)": mte["log_vix"] - np.log(100)}
    for label, cols, mdl in [("VIX alone (fitted regression)", ["log_vix"], LinearRegression()),
                             ("Past volatility only (OLS)", har, LinearRegression()),
                             ("Past volatility + VIX (OLS)", har + ["log_vix"], LinearRegression()),
                             ("All market variables (Ridge)", allm, make_pipeline(StandardScaler(), Ridge(alpha=10))),
                             ("All market variables (Random forest)", allm,
                              RandomForestRegressor(300, min_samples_leaf=5, random_state=SEED))]:
        mp[label] = pd.Series(mdl.fit(mtr[cols], mtr["y"]).predict(mte[cols]), index=mte.index)
    vix_err = np.mean((mte["y"] - mp["VIX as-is (no model)"]) ** 2)
    mres = {k: {"R2 (log vol)": oos_r2(mte["y"], p, mtr["y"].mean()),
                "% less error than raw VIX": 100 * (1 - np.mean((mte["y"] - p) ** 2) / vix_err),
                "Avg error (vol pts)": 100 * np.mean(np.abs(np.exp(mte["y"]) - np.exp(p)))} for k, p in mp.items()}
    mres = pd.DataFrame(mres).T
    print(f"{len(mtr)} training months, {len(mte)} test months")
    print(mres.round(3))
    print(f"Avg VIX level {np.exp(mte['log_vix']).mean():.1f} vs avg realized SPY volatility "
          f"{100 * np.exp(mte['y']).mean():.1f}% in the test period (the VIX usually overestimates)")
    save(mres, "market_vs_vix")

    # ------------------------------------------------------------------ what adds value
    section("8. WHICH DATA ADDS PREDICTIVE POWER? (feature-set ablation, test period)")
    sets = {"Price history only": PRICE_F,
            "+ Market (VIX, rates, credit)": PRICE_F + MARKET_F,
            "+ Fundamentals, sector, earnings (full)": FULL,
            "Full model without VIX": [c for c in FULL if c != "log_vix"],
            "Full model without interest-rate variables": [c for c in FULL if c not in ("tbill_chg_12m", "curve_chg_12m")],
            "Price history + VIX only": PRICE_F + ["log_vix"],
            "Full, raw rate LEVELS (not changes)": [
                {"tbill_chg_12m": "tbill", "curve_chg_12m": "curve"}.get(c, c) for c in FULL],
            "Fundamentals + sector only (no price data)": FUND_F + sec_f + EVENT_F}
    abl = {}
    for sname, cols in sets.items():
        for nm in ["Ridge", "Random forest", "Gradient boosting"]:
            mdl = make_model(nm, best[nm]).fit(df.loc[tr, cols], ytr)
            p = mdl.predict(df.loc[yc.index, cols])
            abl[(sname, nm)] = {"R2 (log vol)": oos_r2(yc, p, base), "Rank corr": rank_corr(yc, p, dc)}
    abl = pd.DataFrame(abl).T
    print(abl.round(3))
    save(abl, "ablation")

    # ------------------------------------------------------------------ interpretation
    section("9. INTERPRETATION")
    keep_cols = [c for c in FULL if Xtr[c].std() > 0]
    Xs = StandardScaler().fit_transform(Xtr[keep_cols])
    ols = sm.OLS(ytr.to_numpy(), sm.add_constant(Xs)).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(dates[tr])[0]})
    coef = pd.DataFrame({"Std. coef": np.asarray(ols.params)[1:], "t-stat (clustered by month)": np.asarray(ols.tvalues)[1:]},
                        index=keep_cols)
    coef = coef[~coef.index.str.startswith("sec_")].sort_values("t-stat (clustered by month)", key=abs, ascending=False)
    print("OLS on standardized features, training data (sector dummies hidden). "
          f"Training R2 = {ols.rsquared:.3f}\nCoefficient = change in log volatility per 1 std. dev. of the feature:")
    print(coef.round(3))
    save(coef, "ols_coefficients")

    samp = Xte.loc[yc.index].sample(min(15000, len(yc)), random_state=SEED)
    pi = permutation_importance(models["Gradient boosting"], samp, yte.loc[samp.index], n_repeats=3,
                                random_state=SEED, n_jobs=N_JOBS, scoring="r2")
    imp = pd.Series(pi.importances_mean, index=FULL).sort_values(ascending=False)
    print("\nGradient boosting permutation importance (drop in test R2 when feature is shuffled), top 15:")
    print(imp.head(15).round(4).to_string())
    save(imp.to_frame("importance"), "gb_importance")

    # ------------------------------------------------------------------ business application
    section("10. BUSINESS APPLICATION: sort stocks by predicted risk each month (test period, equal-weighted)")
    tdf = df.loc[yc.index, ["next_ret", "next_vol", "rf_next", "spy_next_ret"]].copy()
    for nm in ["Ridge", "Gradient boosting"]:
        tdf["pred"] = Pc[nm]
        tdf["q"] = tdf.groupby(level="date")["pred"].transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False))
        port = tdf.groupby([tdf.index.get_level_values("date"), "q"])["next_ret"].mean().unstack()
        rf = tdf.groupby(level="date")["rf_next"].first()
        out = {}
        for q in range(5):
            st = perf_stats(port[q].dropna(), rf.reindex(port[q].dropna().index))
            st["Avg predicted vol %"] = 100 * np.exp(tdf.loc[tdf["q"] == q, "pred"]).mean()
            st["Avg realized vol %"] = 100 * tdf.loc[tdf["q"] == q, "next_vol"].mean()
            out[f"Q{q + 1}" + (" (lowest risk)" if q == 0 else " (highest risk)" if q == 4 else "")] = st
        ew = tdf.groupby(level="date")["next_ret"].mean()
        out["All stocks, equal-weighted"] = perf_stats(ew, rf.reindex(ew.index))
        spy_r = tdf.groupby(level="date")["spy_next_ret"].first().dropna()
        out["SPY"] = perf_stats(spy_r, rf.reindex(spy_r.index))
        pt = pd.DataFrame(out).T
        print(f"\nPortfolios sorted on {nm} forecasts:\n", pt.round(2))
        save(pt, f"portfolios_{nm.replace(' ', '_')}")

    # ------------------------------------------------------------------ real examples
    section("11. REAL EXAMPLES")
    ex = []
    for tk in CASE_TICKERS:
        if tk not in yc.index.get_level_values("ticker"):
            continue
        m_ = yc.index.get_level_values("ticker") == tk
        a = np.exp(yc[m_])
        ex.append({"Ticker": tk, "Actual avg vol %": 100 * a.mean(),
                   "Ridge avg forecast %": 100 * np.exp(Pc.loc[m_, "Ridge"]).mean(),
                   "GB avg forecast %": 100 * np.exp(Pc.loc[m_, "Gradient boosting"]).mean(),
                   "Ridge corr w/ actual": np.corrcoef(a, np.exp(Pc.loc[m_, "Ridge"]))[0, 1],
                   "GB corr w/ actual": np.corrcoef(a, np.exp(Pc.loc[m_, "Gradient boosting"]))[0, 1]})
    if ex:
        print("Selected stocks, test period (monthly forecasts vs what happened):\n", pd.DataFrame(ex).set_index("Ticker").round(2))

    covid = pd.date_range("2019-12-31", "2020-06-30", freq="ME")
    cv = pd.DataFrame({nm: np.exp(Pc[nm]).groupby(dc).mean() for nm in Pc.columns})
    cv.insert(0, "Actual", np.exp(yc).groupby(dc).mean())
    cv = (cv.reindex(covid) * 100)
    cv.index = [f"made {d:%b %Y} -> for {(d + pd.offsets.MonthEnd(1)):%b %Y}" for d in cv.index]
    print("\nCOVID crash: average forecast across all stocks vs actual volatility (%):\n", cv.round(1))
    save(cv, "covid_example")

    # ------------------------------------------------------------------ live forecast
    section("12. FORECAST FOR NEXT MONTH (latest data, models refit on all history)")
    last = dates.max()
    live = df[(dates == last)]
    allx, ally = df.loc[lab, FULL], df.loc[lab, "y"]
    fr = pd.DataFrame(index=live.index)
    for nm in ["Ridge", "Gradient boosting"]:
        fr[nm + " forecast %"] = 100 * np.exp(make_model(nm, best[nm]).fit(allx, ally).predict(live[FULL]))
    fr["Last month vol %"] = 100 * live["vol_1m"]
    fr["Sector"] = live["sector"]
    fr["Name"] = live["name"]
    fr = fr.droplevel("date").sort_values("Ridge forecast %", ascending=False)
    print(f"Features as of {last:%Y-%m-%d}; forecasting {(last + pd.offsets.MonthEnd(1)):%B %Y}")
    print("\nHighest predicted risk:\n", fr.head(10).round(1))
    print("\nLowest predicted risk:\n", fr.tail(10).round(1))
    save(fr, "live_forecast")
    print(f"\nDone in {(time.time() - T0) / 60:.1f} min. Tables saved to {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    import sys
    with open(RESULTS_DIR / "output.txt", "w") as _log:
        sys.stdout = _Tee(sys.__stdout__, _log)
        main()
        sys.stdout = sys.__stdout__