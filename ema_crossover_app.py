import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import csv
import io
from datetime import datetime
from zoneinfo import ZoneInfo

st.set_page_config(
    page_title="21/50 EMA Crossover Screener",
    page_icon="📈",
    layout="wide"
)

st.title("📈 21/50 EMA Crossover Screener")
st.subheader("Nifty 100")
st.caption(
    "21 EMA and 50 EMA crossover uses daily closing prices. "
    "Latest available intraday price is shown as current price."
)

NIFTY100_URL = (
    "https://www.niftyindices.com/IndexConstituent/"
    "ind_nifty100list.csv"
)

@st.cache_data(ttl=86400)
def get_symbols():
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "Referer": "https://www.niftyindices.com/"
    }
    response = requests.get(
        NIFTY100_URL,
        headers=headers,
        timeout=30
    )
    response.raise_for_status()
    text = response.content.decode(
        "utf-8-sig",
        errors="replace"
    )
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 2:
        raise ValueError("Nifty 100 list is empty.")
    header = [str(x).strip().lower() for x in rows[0]]
    if "symbol" not in header:
        raise ValueError("Symbol column not found.")
    idx = header.index("symbol")
    symbols = []
    for row in rows[1:]:
        if len(row) > idx:
            symbol = row[idx].strip()
            if symbol:
                symbols.append(symbol)
    return list(dict.fromkeys(symbols))

@st.cache_data(ttl=300)
def get_daily(symbols):
    return yf.download(
        tickers=[s + ".NS" for s in symbols],
        period="2y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True
    )

@st.cache_data(ttl=120)
def get_intraday(symbols):
    return yf.download(
        tickers=[s + ".NS" for s in symbols],
        period="1d",
        interval="5m",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True
    )

def get_ticker_data(data, ticker):
    if data is None or data.empty:
        return pd.DataFrame()
    try:
        if isinstance(data.columns, pd.MultiIndex):
            l0 = data.columns.get_level_values(0)
            l1 = data.columns.get_level_values(1)
            if ticker in l0:
                result = data[ticker].copy()
            elif ticker in l1:
                result = data.xs(ticker, axis=1, level=1).copy()
            else:
                return pd.DataFrame()
        else:
            result = data.copy()

        wanted = ["Open", "High", "Low", "Close"]
        available = [c for c in wanted if c in result.columns]
        if "Close" not in available:
            return pd.DataFrame()

        result = result[available].copy()
        for c in available:
            result[c] = pd.to_numeric(result[c], errors="coerce")
        return result.dropna(how="all")
    except Exception:
        return pd.DataFrame()

def get_dates(index):
    values = []
    for value in index:
        try:
            ts = pd.Timestamp(value)
            if ts.tzinfo is not None:
                ts = ts.tz_convert("Asia/Kolkata")
            values.append(ts.date())
        except Exception:
            values.append(None)
    return pd.Series(values, index=index)

def calculate_stock(symbol, daily_all, intraday_all, today):
    ticker = symbol + ".NS"

    daily = get_ticker_data(daily_all, ticker)
    if daily.empty:
        return None

    daily = daily.dropna(subset=["Close"]).copy()
    dates = get_dates(daily.index)

    completed = daily.loc[dates < today].copy()
    completed = completed.dropna(subset=["Close"])

    if len(completed) < 50:
        return None

    close = completed["Close"]

    ema21 = close.ewm(
        span=21,
        adjust=False,
        min_periods=21
    ).mean()

    ema50 = close.ewm(
        span=50,
        adjust=False,
        min_periods=50
    ).mean()

    valid = pd.DataFrame({
        "Close": close,
        "EMA21": ema21,
        "EMA50": ema50
    }).dropna()

    if len(valid) < 2:
        return None

    current_21 = float(valid["EMA21"].iloc[-1])
    current_50 = float(valid["EMA50"].iloc[-1])

    intraday = get_ticker_data(
        intraday_all,
        ticker
    )

    live_price = np.nan

    if not intraday.empty and "Close" in intraday.columns:
        prices = intraday["Close"].dropna()
        if not prices.empty:
            live_price = float(prices.iloc[-1])

    if not np.isfinite(live_price) or live_price <= 0:
        live_price = float(valid["Close"].iloc[-1])

    if current_50 != 0:
        difference = (
            current_21 / current_50 - 1
        ) * 100
    else:
        difference = np.nan

    crossover = "No Recent Crossover"
    crossover_date = None
    days_since = np.nan

    for i in range(1, len(valid)):
        p21 = float(valid["EMA21"].iloc[i - 1])
        p50 = float(valid["EMA50"].iloc[i - 1])
        c21 = float(valid["EMA21"].iloc[i])
        c50 = float(valid["EMA50"].iloc[i])

        if p21 <= p50 and c21 > c50:
            crossover = "Bullish"
            crossover_date = pd.Timestamp(valid.index[i])
            if crossover_date.tzinfo is not None:
                crossover_date = crossover_date.tz_convert(
                    "Asia/Kolkata"
                )

        elif p21 >= p50 and c21 < c50:
            crossover = "Bearish"
            crossover_date = pd.Timestamp(valid.index[i])
            if crossover_date.tzinfo is not None:
                crossover_date = crossover_date.tz_convert(
                    "Asia/Kolkata"
                )

    if crossover_date is not None:
        crossover_date_only = crossover_date.date()
        days_since = (today - crossover_date_only).days
        crossover_date_text = crossover_date.strftime("%d-%m-%Y")
    else:
        crossover_date_text = "-"

    if current_21 > current_50:
        structure = "21 EMA > 50 EMA"
    elif current_21 < current_50:
        structure = "21 EMA < 50 EMA"
    else:
        structure = "21 EMA = 50 EMA"

    return {
        "Stock": symbol,
        "Price": round(live_price, 2),
        "21 EMA": round(current_21, 2),
        "50 EMA": round(current_50, 2),
        "EMA Difference %": round(difference, 2),
        "Crossover": crossover,
        "Crossover Date": crossover_date_text,
        "Days Since Crossover": (
            int(days_since)
            if np.isfinite(days_since)
            else np.nan
        ),
        "Structure": structure
    }

def colour_crossover(value):
    if value == "Bullish":
        return (
            "background-color: #198754;"
            "color: white;"
            "font-weight: bold;"
        )
    if value == "Bearish":
        return (
            "background-color: #dc3545;"
            "color: white;"
            "font-weight: bold;"
        )
    return ""

def colour_structure(value):
    if value == "21 EMA > 50 EMA":
        return (
            "background-color: #d1e7dd;"
            "color: #0f5132;"
            "font-weight: bold;"
        )
    if value == "21 EMA < 50 EMA":
        return (
            "background-color: #f8d7da;"
            "color: #842029;"
            "font-weight: bold;"
        )
    return ""

if st.button("🔍 Scan Nifty 100", type="primary"):

    try:
        symbols = get_symbols()
    except Exception as error:
        st.error("Unable to download Nifty 100 list.")
        st.error(str(error))
        st.stop()

    st.info("Nifty 100 stocks found: " + str(len(symbols)))

    with st.spinner("Downloading daily data..."):
        try:
            daily_data = get_daily(symbols)
        except Exception as error:
            st.error("Daily data download failed.")
            st.error(str(error))
            st.stop()

    with st.spinner("Downloading latest prices..."):
        try:
            intraday_data = get_intraday(symbols)
        except Exception:
            intraday_data = pd.DataFrame()
            st.warning(
                "Intraday data unavailable. "
                "Latest daily close will be used."
            )

    now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
    today = now_ist.date()

    results = []
    progress = st.progress(0)
    total = len(symbols)

    for i, symbol in enumerate(symbols):
        try:
            result = calculate_stock(
                symbol,
                daily_data,
                intraday_data,
                today
            )
            if result is not None:
                results.append(result)
        except Exception:
            pass

        if total:
            progress.progress(
                int(((i + 1) / total) * 100)
            )

    progress.empty()

    if not results:
        st.error("No stock data was calculated.")
        st.stop()

    df = pd.DataFrame(results)

    columns = [
        "Stock",
        "Price",
        "21 EMA",
        "50 EMA",
        "EMA Difference %",
        "Crossover",
        "Crossover Date",
        "Days Since Crossover",
        "Structure"
    ]

    df = df[columns]

    # Most recent crossover first.
    df["_days"] = pd.to_numeric(
        df["Days Since Crossover"],
        errors="coerce"
    )
    df = df.sort_values(
        "_days",
        ascending=True,
        na_position="last"
    ).drop(columns="_days")
    df = df.reset_index(drop=True)

    for column in [
        "Price",
        "21 EMA",
        "50 EMA",
        "EMA Difference %"
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        ).round(2)

    updated_time = now_ist.strftime(
        "%d-%m-%Y %I:%M:%S %p IST"
    )

    st.success(
        "🕐 Last updated: " + updated_time
    )

    st.subheader(
        "📋 EMA Crossover Results — "
        + str(len(df))
        + " stocks"
    )

    styled = (
        df.style
        .map(
            colour_crossover,
            subset=["Crossover"]
        )
        .map(
            colour_structure,
            subset=["Structure"]
        )
        .format(
            {
                "Price": "{:.2f}",
                "21 EMA": "{:.2f}",
                "50 EMA": "{:.2f}",
                "EMA Difference %": "{:.2f}",
                "Days Since Crossover": "{:.0f}"
            },
            na_rep="-"
        )
    )

    st.dataframe(
        styled,
        use_container_width=True,
        height=650,
        hide_index=True
    )

    csv_data = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Results CSV",
        data=csv_data,
        file_name="nifty100_21_50_ema_crossover.csv",
        mime="text/csv"
        )
