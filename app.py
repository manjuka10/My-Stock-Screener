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
    page_title="My Stock Screener",
    page_icon="📊",
    layout="wide"
)

st.title("📊 My Stock Screener")
st.subheader("Nifty 100 Technical Screener")
st.caption("Latest available intraday price is used as current price.")

URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv"


@st.cache_data(ttl=86400)
def get_symbols():
    r = requests.get(
        URL,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.niftyindices.com/"
        },
        timeout=30
    )
    r.raise_for_status()

    text = r.content.decode("utf-8-sig", errors="replace")
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
            s = row[idx].strip()
            if s:
                symbols.append(s)

    return list(dict.fromkeys(symbols))


@st.cache_data(ttl=300)
def get_daily(symbols):
    tickers = [s + ".NS" for s in symbols]

    return yf.download(
        tickers=tickers,
        period="2y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True
    )


@st.cache_data(ttl=120)
def get_intraday(symbols):
    tickers = [s + ".NS" for s in symbols]

    return yf.download(
        tickers=tickers,
        period="1d",
        interval="5m",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True
    )


def ticker_data(data, ticker):

    if data is None or data.empty:
        return pd.DataFrame()

    try:
        if isinstance(data.columns, pd.MultiIndex):

            l0 = data.columns.get_level_values(0)
            l1 = data.columns.get_level_values(1)

            if ticker in l0:
                x = data[ticker].copy()

            elif ticker in l1:
                x = data.xs(
                    ticker,
                    axis=1,
                    level=1
                ).copy()

            else:
                return pd.DataFrame()

        else:
            x = data.copy()

        cols = ["Open", "High", "Low", "Close"]

        cols = [
            c for c in cols
            if c in x.columns
        ]

        if "Close" not in cols:
            return pd.DataFrame()

        x = x[cols].copy()

        for c in cols:
            x[c] = pd.to_numeric(
                x[c],
                errors="coerce"
            )

        return x.dropna(how="all")

    except Exception:
        return pd.DataFrame()


def date_value(x):

    try:
        t = pd.Timestamp(x)

        if t.tzinfo is not None:
            t = t.tz_convert("Asia/Kolkata")

        return t.date()

    except Exception:
        return None


def previous_month_close(history, current_date):

    if history.empty:
        return np.nan

    target = (
        pd.Timestamp(current_date)
        - pd.DateOffset(months=1)
    ).date()

    dates = pd.Series(
        [date_value(x) for x in history.index],
        index=history.index
    )

    valid = history.loc[
        dates <= target
    ].dropna(subset=["Close"])

    if valid.empty:
        return np.nan

    return float(valid["Close"].iloc[-1])


def calculate(
    symbol,
    daily_all,
    intraday_all,
    today
):

    ticker = symbol + ".NS"

    daily = ticker_data(
        daily_all,
        ticker
    )

    if daily.empty:
        return None

    daily = daily.dropna(
        subset=["Close"]
    )

    if daily.empty:
        return None

    intraday = ticker_data(
        intraday_all,
        ticker
    )

    # --------------------------------------------------------
    # LIVE PRICE
    # --------------------------------------------------------

    price = np.nan

    if (
        not intraday.empty
        and "Close" in intraday.columns
    ):

        p = intraday["Close"].dropna()

        if not p.empty:
            price = float(p.iloc[-1])

    if (
        not np.isfinite(price)
        or price <= 0
    ):

        price = float(
            daily["Close"].iloc[-1]
        )

    # --------------------------------------------------------
    # HISTORICAL DATA
    # --------------------------------------------------------

    dates = pd.Series(
        [date_value(x) for x in daily.index],
        index=daily.index
    )

    hist = daily.loc[
        dates < today
    ].copy()

    if hist.empty:
        hist = daily.copy()

    hist = hist.dropna(
        subset=["Close"]
    )

    if hist.empty:
        return None

    # --------------------------------------------------------
    # 1 DAY
    # --------------------------------------------------------

    base_1d = float(
        hist["Close"].iloc[-1]
    )

    one_day = (
        (price / base_1d) - 1
    ) * 100 if base_1d > 0 else np.nan

    # --------------------------------------------------------
    # 1 WEEK
    # --------------------------------------------------------

    if len(hist) >= 5:

        base_1w = float(
            hist["Close"].iloc[-5]
        )

        one_week = (
            (price / base_1w) - 1
        ) * 100 if base_1w > 0 else np.nan

    else:
        one_week = np.nan

    # --------------------------------------------------------
    # 1 MONTH
    # --------------------------------------------------------

    base_1m = previous_month_close(
        hist,
        today
    )

    one_month = (
        (price / base_1m) - 1
    ) * 100 if (
        np.isfinite(base_1m)
        and base_1m > 0
    ) else np.nan

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    ema_series = hist["Close"].copy()

    ema_series.loc[
        pd.Timestamp.now(
            tz="Asia/Kolkata"
        )
    ] = price

    ema21 = float(
        ema_series.ewm(
            span=21,
            adjust=False
        ).mean().iloc[-1]
    )

    ema50 = float(
        ema_series.ewm(
            span=50,
            adjust=False
        ).mean().iloc[-1]
    )

    ema200 = float(
        ema_series.ewm(
            span=200,
            adjust=False
        ).mean().iloc[-1]
    )

    # --------------------------------------------------------
    # 52 WEEK HIGH / LOW
    # ACTUAL HIGH / LOW
    # NO 220 DAY RULE
    # --------------------------------------------------------

    cutoff = (
        pd.Timestamp(today)
        - pd.Timedelta(days=365)
    ).date()

    hist_dates = pd.Series(
        [date_value(x) for x in hist.index],
        index=hist.index
    )

    h52 = hist.loc[
        hist_dates >= cutoff
    ].copy()

    if h52.empty:
        h52 = hist.copy()

    if "High" in h52.columns:

        highs = pd.to_numeric(
            h52["High"],
            errors="coerce"
        ).dropna()

        high52 = (
            float(highs.max())
            if not highs.empty
            else np.nan
        )

    else:
        high52 = np.nan

    if "Low" in h52.columns:

        lows = pd.to_numeric(
            h52["Low"],
            errors="coerce"
        ).dropna()

        low52 = (
            float(lows.min())
            if not lows.empty
            else np.nan
        )

    else:
        low52 = np.nan

    # --------------------------------------------------------
    # DISTANCES
    # --------------------------------------------------------

    from_high = (
        (price / high52) - 1
    ) * 100 if (
        np.isfinite(high52)
        and high52 > 0
    ) else np.nan

    from_low = (
        (price / low52) - 1
    ) * 100 if (
        np.isfinite(low52)
        and low52 > 0
    ) else np.nan

    from_ema = (
        (price / ema21) - 1
    ) * 100 if ema21 > 0 else np.nan

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    if (
        price > ema21
        and ema21 > ema50
        and ema50 > ema200
    ):

        trend = "Bullish"

    elif (
        price < ema21
        and ema21 < ema50
        and ema50 < ema200
    ):

        trend = "Bearish"

    else:

        trend = "Neutral"

    return {
        "Stock": symbol,
        "Price": price,
        "1D Return %": one_day,
        "1W Return %": one_week,
        "1M Return %": one_month,
        "21 EMA": ema21,
        "50 EMA": ema50,
        "200 EMA": ema200,
        "52W High": high52,
        "52W Low": low52,
        "From 52W High %": from_high,
        "From 52W Low %": from_low,
        "From 21 EMA %": from_ema,
        "Trend": trend
    }


# ============================================================
# SCAN
# ============================================================

if st.button(
    "🔍 Scan Nifty 100",
    type="primary"
):

    try:
        symbols = get_symbols()

    except Exception as e:

        st.error(
            "Unable to download Nifty 100 list."
        )
        st.error(str(e))
        st.stop()

    st.info(
        "Nifty 100 stocks found: "
        + str(len(symbols))
    )

    with st.spinner(
        "Downloading daily data..."
    ):

        try:
            daily = get_daily(symbols)

        except Exception as e:

            st.error(
                "Daily data download failed."
            )
            st.error(str(e))
            st.stop()

    with st.spinner(
        "Downloading latest prices..."
    ):

        try:
            intraday = get_intraday(symbols)

        except Exception:

            intraday = pd.DataFrame()

            st.warning(
                "Intraday data unavailable. "
                "Latest daily close will be used."
            )

    ist = ZoneInfo("Asia/Kolkata")

    now = datetime.now(ist)

    today = now.date()

    results = []

    progress = st.progress(0)

    total = len(symbols)

    for i, symbol in enumerate(symbols):

        try:

            result = calculate(
                symbol,
                daily,
                intraday,
                today
            )

            if result is not None:
                results.append(result)

        except Exception:
            pass

        if total > 0:
            progress.progress(
                int(
                    ((i + 1) / total) * 100
                )
            )

    progress.empty()

    if not results:

        st.error(
            "No stock data was calculated."
        )

        st.stop()

    # --------------------------------------------------------
    # DATAFRAME
    # --------------------------------------------------------

    df = pd.DataFrame(results)

    # EXACT COLUMN ORDER

    df = df[
        [
            "Stock",
            "Price",
            "1D Return %",
            "1W Return %",
            "1M Return %",
            "21 EMA",
            "50 EMA",
            "200 EMA",
            "52W High",
            "52W Low",
            "From 52W High %",
            "From 52W Low %",
            "From 21 EMA %",
            "Trend"
        ]
    ]

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    df = df.sort_values(
        by="From 21 EMA %",
        ascending=False,
        na_position="last"
    )

    df = df.reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # ROUND TO 2 DECIMALS
    # --------------------------------------------------------

    cols = [
        "Price",
        "1D Return %",
        "1W Return %",
        "1M Return %",
        "21 EMA",
        "50 EMA",
        "200 EMA",
        "52W High",
        "52W Low",
        "From 52W High %",
        "From 52W Low %",
        "From 21 EMA %"
    ]

    for c in cols:

        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        ).round(2)

    # --------------------------------------------------------
    # DISPLAY COPY
    # --------------------------------------------------------

    display_df = df.copy()

    for c in cols:

        display_df[c] = display_df[c].map(
            lambda x: (
                f"{x:.2f}"
                if pd.notna(x)
                else "—"
            )
        )

    # --------------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------------

    st.success(
        "🕐 Last updated: "
        + now.strftime(
            "%d-%m-%Y %I:%M:%S %p IST"
        )
    )

    st.subheader(
        "📋 Results — "
        + str(len(df))
        + " stocks"
    )

    # --------------------------------------------------------
    # TABLE
    #
    # Trend colour is handled by Streamlit column_config.
    # This avoids pandas Styler completely.
    # --------------------------------------------------------

    st.dataframe(
        display_df,
        use_container_width=True,
        height=650,
        hide_index=True,
        column_config={
            "Trend": st.column_config.TextColumn(
                "Trend"
            )
        }
    )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    csv_data = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Results CSV",
        data=csv_data,
        file_name="nifty100_screener.csv",
        mime="text/csv"
                )
