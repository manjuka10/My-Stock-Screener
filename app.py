import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import csv
import io
from datetime import datetime
from zoneinfo import ZoneInfo


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="My Stock Screener",
    page_icon="📊",
    layout="wide"
)

st.title("📊 My Stock Screener")
st.subheader("Nifty 100 Options Selling Screener")

st.caption(
    "Current Price = latest available intraday price."
)


# ============================================================
# NIFTY 100 URL
# ============================================================

NIFTY100_URL = (
    "https://www.niftyindices.com/IndexConstituent/"
    "ind_nifty100list.csv"
)


# ============================================================
# GET NIFTY 100
# ============================================================

@st.cache_data(ttl=86400)
def get_nifty100():

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

    rows = list(
        csv.reader(
            io.StringIO(text)
        )
    )

    if len(rows) < 2:
        raise ValueError(
            "Nifty 100 list is empty."
        )

    header = [
        str(x).strip().lower()
        for x in rows[0]
    ]

    if "symbol" not in header:
        raise ValueError(
            "Symbol column not found."
        )

    symbol_index = header.index("symbol")

    symbols = []

    for row in rows[1:]:

        if len(row) <= symbol_index:
            continue

        symbol = row[symbol_index].strip()

        if symbol:
            symbols.append(symbol)

    return list(
        dict.fromkeys(symbols)
    )


# ============================================================
# DAILY DATA
# ============================================================

@st.cache_data(ttl=300)
def download_daily(symbols):

    tickers = [
        symbol + ".NS"
        for symbol in symbols
    ]

    return yf.download(
        tickers=tickers,
        period="2y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True
    )


# ============================================================
# INTRADAY DATA
# ============================================================

@st.cache_data(ttl=120)
def download_intraday(symbols):

    tickers = [
        symbol + ".NS"
        for symbol in symbols
    ]

    return yf.download(
        tickers=tickers,
        period="1d",
        interval="5m",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True
    )


# ============================================================
# GET INDIVIDUAL STOCK DATA
# ============================================================

def get_stock_data(data, ticker):

    if data is None:
        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    try:

        if isinstance(
            data.columns,
            pd.MultiIndex
        ):

            level0 = data.columns.get_level_values(0)
            level1 = data.columns.get_level_values(1)

            if ticker in level0:

                result = data[ticker].copy()

            elif ticker in level1:

                result = data.xs(
                    ticker,
                    axis=1,
                    level=1
                ).copy()

            else:

                return pd.DataFrame()

        else:

            result = data.copy()

        wanted = [
            "Open",
            "High",
            "Low",
            "Close"
        ]

        available = [
            column
            for column in wanted
            if column in result.columns
        ]

        if "Close" not in available:
            return pd.DataFrame()

        result = result[available].copy()

        for column in available:

            result[column] = pd.to_numeric(
                result[column],
                errors="coerce"
            )

        return result.dropna(
            how="all"
        )

    except Exception:

        return pd.DataFrame()


# ============================================================
# GET INDIA DATE
# ============================================================

def india_date(value):

    try:

        ts = pd.Timestamp(value)

        if ts.tzinfo is not None:

            ts = ts.tz_convert(
                "Asia/Kolkata"
            )

        return ts.date()

    except Exception:

        return None


# ============================================================
# PREVIOUS MONTH CLOSE
#
# Example:
#
# 28-Aug current price
# 28-Jul closing price
#
# If target date is holiday/weekend,
# previous available trading day is used.
# ============================================================

def previous_month_close(
    history,
    current_date
):

    if history.empty:
        return np.nan

    target = (
        pd.Timestamp(current_date)
        - pd.DateOffset(months=1)
    )

    target_date = target.date()

    dates = pd.Series(
        [
            india_date(x)
            for x in history.index
        ],
        index=history.index
    )

    valid = history.loc[
        dates <= target_date
    ]

    if valid.empty:
        return np.nan

    valid = valid.dropna(
        subset=["Close"]
    )

    if valid.empty:
        return np.nan

    return float(
        valid["Close"].iloc[-1]
    )


# ============================================================
# CALCULATE ONE STOCK
# ============================================================

def calculate_stock(
    symbol,
    daily_all,
    intraday_all,
    today
):

    ticker = symbol + ".NS"

    # --------------------------------------------------------
    # DAILY
    # --------------------------------------------------------

    daily = get_stock_data(
        daily_all,
        ticker
    )

    if daily.empty:
        return None

    daily = daily.dropna(
        subset=["Close"]
    ).copy()

    if daily.empty:
        return None

    # --------------------------------------------------------
    # INTRADAY
    # --------------------------------------------------------

    intraday = get_stock_data(
        intraday_all,
        ticker
    )

    # --------------------------------------------------------
    # CURRENT LIVE PRICE
    # --------------------------------------------------------

    current_price = np.nan

    if (
        not intraday.empty
        and "Close" in intraday.columns
    ):

        prices = (
            intraday["Close"]
            .dropna()
        )

        if not prices.empty:

            current_price = float(
                prices.iloc[-1]
            )

    # Fallback

    if (
        not np.isfinite(current_price)
        or current_price <= 0
    ):

        current_price = float(
            daily["Close"].iloc[-1]
        )

    # --------------------------------------------------------
    # DAILY DATES
    # --------------------------------------------------------

    daily_dates = pd.Series(
        [
            india_date(x)
            for x in daily.index
        ],
        index=daily.index
    )

    # --------------------------------------------------------
    # COMPLETED DAILY DATA
    # --------------------------------------------------------

    historical = daily.loc[
        daily_dates < today
    ].copy()

    if historical.empty:

        historical = daily.copy()

    historical = historical.dropna(
        subset=["Close"]
    )

    if historical.empty:
        return None

    # ========================================================
    # 1 WEEK RETURN
    #
    # Current price compared with 5 trading sessions back.
    # ========================================================

    if len(historical) >= 5:

        week_base = float(
            historical["Close"].iloc[-5]
        )

        if week_base > 0:

            one_week = (
                current_price /
                week_base
                - 1
            ) * 100

        else:

            one_week = np.nan

    else:

        one_week = np.nan

    # ========================================================
    # 1 MONTH RETURN
    #
    # Previous month's corresponding date.
    # ========================================================

    month_base = previous_month_close(
        historical,
        today
    )

    if (
        np.isfinite(month_base)
        and month_base > 0
    ):

        one_month = (
            current_price /
            month_base
            - 1
        ) * 100

    else:

        one_month = np.nan

    # ========================================================
    # EMA
    # ========================================================

    ema_data = historical["Close"].copy()

    ema_data.loc[
        pd.Timestamp.now(
            tz="Asia/Kolkata"
        )
    ] = current_price

    ema21 = float(
        ema_data.ewm(
            span=21,
            adjust=False
        ).mean().iloc[-1]
    )

    ema50 = float(
        ema_data.ewm(
            span=50,
            adjust=False
        ).mean().iloc[-1]
    )

    ema200 = float(
        ema_data.ewm(
            span=200,
            adjust=False
        ).mean().iloc[-1]
    )

    # ========================================================
    # DISTANCE FROM 21 EMA
    #
    # Based on CURRENT LIVE PRICE.
    # ========================================================

    if (
        np.isfinite(ema21)
        and ema21 > 0
    ):

        from_21_ema = (
            (
                current_price -
                ema21
            ) / ema21
        ) * 100

    else:

        from_21_ema = np.nan

    # ========================================================
    # 52 WEEK HIGH / LOW
    #
    # Historical High/Low + today's intraday High/Low.
    # ========================================================

    cutoff = (
        pd.Timestamp(today)
        - pd.Timedelta(days=365)
    ).date()

    historical_dates = pd.Series(
        [
            india_date(x)
            for x in historical.index
        ],
        index=historical.index
    )

    last_52w = historical.loc[
        historical_dates >= cutoff
    ].copy()

    if last_52w.empty:

        last_52w = historical.copy()

    # --------------------------------------------------------
    # HIGH
    # --------------------------------------------------------

    high_values = []

    if "High" in last_52w.columns:

        values = pd.to_numeric(
            last_52w["High"],
            errors="coerce"
        ).dropna()

        if not values.empty:

            high_values.append(
                float(values.max())
            )

    if (
        not intraday.empty
        and "High" in intraday.columns
    ):

        values = pd.to_numeric(
            intraday["High"],
            errors="coerce"
        ).dropna()

        if not values.empty:

            high_values.append(
                float(values.max())
            )

    if high_values:

        week52_high = max(
            high_values
        )

    else:

        week52_high = np.nan

    # --------------------------------------------------------
    # LOW
    # --------------------------------------------------------

    low_values = []

    if "Low" in last_52w.columns:

        values = pd.to_numeric(
            last_52w["Low"],
            errors="coerce"
        ).dropna()

        if not values.empty:

            low_values.append(
                float(values.min())
            )

    if (
        not intraday.empty
        and "Low" in intraday.columns
    ):

        values = pd.to_numeric(
            intraday["Low"],
            errors="coerce"
        ).dropna()

        if not values.empty:

            low_values.append(
                float(values.min())
            )

    if low_values:

        week52_low = min(
            low_values
        )

    else:

        week52_low = np.nan

    # ========================================================
    # DISTANCE FROM 52 WEEK HIGH
    # ========================================================

    if (
        np.isfinite(week52_high)
        and week52_high > 0
    ):

        from_52w_high = (
            current_price /
            week52_high
            - 1
        ) * 100

    else:

        from_52w_high = np.nan

    # ========================================================
    # DISTANCE FROM 52 WEEK LOW
    # ========================================================

    if (
        np.isfinite(week52_low)
        and week52_low > 0
    ):

        from_52w_low = (
            current_price /
            week52_low
            - 1
        ) * 100

    else:

        from_52w_low = np.nan

    # ========================================================
    # TREND
    # ========================================================

    if (
        current_price > ema21
        and ema21 > ema50
        and ema50 > ema200
    ):

        trend = "Bullish"

    elif (
        current_price < ema21
        and ema21 < ema50
        and ema50 < ema200
    ):

        trend = "Bearish"

    else:

        trend = "Neutral"

    # ========================================================
    # RETURN
    # ========================================================

    return {
        "Stock": symbol,
        "Price": current_price,
        "1W Return %": one_week,
        "1M Return %": one_month,
        "21 EMA": ema21,
        "50 EMA": ema50,
        "200 EMA": ema200,
        "From 52W High %": from_52w_high,
        "From 52W Low %": from_52w_low,
        "From 21 EMA %": from_21_ema,
        "Trend": trend
    }


# ============================================================
# TREND COLOUR FUNCTION
# ============================================================

def trend_colour(value):

    if value == "Bullish":
        return "background-color: green; color: white;"

    if value == "Bearish":
        return "background-color: red; color: white;"

    if value == "Neutral":
        return "background-color: orange; color: black;"

    return ""


# ============================================================
# SCAN
# ============================================================

if st.button(
    "🔍 Scan Nifty 100",
    type="primary"
):

    # --------------------------------------------------------
    # NIFTY 100
    # --------------------------------------------------------

    try:

        symbols = get_nifty100()

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

    # --------------------------------------------------------
    # DAILY DATA
    # --------------------------------------------------------

    with st.spinner(
        "Downloading daily data..."
    ):

        try:

            daily_data = download_daily(
                symbols
            )

        except Exception as e:

            st.error(
                "Daily data download failed."
            )

            st.error(str(e))

            st.stop()

    # --------------------------------------------------------
    # INTRADAY DATA
    # --------------------------------------------------------

    with st.spinner(
        "Downloading latest prices..."
    ):

        try:

            intraday_data = download_intraday(
                symbols
            )

        except Exception:

            intraday_data = pd.DataFrame()

            st.warning(
                "Intraday data unavailable. "
                "Latest daily close will be used."
            )

    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    ist = ZoneInfo(
        "Asia/Kolkata"
    )

    now_ist = datetime.now(
        ist
    )

    today = now_ist.date()

    # --------------------------------------------------------
    # CALCULATE
    # --------------------------------------------------------

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

        if total > 0:

            progress.progress(
                int(
                    ((i + 1) / total) * 100
                )
            )

    progress.empty()

    # --------------------------------------------------------
    # CHECK RESULTS
    # --------------------------------------------------------

    if len(results) == 0:

        st.error(
            "No stock data was calculated."
        )

        st.stop()

    # --------------------------------------------------------
    # DATAFRAME
    # --------------------------------------------------------

    df = pd.DataFrame(
        results
    )

    # ========================================================
    # EXACT COLUMN ORDER
    # ========================================================

    df = df[
        [
            "Stock",
            "Price",
            "1W Return %",
            "1M Return %",
            "21 EMA",
            "50 EMA",
            "200 EMA",
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

    numeric_columns = [
        "Price",
        "1W Return %",
        "1M Return %",
        "21 EMA",
        "50 EMA",
        "200 EMA",
        "From 52W High %",
        "From 52W Low %",
        "From 21 EMA %"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        ).round(2)

    # ========================================================
    # LAST UPDATED
    # ========================================================

    updated_time = now_ist.strftime(
        "%d-%m-%Y %I:%M:%S %p IST"
    )

    st.success(
        "🕐 Last updated: "
        + updated_time
    )

    # ========================================================
    # RESULTS
    # ========================================================

    st.subheader(
        "📋 Results — "
        + str(len(df))
        + " stocks"
    )

    # ========================================================
    # TABLE
    #
    # IMPORTANT:
    # NO pandas Styler.
    # Normal Streamlit dataframe is used so the table
    # cannot disappear because of Styler compatibility.
    # ========================================================

    st.dataframe(
        df,
        use_container_width=True,
        height=650,
        hide_index=True,
        column_config={
            "Price": st.column_config.N
