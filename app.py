import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import csv
import io
from datetime import datetime
from zoneinfo import ZoneInfo


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="My Stock Screener",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("📊 My Stock Screener")
st.subheader("Nifty 100 Technical Screener")

st.caption(
    "Latest available intraday price is used where available."
)


# =========================================================
# NIFTY 100 URL
# =========================================================

NIFTY100_URL = (
    "https://www.niftyindices.com/IndexConstituent/"
    "ind_nifty100list.csv"
)


# =========================================================
# GET NIFTY 100 LIST
# =========================================================

@st.cache_data(ttl=86400)
def get_nifty100_list():

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "Accept": (
            "text/csv,application/csv,"
            "text/plain,*/*"
        ),
        "Referer": (
            "https://www.niftyindices.com/"
        )
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
        str(x).strip()
        for x in rows[0]
    ]

    symbol_index = None

    for i, column in enumerate(header):

        if column.lower() == "symbol":

            symbol_index = i
            break

    if symbol_index is None:

        raise ValueError(
            "Symbol column was not found."
        )

    symbols = []

    for row in rows[1:]:

        if len(row) <= symbol_index:
            continue

        symbol = row[symbol_index].strip()

        if symbol:
            symbols.append(symbol)

    symbols = list(
        dict.fromkeys(symbols)
    )

    return symbols


# =========================================================
# DOWNLOAD DAILY DATA
# =========================================================

@st.cache_data(ttl=300)
def download_daily_data(symbols):

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


# =========================================================
# DOWNLOAD 5-MINUTE DATA
# =========================================================

@st.cache_data(ttl=120)
def download_intraday_data(symbols):

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


# =========================================================
# EXTRACT ONE TICKER
# =========================================================

def extract_ticker(
    data,
    ticker
):

    if data is None:
        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    try:

        # -------------------------------------------------
        # MultiIndex
        # -------------------------------------------------

        if isinstance(
            data.columns,
            pd.MultiIndex
        ):

            level0 = list(
                data.columns
                .get_level_values(0)
            )

            level1 = list(
                data.columns
                .get_level_values(1)
            )

            # Ticker is first level
            if ticker in level0:

                result = data[
                    ticker
                ].copy()

            # Ticker is second level
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

        # -------------------------------------------------
        # Flatten any remaining MultiIndex
        # -------------------------------------------------

        if isinstance(
            result.columns,
            pd.MultiIndex
        ):

            result.columns = [
                str(c[-1])
                for c in result.columns
            ]

        # -------------------------------------------------
        # Required columns
        # -------------------------------------------------

        wanted = [
            "Open",
            "High",
            "Low",
            "Close"
        ]

        available = [
            c
            for c in wanted
            if c in result.columns
        ]

        if "Close" not in available:

            return pd.DataFrame()

        result = result[
            available
        ].copy()

        # -------------------------------------------------
        # Numeric conversion
        # -------------------------------------------------

        for column in result.columns:

            result[column] = pd.to_numeric(
                result[column],
                errors="coerce"
            )

        result = result.dropna(
            how="all"
        )

        return result

    except Exception:

        return pd.DataFrame()


# =========================================================
# CONVERT INDEX TO INDIA DATE
# =========================================================

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


# =========================================================
# PREVIOUS MONTH CLOSE
# =========================================================

def previous_month_close(
    history,
    current_date
):

    if history.empty:
        return np.nan

    # Example:
    #
    # 28-Aug
    #     ↓
    # 28-Jul
    #
    # If 28-Jul is not a trading day,
    # use the latest trading day before it.

    current = pd.Timestamp(
        current_date
    )

    target = (
        current -
        pd.DateOffset(months=1)
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
    ].copy()

    valid = valid.dropna(
        subset=["Close"]
    )

    if valid.empty:
        return np.nan

    return float(
        valid["Close"].iloc[-1]
    )


# =========================================================
# CALCULATE STOCK
# =========================================================

def calculate_stock(
    symbol,
    daily_data,
    intraday_data,
    today
):

    ticker = symbol + ".NS"

    # -----------------------------------------------------
    # DAILY
    # -----------------------------------------------------

    daily = extract_ticker(
        daily_data,
        ticker
    )

    if daily.empty:
        return None, "No daily data"

    if "Close" not in daily.columns:
        return None, "No daily close"

    daily = daily.dropna(
        subset=["Close"]
    ).copy()

    if daily.empty:
        return None, "Empty daily data"

    # -----------------------------------------------------
    # COMPLETED DAILY HISTORY
    # -----------------------------------------------------

    daily_dates = pd.Series(
        [
            india_date(x)
            for x in daily.index
        ],
        index=daily.index
    )

    historical = daily.loc[
        daily_dates < today
    ].copy()

    if historical.empty:

        historical = daily.copy()

    # =====================================================
    # LIVE PRICE
    # =====================================================

    live_price = np.nan

    intraday = extract_ticker(
        intraday_data,
        ticker
    )

    if (
        not intraday.empty
        and "Close" in intraday.columns
    ):

        intraday_close = (
            intraday["Close"]
            .dropna()
        )

        if not intraday_close.empty:

            live_price = float(
                intraday_close.iloc[-1]
            )

    # -----------------------------------------------------
    # FALLBACK
    # -----------------------------------------------------

    if (
        not np.isfinite(live_price)
        or live_price <= 0
    ):

        live_price = float(
            daily["Close"].iloc[-1]
        )

    # =====================================================
    # 1 DAY RETURN
    # =====================================================

    previous_close = float(
        historical["Close"].iloc[-1]
    )

    if previous_close > 0:

        one_day = (
            live_price /
            previous_close -
            1
        ) * 100

    else:

        one_day = np.nan

    # =====================================================
    # 1 WEEK RETURN
    #
    # 5 TRADING SESSIONS
    # =====================================================

    if len(historical) >= 6:

        week_close = float(
            historical["Close"].iloc[-6]
        )

        if week_close > 0:

            one_week = (
                live_price /
                week_close -
                1
            ) * 100

        else:

            one_week = np.nan

    else:

        one_week = np.nan

    # =====================================================
    # 1 MONTH RETURN
    #
    # PREVIOUS MONTH'S CORRESPONDING DATE
    # =====================================================

    month_close = previous_month_close(
        historical,
        today
    )

    if (
        np.isfinite(month_close)
        and month_close > 0
    ):

        one_month = (
            live_price /
            month_close -
            1
        ) * 100

    else:

        one_month = np.nan

    # =====================================================
    # EMA
    # =====================================================

    closes = historical[
        "Close"
    ].copy()

    live_row = pd.Series(
        [live_price],
        index=[
            pd.Timestamp.now()
        ]
    )

    closes = pd.concat(
        [
            closes,
            live_row
        ]
    )

    ema21 = (
        closes
        .ewm(
            span=21,
            adjust=False
        )
        .mean()
        .iloc[-1]
    )

    ema50 = (
        closes
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
        .iloc[-1]
    )

    ema200 = (
        closes
        .ewm(
            span=200,
            adjust=False
        )
        .mean()
        .iloc[-1]
    )

    # =====================================================
    # 52 WEEK HIGH / LOW
    #
    # IMPORTANT:
    # HIGH = ACTUAL DAILY HIGH
    # LOW  = ACTUAL DAILY LOW
    # =====================================================

    cutoff = (
        pd.Timestamp(today)
        - pd.Timedelta(days=365)
    ).date()

    if (
        "High" in historical.columns
        and
        "Low" in historical.columns
    ):

        hist_dates = pd.Series(
            [
                india_date(x)
                for x in historical.index
            ],
            index=historical.index
        )

        last_52w = historical.loc[
            hist_dates >= cutoff
        ].copy()

        if last_52w.empty:

            last_52w = historical.copy()

        high_series = pd.to_numeric(
            last_52w["High"],
            errors="coerce"
        ).dropna()

        low_series = pd.to_numeric(
            last_52w["Low"],
            errors="coerce"
        ).dropna()

        if not high_series.empty:

            high_52w = float(
                high_series.max()
            )

        else:

            high_52w = np.nan

        if not low_series.empty:

            low_52w = float(
                low_series.min()
            )

        else:

            low_52w = np.nan

    else:

        high_52w = np.nan
        low_52w = np.nan

    # =====================================================
    # DISTANCE FROM 52W HIGH
    # =====================================================

    if (
        np.isfinite(high_52w)
        and high_52w > 0
    ):

        from_high = (
            live_price /
            high_52w -
            1
        ) * 100

    else:

        from_high = np.nan

    # =====================================================
    # DISTANCE FROM 52W LOW
    # =====================================================

    if (
        np.isfinite(low_52w)
        and low_52w > 0
    ):

        from_low = (
            live_price /
            low_52w -
            1
        ) * 100

    else:

        from_low = np.nan

    # =====================================================
    # DISTANCE FROM 21 EMA
    # =====================================================

    if (
        np.isfinite(ema21)
        and ema21 > 0
    ):

        from_ema21 = (
            live_price /
            ema21 -
            1
        ) * 100

    else:

        from_ema21 = np.nan

    # =====================================================
    # TREND
    # =====================================================

    if (
        live_price > ema21
        and ema21 > ema50
        and ema50 > ema200
    ):

        trend = "Bullish"

    elif (
        live_price < ema21
        and ema21 < ema50
        and ema50 < ema200
    ):

        trend = "Bearish"

    else:

        trend = "Neutral"

    # =====================================================
    # RESULT
    # =====================================================

    result = {

        "Stock": symbol,

        "Price": live_price,

        "1D Return %": one_day,

        "1W Return %": one_week,

        "1M Return %": one_month,

        "21 EMA": float(ema21),

        "50 EMA": float(ema50),

        "200 EMA": float(ema200),

        "52W High": high_52w,

        "52W Low": low_52w,

        "From 52W High %": from_high,

        "From 52W Low %": from_low,

        "From 21 EMA %": from_ema21,

        "Trend": trend
    }

    return result, None


# =========================================================
# TREND COLOUR
# =========================================================

def colour_trend(
    value
):

    if value == "Bullish":

        return (
            "background-color: #198754;"
            "color: white;"
            "font-weight: bold;"
        )

    elif value == "Bearish":

        return (
            "background-color: #DC3545;"
            "color: white;"
            "font-weight: bold;"
        )

    elif value == "Neutral":

        return (
            "background-color: #F5B642;"
            "color: black;"
            "font-weight: bold;"
        )

    return ""


# =========================================================
# MAIN SCAN
# =========================================================

if st.button(
    "🔍 Scan Nifty 100"
):

    # =====================================================
    # GET STOCK LIST
    # =====================================================

    try:

        symbols = get_nifty100_list()

    except Exception as e:

        st.error(
            "Unable to download Nifty 100 list."
        )

        st.exception(e)

        st.stop()

    st.info(
        f"Nifty 100 stocks found: {len(symbols)}"
    )

    # =====================================================
    # DOWNLOAD DAILY
    # =====================================================

    with st.spinner(
        "Downloading daily data..."
    ):

        try:

            daily_data = (
                download_daily_data(
                    symbols
                )
            )

        except Exception as e:

            st.error(
                "Daily Yahoo Finance download failed."
            )

            st.exception(e)

            st.stop()

    # =====================================================
    # DOWNLOAD INTRADAY
    # =====================================================

    with st.spinner(
        "Downloading live/intraday prices..."
    ):

        try:

            intraday_data = (
                download_intraday_data(
                    symbols
                )
            )

        except Exception as e:

            st.warning(
                "Intraday download failed. "
                "The latest daily close will be used "
                "as fallback."
            )

            intraday_data = pd.DataFrame()

    # =====================================================
    # CURRENT INDIA DATE
    # =====================================================

    ist = ZoneInfo(
        "Asia/Kolkata"
    )

    now = datetime.now(
        ist
    )

    today = now.date()

    # =====================================================
    # SCAN
    # =====================================================

    results = []

    failed = []

    progress = st.progress(0)

    total = len(symbols)

    for i, symbol in enumerate(symbols):

        try:

            result, reason = calculate_stock(
                symbol,
                daily_data,
                intraday_data,
                today
            )

            if result is not None:

                results.append(result)

            else:

                failed.append(
                    f"{symbol}: {reason}"
                )

        except Exception as e:

            failed.append(
                f"{symbol}: {str(e)}"
            )

        progress.progress(
            (i + 1) / total
        )

    progress.empty()

    # =====================================================
    # RESULTS CHECK
    # =====================================================

    if len(results) == 0:

        st.error(
            "No stocks could be calculated."
        )

        if failed:

            st.write(
                "First errors:"
            )

            st.code(
                "\n".join(
                    failed[:20]
                )
            )

        st.stop()

    # =====================================================
    # DATAFRAME
    # =====================================================

    df = pd.DataFrame(
        results
    )

    # =====================================================
    # COLUMN ORDER
    # =====================================================

    columns = [

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

    df = df[columns]

    # =====================================================
    # NUMERIC COLUMNS
    # =====================================================

    number_columns = [

        "Price",

        "1D Return %",
        "1W Return %",
        "1M Return %",

        "21 EMA",
        "50 EMA",
        "200 EMA",

        "52W High",
        "52W Low",

 
