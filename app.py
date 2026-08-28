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
    "Latest available intraday price is used for calculations. "
    "Historical returns use NSE/Yahoo daily closing prices."
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
            "Nifty 100 CSV returned no data."
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
            "Symbol column not found in Nifty 100 CSV."
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

    data = yf.download(
        tickers=tickers,
        period="2y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True
    )

    return data


# =========================================================
# DOWNLOAD INTRADAY DATA
# =========================================================

@st.cache_data(ttl=120)
def download_intraday_data(symbols):

    tickers = [
        symbol + ".NS"
        for symbol in symbols
    ]

    data = yf.download(
        tickers=tickers,
        period="1d",
        interval="5m",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True
    )

    return data


# =========================================================
# EXTRACT ONE TICKER FROM YFINANCE DATA
# =========================================================

def get_ticker_data(data, ticker):

    if data is None:
        return pd.DataFrame()

    if not isinstance(data, pd.DataFrame):
        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    try:

        # -------------------------------------------------
        # MULTIINDEX COLUMNS
        # -------------------------------------------------

        if isinstance(
            data.columns,
            pd.MultiIndex
        ):

            level0 = (
                data.columns
                .get_level_values(0)
            )

            level1 = (
                data.columns
                .get_level_values(1)
            )

            # Format:
            # TICKER -> OHLC
            if ticker in level0:

                result = data[
                    ticker
                ].copy()

            # Format:
            # OHLC -> TICKER
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

        required_columns = [
            "Open",
            "High",
            "Low",
            "Close"
        ]

        available_columns = [
            column
            for column in required_columns
            if column in result.columns
        ]

        if "Close" not in available_columns:
            return pd.DataFrame()

        result = result[
            available_columns
        ].copy()

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
# CONVERT INDEX VALUE TO DATE
# =========================================================

def get_date(index_value):

    try:

        ts = pd.Timestamp(
            index_value
        )

        if ts.tzinfo is not None:

            ts = ts.tz_convert(
                "Asia/Kolkata"
            )

        return ts.date()

    except Exception:

        return None


# =========================================================
# GET PREVIOUS MONTH CORRESPONDING CLOSE
#
# Example:
#
# 28-Aug-2026
#        ↓
# 28-Jul-2026
#
# If 28-Jul is a holiday/non-trading day,
# use the latest available trading day BEFORE
# or ON 28-Jul.
# =========================================================

def get_previous_month_close(
    history,
    current_date
):

    if history.empty:
        return np.nan

    current_timestamp = pd.Timestamp(
        current_date
    )

    target_timestamp = (
        current_timestamp
        - pd.DateOffset(months=1)
    )

    target_date = (
        target_timestamp.date()
    )

    history_dates = pd.Series(
        [
            get_date(x)
            for x in history.index
        ],
        index=history.index
    )

    valid = history.loc[
        history_dates <= target_date
    ].copy()

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


# =========================================================
# CALCULATE ONE STOCK
# =========================================================

def calculate_stock(
    symbol,
    daily_data,
    intraday_data,
    today
):

    ticker = symbol + ".NS"

    # =====================================================
    # DAILY DATA
    # =====================================================

    daily = get_ticker_data(
        daily_data,
        ticker
    )

    if daily.empty:
        return None

    if "Close" not in daily.columns:
        return None

    daily = daily.dropna(
        subset=["Close"]
    ).copy()

    if daily.empty:
        return None

    # =====================================================
    # INTRADAY DATA
    # =====================================================

    intraday = get_ticker_data(
        intraday_data,
        ticker
    )

    # =====================================================
    # LIVE PRICE
    #
    # Latest 5-minute intraday price.
    #
    # If unavailable, latest daily close is used only
    # as fallback so the stock remains in the table.
    # =====================================================

    live_price = np.nan

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

    if (
        not np.isfinite(live_price)
        or live_price <= 0
    ):

        live_price = float(
            daily["Close"].iloc[-1]
        )

    # =====================================================
    # DAILY DATES
    # =====================================================

    daily_dates = pd.Series(
        [
            get_date(x)
            for x in daily.index
        ],
        index=daily.index
    )

    # =====================================================
    # HISTORICAL DATA
    #
    # Exclude today's daily row because current price
    # is being supplied separately from intraday data.
    # =====================================================

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

    # =====================================================
    # 1 DAY RETURN
    #
    # Previous trading day's closing price
    # to current live price.
    # =====================================================

    previous_close = np.nan

    if len(historical) >= 1:

        previous_close = float(
            historical["Close"].iloc[-1]
        )

    if (
        np.isfinite(previous_close)
        and previous_close > 0
    ):

        one_day_return = (
            (
                live_price /
                previous_close
            ) - 1
        ) * 100

    else:

        one_day_return = np.nan

    # =====================================================
    # 1 WEEK RETURN
    #
    # 5 trading sessions back.
    #
    # Example:
    # Friday close -> following Friday live price
    # =====================================================

    if len(historical) >= 5:

        week_base = float(
            historical["Close"].iloc[-5]
        )

        if week_base > 0:

            one_week_return = (
                (
                    live_price /
                    week_base
                ) - 1
            ) * 100

        else:

            one_week_return = np.nan

    else:

        one_week_return = np.nan

    # =====================================================
    # 1 MONTH RETURN
    #
    # CORRESPONDING CALENDAR DATE OF PREVIOUS MONTH.
    #
    # Example:
    #
    # 28-Aug closing/live price
    # compared with
    # 28-Jul closing price.
    #
    # If 28-Jul was not a trading day, the latest
    # available trading close on/before 28-Jul is used.
    # =====================================================

    month_base = get_previous_month_close(
        historical,
        today
    )

    if (
        np.isfinite(month_base)
        and month_base > 0
    ):

        one_month_return = (
            (
                live_price /
                month_base
            ) - 1
        ) * 100

    else:

        one_month_return = np.nan

    # =====================================================
    # EMA DATA
    #
    # Include live price as today's latest price.
    # =====================================================

    close_for_ema = (
        historical["Close"]
        .copy()
    )

    live_index = pd.Timestamp.now(
        tz="Asia/Kolkata"
    )

    live_series = pd.Series(
        [live_price],
        index=[live_index]
    )

    close_for_ema = pd.concat(
        [
            close_for_ema,
            live_series
        ]
    )

    # =====================================================
    # 21 EMA
    # =====================================================

    ema21 = float(
        close_for_ema
        .ewm(
            span=21,
            adjust=False
        )
        .mean()
        .iloc[-1]
    )

    # =====================================================
    # 50 EMA
    # =====================================================

    ema50 = float(
        close_for_ema
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
        .iloc[-1]
    )

    # =====================================================
    # 200 EMA
    # =====================================================

    ema200 = float(
        close_for_ema
        .ewm(
            span=200,
            adjust=False
        )
        .mean()
        .iloc[-1]
    )

    # =====================================================
    # 52 WEEK ACTUAL HIGH / LOW
    #
    # IMPORTANT:
    # Uses actual daily High and Low.
    #
    # NOT closing prices.
    #
    # NO 220 TRADING-DAY CRITERIA.
    # =====================================================

    cutoff_date = (
        pd.Timestamp(today)
        - pd.Timedelta(days=365)
    ).date()

    historical_dates = pd.Series(
        [
            get_date(x)
            for x in historical.index
        ],
        index=historical.index
    )

    last_52_weeks = historical.loc[
        historical_dates >= cutoff_date
    ].copy()

    if last_52_weeks.empty:

        last_52_weeks = historical.copy()

    # -----------------------------------------------------
    # ACTUAL HIGH
    # -----------------------------------------------------

    if "High" in last_52_weeks.columns:

        high_values = pd.to_numeric(
            last_52_weeks["High"],
            errors="coerce"
        ).dropna()

        if not high_values.empty:

            week52_high = float(
                high_values.max()
            )

        else:

            week52_high = np.nan

    else:

        week52_high = np.nan

    # -----------------------------------------------------
    # ACTUAL LOW
    # -----------------------------------------------------

    if "Low" in last_52_weeks.columns:

        low_values = pd.to_numeric(
            last_52_weeks["Low"],
            errors="coerce"
        ).dropna()

        if not low_values.empty:

            week52_low = float(
                low_values.min()
            )

        else:

            week52_low = np.nan

    else:

        week52_low = np.nan

    # =====================================================
    # DISTANCE FROM 52 WEEK HIGH
    # =====================================================

    if (
        np.isfinite(week52_high)
        and week52_high > 0
    ):

        from_52w_high = (
            (
                live_price /
                week52_high
            ) - 1
        ) * 100

    else:

        from_52w_high = np.nan

    # =====================================================
    # DISTANCE FROM 52 WEEK LOW
    # =====================================================

    if (
        np.isfinite(week52_low)
        and week52_low > 0
    ):

        from_52w_low = (
            (
                live_price /
                week52_low
            ) - 1
        ) * 100

    else:

        from_52w_low = np.nan

    # =====================================================
    # DISTANCE FROM 21 EMA
    # =====================================================

    if (
        np.isfinite(ema21)
        and ema21 > 0
    ):

        from_21_ema = (
            (
                live_price /
                ema21
            ) - 1
        ) * 100

    else:

        from_21_ema = np.nan

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
    # RETURN RESULT
    # =====================================================

    result = {
        "Stock": symbol,
        "Price": live_price,

        "1D Return %": one_day_return,
        "1W Return %": one_week_return,
        "1M Return %": one_month_return,

        "21 EMA": ema21,
        "50 EMA": ema50,
        "200 EMA": ema200,

        "52W High": week52_high,
        "52W Low": week52_low,

        "From 52W High %": from_52w_high,
        "From 52W Low %": from_52w_low,
        "From 21 EMA %": from_21_ema,

        "Trend": trend
    }

    return result


# =========================================================
# TREND COLOUR FUNCTION
#
# Using Styler.apply instead of applymap/map.
# This is more compatible with different pandas versions.
# =========================================================

def colour_trend_column(column):

    styles = []

    for value in column:

        if value == "Bullish":

            styles.append(
                "background-color: #198754; "
                "color: white; "
                "font-weight: bold; "
                "text-align: center;"
            )

        elif value == "Neutral":

            styles.append(
                "background-color: #F5B642; "
                "color: black; "
                "font-weight: bold; "
                "text-align: center;"
            )

        elif value == "Bearish":

            styles.append(
                "background-color: #DC3545; "
                "color: white; "
                "font-weight: bold; "
                "text-align: center;"
            )

        else:

            styles.append("")

    return styles


# =========================================================
# SCAN BUTTON
# =========================================================

if st.button(
    "🔍 Scan Nifty 100",
    type="primary"
):

    # =====================================================
    # GET NIFTY 100
    # =====================================================

    try:

        symbols = get_nifty100_list()

    except Exception as e:

        st.error(
            "Unable to get Nifty 100 list."
        )

        st.error(
            f"Error: {e}"
        )

        st.stop()

    st.info(
        f"Current Nifty 100 list: {len(symbols)} stocks"
    )

    # =====================================================
    # DOWNLOAD DAILY DATA
    # =====================================================

    with st.spinner(
        "Downloading daily data..."
    ):

        try:

            daily_data = download_daily_data(
                symbols
            )

        except Exception as e:

            st.error(
                "Daily data download failed."
            )

            st.error(
                f"Error: {e}"
            )

            st.stop()

    # =====================================================
    # DOWNLOAD INTRADAY DATA
    # =====================================================

    with st.spinner(
        "Downloading latest live prices..."
    ):

        try:

            intraday_data = (
                download_intraday_data(
                    symbols
                )
            )

        except Exception as e:

            st.warning(
                "Intraday data could not be downloaded. "
                "Latest daily close will be used as fallback."
            )

            intraday_data = pd.DataFrame()

    # =====================================================
    # CURRENT IST TIME
    # =====================================================

    ist = ZoneInfo(
  
