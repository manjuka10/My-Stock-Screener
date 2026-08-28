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
    "Price uses the latest available Yahoo Finance intraday price."
)


# =========================================================
# NIFTY 100 URL
# =========================================================

NIFTY100_URL = (
    "https://www.niftyindices.com/IndexConstituent/"
    "ind_nifty100list.csv"
)


# =========================================================
# SPECIAL YAHOO SYMBOL MAPPING
# =========================================================

# Normally NSE symbol + .NS is used.
# This dictionary allows special symbols to be handled
# separately if required.

YAHOO_SYMBOL_MAP = {
    "TMCV": "TMCV.NS",
}


# =========================================================
# GET YAHOO SYMBOL
# =========================================================

def get_yahoo_symbol(symbol):

    if symbol in YAHOO_SYMBOL_MAP:
        return YAHOO_SYMBOL_MAP[symbol]

    return symbol + ".NS"


# =========================================================
# GET NIFTY 100
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
        get_yahoo_symbol(symbol)
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
        get_yahoo_symbol(symbol)
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
# GET INDIVIDUAL TICKER DATA
# =========================================================

def get_ticker_data(data, ticker):

    if data is None:
        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    try:

        # -------------------------------------------------
        # MULTI INDEX
        # -------------------------------------------------

        if isinstance(data.columns, pd.MultiIndex):

            level0 = data.columns.get_level_values(0)
            level1 = data.columns.get_level_values(1)

            # Format:
            # TICKER -> OHLC

            if ticker in level0:

                result = data[ticker].copy()

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

        # -------------------------------------------------
        # REQUIRED COLUMNS
        # -------------------------------------------------

        required = [
            "Open",
            "High",
            "Low",
            "Close"
        ]

        available = [
            c
            for c in required
            if c in result.columns
        ]

        if "Close" not in available:
            return pd.DataFrame()

        result = result[available].copy()

        # -------------------------------------------------
        # NUMERIC
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
# GET DATE
# =========================================================

def get_date(index_value):

    try:

        ts = pd.Timestamp(index_value)

        if ts.tzinfo is not None:

            ts = ts.tz_convert(
                "Asia/Kolkata"
            )

        return ts.date()

    except Exception:

        return None


# =========================================================
# GET PREVIOUS MONTH CLOSE
# =========================================================

def get_previous_month_close(
    history,
    current_date
):

    if history.empty:
        return np.nan

    current_ts = pd.Timestamp(
        current_date
    )

    # Same calendar date one month earlier.
    #
    # Example:
    # 28-Aug -> 28-Jul
    #
    # If 28-Jul is a holiday/weekend,
    # use the latest available trading day
    # before/on that date.

    target_ts = (
        current_ts -
        pd.DateOffset(months=1)
    )

    target_date = target_ts.date()

    date_series = pd.Series(
        [
            get_date(x)
            for x in history.index
        ],
        index=history.index
    )

    valid = history.loc[
        date_series <= target_date
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

    ticker = get_yahoo_symbol(symbol)

    # =====================================================
    # DAILY DATA
    # =====================================================

    daily = get_ticker_data(
        daily_data,
        ticker
    )

    # =====================================================
    # INTRADAY DATA
    # =====================================================

    intraday = get_ticker_data(
        intraday_data,
        ticker
    )

    # =====================================================
    # CHECK DAILY DATA
    # =====================================================

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
    # LIVE PRICE
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

    # Fallback to latest daily close
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

    # Only completed trading days.
    historical = daily.loc[
        daily_dates < today
    ].copy()

    if historical.empty:

        historical = daily.copy()

    historical = historical.dropna(
        subset=["Close"]
    )

    # =====================================================
    # 1D RETURN
    # =====================================================

    if len(historical) >= 1:

        previous_close = float(
            historical["Close"].iloc[-1]
        )

        if previous_close > 0:

            one_day_return = (
                (
                    live_price /
                    previous_close
                ) - 1
            ) * 100

        else:

            one_day_return = np.nan

    else:

        one_day_return = np.nan

    # =====================================================
    # 1W RETURN
    #
    # Friday to Friday logic.
    #
    # If today is Friday:
    # current live price compared with
    # previous Friday close.
    #
    # With completed daily rows excluded,
    # previous Friday is normally 5 trading
    # sessions before the current Friday.
    # =====================================================

    one_week_return = np.nan

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

    # =====================================================
    # 1M RETURN
    #
    # Corresponding date of previous month.
    #
    # Example:
    # 28-Jul close -> 28-Aug live/current price
    #
    # If 28-Jul is not a trading day,
    # previous available trading day is used.
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
    # EMA
    #
    # Use historical closes plus live price.
    # =====================================================

    close_for_ema = historical[
        "Close"
    ].copy()

    live_timestamp = pd.Timestamp.now(
        tz="Asia/Kolkata"
    )

    live_series = pd.Series(
        [live_price],
        index=[live_timestamp]
    )

    close_for_ema = pd.concat(
        [
            close_for_ema,
            live_series
        ]
    )

    ema21 = float(
        close_for_ema
        .ewm(
            span=21,
            adjust=False
        )
        .mean()
        .iloc[-1]
    )

    ema50 = float(
        close_for_ema
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
        .iloc[-1]
    )

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
    # 52 WEEK HIGH / LOW
    #
    # IMPORTANT:
    # Uses actual HIGH and LOW.
    # NOT closing prices.
    #
    # No 220 trading-day restriction.
    #
    # Also includes today's intraday High/Low
    # when available.
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

    last_52w = historical.loc[
        historical_dates >= cutoff_date
    ].copy()

    if last_52w.empty:

        last_52w = historical.copy()

    week52_high = np.nan
    week52_low = np.nan

    if "High" in last_52w.columns:

        high_values = pd.to_numeric(
            last_52w["High"],
            errors="coerce"
        ).dropna()

        if not high_values.empty:

            week52_high = float(
                high_values.max()
            )

    if "Low" in last_52w.columns:

        low_values = pd.to_numeric(
            last_52w["Low"],
            errors="coerce"
        ).dropna()

        if not low_values.empty:

            week52_low = float(
                low_values.min()
            )

    # -----------------------------------------------------
    # Include today's actual intraday high/low
    # -----------------------------------------------------

    if not intraday.empty:

        if "High" in intraday.columns:

            today_highs = pd.to_numeric(
                intraday["High"],
                errors="coerce"
            ).dropna()

            if not today_highs.empty:

                intraday_high = float(
                    today_highs.max()
                )

                if (
                    not np.isfinite(week52_high)
                    or intraday_high > week52_high
                ):

                    week52_high = intraday_high

        if "Low" in intraday.columns:

            today_lows = pd.to_numeric(
                intraday["Low"],
                errors="coerce"
            ).dropna()

            if not today_lows.empty:

                intraday_low = float(
                    today_lows.min()
                )

                if (
                    not np.isfinite(week52_low)
                    or intraday_low < week52_low
                ):

                    week52_low = intraday_low

    # =====================================================
    # FROM 52 WEEK HIGH
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
    # FROM 52 WEEK LOW
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
    # FROM 21 EMA
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
    # RETURN
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
# TREND STYLING
# =========================================================

def colour_trend_column(column):

    styles = []

    for value in column:

        if value == "Bullish":

            styles.append(
                "background-color: #198754; "
                "color: white; "
                "font-weight: bold;"
            )

        elif value == "Neutral":

            styles.append(
                "background-color: #F5B642; "
                "color: black; "
                "font-weight: bold;"
            )

        elif value == "Bearish":

            styles.append(
                "background-color: #DC3545; "
                "color: white; "
                "font-weight: bold;"
            )

        else:

            styles.append("")

    return styles


# =========================================================
# MAIN SCAN
# =========================================================

if st.button("🔍 Scan Nifty 100"):

    # =====================================================
    # GET NIFTY 100
    # =====================================================

    try:

        symbols = get_nifty100_list()

    except Exception as e:

        st.error(
            "Unable to get Nifty 100 list."
        )

        st.error(str(e))

        st.stop()

    st.info(
        f"Nifty 100 stocks found: {len(symbols)}"
    )

    # =====================================================
    # DAILY DATA
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

            st.error(str(e))

            st.stop()

    # =====================================================
    # INTRADAY DATA
    # =====================================================

    with st.spinner(
        "Downloading latest intraday prices..."
    ):

        try:

            intraday_data = (
                download_intraday_data(
                    symbols
                )
            )

        except Exception:

            st.warning(
                "Intraday data unavailable. "
                "Latest daily close will be used "
                "where necessary."
            )

            intraday_data = pd.DataFrame()

    # =====================================================
    # CURRE
