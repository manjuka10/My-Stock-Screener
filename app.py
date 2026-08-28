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


# ============================================================
# TITLE
# ============================================================

st.title("📊 My Stock Screener")
st.subheader("Nifty 100 Technical Screener")

st.caption(
    "Current Price = latest available intraday price. "
    "Returns are calculated using closing prices."
)


# ============================================================
# NIFTY 100 URL
# ============================================================

NIFTY100_URL = (
    "https://www.niftyindices.com/IndexConstituent/"
    "ind_nifty100list.csv"
)


# ============================================================
# GET NIFTY 100 SYMBOLS
# ============================================================

@st.cache_data(ttl=86400)
def get_symbols():

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
        str(x).strip().lower()
        for x in rows[0]
    ]

    if "symbol" not in header:
        raise ValueError(
            "Symbol column not found."
        )

    symbol_index = header.index(
        "symbol"
    )

    symbols = []

    for row in rows[1:]:

        if len(row) <= symbol_index:
            continue

        symbol = str(
            row[symbol_index]
        ).strip()

        if symbol:
            symbols.append(symbol)

    symbols = list(
        dict.fromkeys(symbols)
    )

    return symbols


# ============================================================
# DOWNLOAD DAILY DATA
# ============================================================

@st.cache_data(ttl=300)
def get_daily(symbols):

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


# ============================================================
# DOWNLOAD INTRADAY DATA
# ============================================================

@st.cache_data(ttl=120)
def get_intraday(symbols):

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


# ============================================================
# GET INDIVIDUAL TICKER DATA
# ============================================================

def get_ticker_data(data, ticker):

    if data is None:
        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    try:

        # ----------------------------------------------------
        # MultiIndex data
        # ----------------------------------------------------

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

        result = result[
            available
        ].copy()

        for column in available:

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


# ============================================================
# CONVERT INDEX TO INDIA DATE
# ============================================================

def get_date(value):

    try:

        timestamp = pd.Timestamp(
            value
        )

        if timestamp.tzinfo is not None:

            timestamp = timestamp.tz_convert(
                "Asia/Kolkata"
            )

        return timestamp.date()

    except Exception:

        return None


# ============================================================
# GET PREVIOUS MONTH CORRESPONDING CLOSE
#
# Example:
#
# Current date = 28-Aug
# Target date = 28-Jul
#
# If 28-Jul is a holiday/weekend,
# use the latest trading day ON OR BEFORE 28-Jul.
# ============================================================

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

    dates = pd.Series(
        [
            get_date(index_value)
            for index_value in history.index
        ],
        index=history.index
    )

    valid = history.loc[
        dates <= target_date
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

    # ========================================================
    # DAILY DATA
    # ========================================================

    daily = get_ticker_data(
        daily_all,
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

    # ========================================================
    # INTRADAY DATA
    # ========================================================

    intraday = get_ticker_data(
        intraday_all,
        ticker
    )

    # ========================================================
    # CURRENT PRICE
    # ========================================================

    current_price = np.nan

    if (
        not intraday.empty
        and "Close" in intraday.columns
    ):

        intraday_prices = (
            pd.to_numeric(
                intraday["Close"],
                errors="coerce"
            )
            .dropna()
        )

        if not intraday_prices.empty:

            current_price = float(
                intraday_prices.iloc[-1]
            )

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    if (
        not np.isfinite(current_price)
        or current_price <= 0
    ):

        current_price = float(
            daily["Close"].iloc[-1]
        )

    # ========================================================
    # DAILY DATES
    # ========================================================

    daily_dates = pd.Series(
        [
            get_date(index_value)
            for index_value in daily.index
        ],
        index=daily.index
    )

    # ========================================================
    # COMPLETED DAILY DATA
    #
    # Today's incomplete daily candle is excluded.
    # ========================================================

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
    # 1 DAY RETURN
    #
    # Previous trading-day closing price
    # to current price.
    # ========================================================

    previous_close = float(
        historical["Close"].iloc[-1]
    )

    if previous_close > 0:

        one_day = (
            (
                current_price /
                previous_close
            ) - 1
        ) * 100

    else:

        one_day = np.nan

    # ========================================================
    # 1 WEEK RETURN
    #
    # 5 trading sessions back.
    #
    # This maintains the earlier Friday-to-Friday style
    # calculation when current price is the current session.
    # ========================================================

    if len(historical) >= 5:

        week_close = float(
            historical["Close"].iloc[-5]
        )

        if week_close > 0:

            one_week = (
                (
                    current_price /
                    week_close
                ) - 1
            ) * 100

        else:

            one_week = np.nan

    else:

        one_week = np.nan

    # ========================================================
    # 1 MONTH RETURN
    #
    # Corresponding date of previous month.
    #
    # Example:
    # 28-Jul closing price -> 28-Aug current price
    # ========================================================

    month_close = get_previous_month_close(
        historical,
        today
    )

    if (
        np.isfinite(month_close)
        and month_close > 0
    ):

        one_month = (
            (
                current_price /
                month_close
            ) - 1
        ) * 100

    else:

        one_month = np.nan

    # ========================================================
    # EMA
    # ========================================================

    ema_series = historical[
        "Close"
    ].copy()

    # Add current price to EMA calculation

    current_timestamp = pd.Timestamp(
        datetime.now(
            ZoneInfo("Asia/Kolkata")
        )
    )

    ema_series.loc[
        current_timestamp
    ] = current_price

    ema21 = float(
        ema_series
        .ewm(
            span=21,
            adjust=False
        )
        .mean()
        .iloc[-1]
    )

    ema50 = float(
        ema_series
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
        .iloc[-1]
    )

    ema200 = float(
        ema_series
        .ewm(
            span=200,
            adjust=False
        )
        .mean()
        .iloc[-1]
    )

    # ========================================================
    # 52 WEEK HIGH / LOW
    #
    # IMPORTANT:
    #
    # Uses ACTUAL daily High and Low.
    # NOT closing prices.
    #
    # Today's intraday High and Low are also included.
    # ========================================================

    cutoff_date = (
        pd.Timestamp(today)
        - pd.Timedelta(days=365)
    ).date()

    historical_dates = pd.Series(
        [
            get_date(index_value)
            for index_value in historical.index
        ],
        index=historical.index
    )

    last_52w = historical.loc[
        historical_dates >= cutoff_date
    ].copy()

    if last_52w.empty:

        last_52w = historical.copy()

    # ========================================================
    # HISTORICAL 52W HIGH
    # ========================================================

    historical_high = np.nan

    if "High" in last_52w.columns:

        high_values = pd.to_numeric(
            last_52w["High"],
            errors="coerce"
        ).dropna()

        if not high_values.empty:

            historical_high = float(
                high_values.max()
            )

    # ========================================================
    # HISTORICAL 52W LOW
    # ========================================================

    historical_low = np.nan

    if "Low" in last_52w.columns:

        low_values = pd.to_numeric(
            last_52w["Low"],
            errors="coerce"
        ).dropna()

        if not low_values.empty:

            historical_low = float(
                low_values.min()
            )

    # ========================================================
    # TODAY INTRADAY HIGH
    # ========================================================

    today_high = np.nan

    if (
        not intraday.empty
        and "High" in intraday.columns
    ):

        today_high_values = pd.to_numeric(
            intraday["High"],
            errors="coerce"
        ).dropna()

        if not today_high_values.empty:

            today_high = float(
                today_high_values.max()
            )

    # ========================================================
    # TODAY INTRADAY LOW
    # ========================================================

    today_low = np.nan

    if (
        not intraday.empty
        and "Low" in intraday.columns
    ):

        today_low_values = pd.to_numeric(
            intraday["Low"],
            errors="coerce"
        ).dropna()

        if not today_low_values.empty:

            today_low = float(
                today_low_values.min()
            )

    # ========================================================
    # FINAL 52 WEEK HIGH
    # ========================================================

    high_candidates = []

    if np.isfinite(
        historical_high
    ):

        high_candidates.append(
            historical_high
        )

    if np.isfinite(
        today_high
    ):

        high_candidates.append(
            today_high
        )

    if high_candidates:

        week52_high = max(
            high_candidates
        )

    else:

        week52_high = np.nan

    # ========================================================
    # FINAL 52 WEEK LOW
    # ========================================================

    low_candidates = []

    if np.isfinite(
        historical_low
    ):

        low_candidates.append(
            historical_low
        )

    if np.isfinite(
        today_low
    ):

        low_candidates.append(
            today_low
        )

    if low_candidates:

        week52_low = min(
            low_candidates
        )

    else:

        week52_low = np.nan

    # ========================================================
    # DISTANCE FROM 52W HIGH
    # ========================================================

    if (
        np.isfinite(week52_high)
        and week52_high > 0
    ):

        from_high = (
            (
                current_price /
                week52_high
            ) - 1
        ) * 100

    else:

        from_high = np.nan

    # ========================================================
    # DISTANCE FROM 52W LOW
    # ========================================================

    if (
        np.isfinite(week52_low)
        and week52_low > 0
    ):

        from_low = (
            (
                current_price /
                week52_low
            ) - 1
        ) * 100

    else:

        from_low = np.nan

    # ========================================================
    # DISTANCE FROM 21 EMA
    # ========================================================

    if (
        np.isfinite(ema21)
        and ema21 > 0
    ):

        from_ema21 = (
            (
                current_price /
                ema21
            ) - 1
        ) * 100

    else:

        from_ema21 = np.nan

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

        "1D Return %": one_day,
        "1W Return %": one_week,
        "1M Return %": one_month,

        "21 EMA": ema21,
        "50 EMA": ema50,
        "200 EMA": ema200,

        "52W High": week52_high,
        "52W Low": week52_low,

        "From 52W High %": from_high,
        "From 52W Low %": from_low,
        "From 21 EMA %": from_ema21,

        "Trend": trend
    }


# ============================================================
# TREND COLOUR FUNCTION
# ============================================================

def colour_trend(value):

    if value == "Bullish":

        return (
            "background-color: #198754; "
            "color: white; "
            "font-weight: bold;"
        )

    elif value == "Bearish":

        return (
            "background-color: #DC3545; "
            "color: white; "
            "font-weight: bold;"
        )

    elif value == "Neutral":

        return (
            "background-color: #F5B642; "
            "color: black; "
            "font-weight: bold;"
        )

    return ""


# ============================================================
# SCAN BUTTON
# ============================================================

if st.button(
    "🔍 Scan Nifty 100",
    type="primary"
):

    # ========================================================
    # GET SYMBOLS
    # ========================================================

    try:

        symbols = get_symbols()

    except Exception as error:

        st.error(
            "Unable to download Nifty 100 list."
        )

        st.error(
            str(error)
        )

        st.stop()

    st.info(
        "Nifty 100 stocks found: "
        + str(len(symbols))
    )

    # ========================================================
    # DOWNLOAD DAILY DATA
    # ========================================================

    with st.spinner(
        "Downloading daily data..."
    ):

        try:

            daily_data = get_daily(
                symbols
            )

        except Exception as error:

            st.error(
                "Daily data download failed."
            )

            st.error(
                str(error)
            )

            st.stop()

    # ==================================================
