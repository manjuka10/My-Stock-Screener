import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import csv
import io
from datetime import datetime, timedelta
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
    "Latest available Yahoo Finance intraday price is used for calculations."
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

    # Remove duplicates
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
# GET INDIVIDUAL TICKER DATA
# =========================================================

def get_ticker_data(data, ticker):

    if data is None:
        return pd.DataFrame()

    if not isinstance(data, pd.DataFrame):
        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    try:

        # =================================================
        # MULTI INDEX
        # =================================================

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

            # ---------------------------------------------
            # TICKER -> OHLC
            # ---------------------------------------------

            if ticker in level0:

                result = data[
                    ticker
                ].copy()

            # ---------------------------------------------
            # OHLC -> TICKER
            # ---------------------------------------------

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

        # =================================================
        # REQUIRED COLUMNS
        # =================================================

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

        result = result[
            available
        ].copy()

        # =================================================
        # NUMERIC
        # =================================================

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
# PREVIOUS TRADING DAY CLOSE
# =========================================================

def get_previous_trading_close(
    historical,
    target_date
):

    if historical.empty:
        return np.nan

    dates = pd.Series(
        [
            get_date(x)
            for x in historical.index
        ],
        index=historical.index
    )

    valid = historical.loc[
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
# MONTH BASE PRICE
#
# Example:
#
# Current date  = 28 Aug
# Target date   = 28 Jul
#
# We use 28 Jul closing price.
#
# If 28 Jul was a holiday/weekend,
# latest available trading close before that
# date is used.
# =========================================================

def get_previous_month_close(
    historical,
    current_date
):

    if historical.empty:
        return np.nan

    current = pd.Timestamp(
        current_date
    )

    # Move exactly one month backwards.
    target = (
        current -
        pd.DateOffset(months=1)
    )

    target_date = target.date()

    return get_previous_trading_close(
        historical,
        target_date
    )


# =========================================================
# GET LIVE PRICE
# =========================================================

def get_live_price(
    daily,
    intraday
):

    live_price = np.nan

    # =====================================================
    # FIRST CHOICE: INTRADAY
    # =====================================================

    if (
        not intraday.empty
        and "Close" in intraday.columns
    ):

        intraday_close = (
            intraday["Close"]
            .dropna()
        )

        if not intraday_close.empty:

            value = float(
                intraday_close.iloc[-1]
            )

            if (
                np.isfinite(value)
                and value > 0
            ):

                live_price = value

    # =====================================================
    # FALLBACK: LATEST DAILY CLOSE
    # =====================================================

    if (
        not np.isfinite(live_price)
        or live_price <= 0
    ):

        if (
            not daily.empty
            and "Close" in daily.columns
        ):

            close_values = (
                daily["Close"]
                .dropna()
            )

            if not close_values.empty:

                live_price = float(
                    close_values.iloc[-1]
                )

    return live_price


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
    # GET DAILY DATA
    # =====================================================

    daily = get_ticker_data(
        daily_data,
        ticker
    )

    if daily.empty:
        return None

    if "Close" not in daily.columns:
        return None

    # =====================================================
    # CLEAN DAILY DATA
    # =====================================================

    daily = daily.dropna(
        subset=["Close"]
    ).copy()

    if daily.empty:
        return None

    # =====================================================
    # GET INTRADAY
    # =====================================================

    intraday = get_ticker_data(
        intraday_data,
        ticker
    )

    # =====================================================
    # LIVE PRICE
    # =====================================================

    live_price = get_live_price(
        daily,
        intraday
    )

    if (
        not np.isfinite(live_price)
        or live_price <= 0
    ):

        return None

    # =====================================================
    # DAILY DATES
    # =====================================================

    dates = pd.Series(
        [
            get_date(x)
            for x in daily.index
        ],
        index=daily.index
    )

    # =====================================================
    # COMPLETED DAILY DATA
    #
    # Today's daily candle is excluded because
    # we use live intraday price for today.
    # =====================================================

    historical = daily.loc[
        dates < today
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
    # Previous trading-day close -> live price
    # =====================================================

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

    # =====================================================
    # 1 WEEK RETURN
    #
    # Friday close -> Friday close
    #
    # For example:
    #
    # 28 Aug Friday current
    # 21 Aug Friday base
    #
    # If current day is not Friday, we use the
    # corresponding previous week's trading day.
    # =====================================================

    current_day = today.weekday()

    # Previous completed Friday
    days_since_friday = (
        current_day - 4
    ) % 7

    if days_since_friday == 0:

        previous_week_end = (
            today -
            timedelta(days=7)
        )

    else:

        previous_week_end = (
            today -
            timedelta(
                days=days_since_friday
            )
        )

    week_base = get_previous_trading_close(
        historical,
        previous_week_end
    )

    if (
        np.isfinite(week_base)
        and week_base > 0
    ):

        one_week_return = (
            (
                live_price /
                week_base
            ) - 1
        ) * 100

    else:

        one_week_return = np.nan

    # =====================================================
    # 1 MONTH RETURN
    #
    # Corresponding date of previous month.
    #
    # Example:
    #
    # 28 July close -> 28 August live price
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
    # Add live price as today's latest value.
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
    # EMA 21
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
    # EMA 50
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
    # EMA 200
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
    #
    # We use High column for 52W High
    # We use Low column for 52W Low
    #
    # NOT closing prices.
    # =====================================================

    cutoff_date = (
        today -
        timedelta(days=365)
    )

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

    # =====================================================
    # 52W HIGH
    # =====================================================

    if "High" in last_52w.columns:

        high_values = (
            pd.to_numeric(
                last_52w["High"],
                errors="coerce"
            )
            .dropna()
        )

        if not high_values.empty:

            week52_high = float(
                high_values.max()
            )

        else:

            week52_high = np.nan

    else:

        week52_high = np.nan

    # =====================================================
    # 52W LOW
    # =====================================================

    if "Low" in last_52w.columns:

        low_values = (
            pd.to_numeric(
                last_52w["Low"],
                errors="coerce"
            )
            .dropna()
        )

        if not low_values.empty:

            week52_low = float(
                low_values.min()
            )

        else:

            week52_low = np.nan

    else:

        week52_low = np.nan

    # =====================================================
    # INCLUDE TODAY'S INTRADAY HIGH/LOW
    #
    # This makes the 52W range reflect today's
    # actual intraday high/low if available.
    # =====================================================

    if (
        not intraday.empty
        and "High" in intraday.columns
    ):

        intraday_highs = (
            pd.to_numeric(
                intraday["High"],
                errors="coerce"
            )
            .dropna()
        )

        if not intraday_highs.empty:

            today_high = float(
                intraday_highs.max()
            )

            if (
                not np.isfinite(week52_high)
                or today_high > week52_high
            ):

                week52_high = today_high

    if (
        not intraday.empty
        and "Low" in intraday.columns
    ):

        intraday_lows = (
            pd.to_numeric(
                intraday["Low"],
                errors="coerce"
            )
            .dropna()
        )

        if not intraday_lows.empty:

            today_low = float(
                intraday_lows.min()
            )

            if (
                not np.isfinite(week52_low)
                or today_low < week52_low
            ):

                week52_low = today_low

    # =====================================================
    # DISTANCE FROM 52W HIGH
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
    # DISTANCE FROM 52W LOW
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
    # RETURN
    # =====================================================

    return {

        "Stock": symbol,

        "Price": live_price,

        "1D Return %": one_day_return,

        "1W Return %": one_week_return,

        "1M Return %": one_month_return,

        "21 EMA": ema21,

        "50 EMA": ema50,

        "200 EMA": ema200,

        
