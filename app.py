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
# GET NIFTY 100 SYMBOLS
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
# DOWNLOAD DAILY DATA
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
# DOWNLOAD INTRADAY DATA
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
# GET ONE STOCK FROM YAHOO DATA
# ============================================================

def get_stock_data(data, ticker):

    if data is None:
        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    try:

        if isinstance(data.columns, pd.MultiIndex):

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

        needed = [
            "Open",
            "High",
            "Low",
            "Close"
        ]

        available = [
            column
            for column in needed
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

        result = result.dropna(
            how="all"
        )

        return result

    except Exception:

        return pd.DataFrame()


# ============================================================
# CONVERT INDEX VALUE TO INDIA DATE
# ============================================================

def india_date(value):

    try:

        timestamp = pd.Timestamp(value)

        if timestamp.tzinfo is not None:

            timestamp = timestamp.tz_convert(
                "Asia/Kolkata"
            )

        return timestamp.date()

    except Exception:

        return None


# ============================================================
# PREVIOUS MONTH CORRESPONDING DATE CLOSE
#
# Example:
#
# 28-Aug current date
#       ↓
# 28-Jul closing price
#
# If 28-Jul is a holiday/weekend,
# use the latest trading day before 28-Jul.
# ============================================================

def previous_month_close(
    historical,
    current_date
):

    if historical.empty:
        return np.nan

    target = (
        pd.Timestamp(current_date)
        - pd.DateOffset(months=1)
    )

    target_date = target.date()

    dates = pd.Series(
        [
            india_date(x)
            for x in historical.index
        ],
        index=historical.index
    )

    valid = historical.loc[
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
# CALCULATE STOCK
# ============================================================

def calculate_stock(
    symbol,
    daily_all,
    intraday_all,
    today
):

    ticker = symbol + ".NS"

    # --------------------------------------------------------
    # DAILY DATA
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
    # INTRADAY DATA
    # --------------------------------------------------------

    intraday = get_stock_data(
        intraday_all,
        ticker
    )

    # --------------------------------------------------------
    # CURRENT PRICE
    #
    # IMPORTANT:
    # Latest intraday price is used.
    # Previous close is NOT used as current price.
    # --------------------------------------------------------

    current_price = np.nan

    if (
        not intraday.empty
        and "Close" in intraday.columns
    ):

        intraday_prices = (
            intraday["Close"]
            .dropna()
        )

        if not intraday_prices.empty:

            current_price = float(
                intraday_prices.iloc[-1]
            )

    # Fallback if intraday unavailable

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
    # COMPLETED DAILY CANDLES
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
    # Friday closing to Friday closing when scanning on
    # Friday, with the current/live price used as the
    # numerator when the market is open.
    #
    # 5 trading sessions back.
    # ========================================================

    if len(historical) >= 5:

        week_base = float(
            historical["Close"].iloc[-5]
        )

        if week_base > 0:

            weekly_return = (
                (
                    current_price /
                    week_base
                ) - 1
            ) * 100

        else:

            weekly_return = np.nan

    else:

        weekly_return = np.nan

    # ========================================================
    # 1 MONTH RETURN
    #
    # Corresponding date of previous month.
    #
    # Example:
    # 28-Jul close -> 28-Aug current price
    # ========================================================

    month_base = previous_month_close(
        historical,
        today
    )

    if (
        np.isfinite(month_base)
        and month_base > 0
    ):

        monthly_return = (
            (
                current_price /
                month_base
            ) - 1
        ) * 100

    else:

        monthly_return = np.nan

    # ========================================================
    # EMA DATA
    #
    # Historical closes + current live price
    # ========================================================

    ema_data = historical["Close"].copy()

    live_timestamp = pd.Timestamp.now(
        tz="Asia/Kolkata"
    )

    ema_data.loc[
        live_timestamp
    ] = current_price

    ema21 = float(
        ema_data
        .ewm(
            span=21,
            adjust=False
        )
        .mean()
        .iloc[-1]
    )

    ema50 = float(
        ema_data
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
        .iloc[-1]
    )

    ema200 = float(
        ema_data
        .ewm(
            span=200,
            adjust=False
        )
        .mean()
        .iloc[-1]
    )

    # ========================================================
    # DISTANCE FROM 21 EMA
    #
    # Current live price vs 21 EMA
    # ========================================================

    if (
        np.isfinite(ema21)
        and ema21 > 0
    ):

        distance_21_ema = (
            (
                current_price -
                ema21
            ) / ema21
        ) * 100

    else:

        distance_21_ema = np.nan

    # ========================================================
    # 52 WEEK HIGH / LOW
    #
    # Uses actual High / Low prices.
    #
    # Includes today's intraday High / Low when available.
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

    last_52_weeks = historical.loc[
        historical_dates >= cutoff
    ].copy()

    if last_52_weeks.empty:

        last_52_weeks = historical.copy()

    # --------------------------------------------------------
    # HISTORICAL 52W HIGH
    # --------------------------------------------------------

    historical_high = np.nan

    if "High" in last_52_weeks.columns:

        high_values = pd.to_numeric(
            last_52_weeks["High"],
            errors="coerce"
        ).dropna()

        if not high_values.empty:

            historical_high = float(
                high_values.max()
            )

    # --------------------------------------------------------
    # HISTORICAL 52W LOW
    # --------------------------------------------------------

    historical_low = np.nan

    if "Low" in last_52_weeks.columns:

        low_values = pd.to_numeric(
            last_52_weeks["Low"],
            errors="coerce"
        ).dropna()

        if not low_values.empty:

            historical_low = float(
                low_values.min()
            )

    # --------------------------------------------------------
    # TODAY'S INTRADAY HIGH / LOW
    # --------------------------------------------------------

    today_high = np.nan
    today_low = np.nan

    if not intraday.empty:

        if "High" in intraday.columns:

            values = pd.to_numeric(
                intraday["High"],
                errors="coerce"
            ).dropna()

            if not values.empty:

                today_high = float(
                    values.max()
                )

        if "Low" in intraday.columns:

            values = pd.to_numeric(
                intraday["Low"],
                errors="coerce"
            ).dropna()

            if not values.empty:

                today_low = float(
                    values.min()
                )

    # --------------------------------------------------------
    # FINAL 52W HIGH
    # --------------------------------------------------------

    high_candidates = []

    if np.isfinite(historical_high):

        high_candidates.append(
            historical_high
        )

    if np.isfinite(today_high):

        high_candidates.append(
            today_high
        )

    if high_candidates:

        week52_high = max(
            high_candidates
        )

    else:

        week52_high = np.nan

    # --------------------------------------------------------
    # FINAL 52W LOW
    # --------------------------------------------------------

    low_candidates = []

    if np.isfinite(historical_low):

        low_candidates.append(
            historical_low
        )

    if np.isfinite(today_low):

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

        distance_high = (
            (
                current_price /
                week52_high
            ) - 1
        ) * 100

    else:

        distance_high = np.nan

    # ========================================================
    # DISTANCE FROM 52W LOW
    # ========================================================

    if (
        np.isfinite(week52_low)
        and week52_low > 0
    ):

        distance_low = (
            (
                current_price /
                week52_low
            ) - 1
        ) * 100

    else:

        distance_low = np.nan

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
    # RESULT
    # ========================================================

    return {
        "Stock": symbol,
        "Price": current_price,
        "1W Return %": weekly_return,
        "1M Return %": monthly_return,
        "21 EMA": ema21,
        "50 EMA": ema50,
        "200 EMA": ema200,
        "From 52W High %": distance_high,
        "From 52W Low %": distance_low,
        "From 21 EMA %": distance_21_ema,
        "Trend": trend
    }


# ============================================================
# TREND COLOUR
# ============================================================

def colour_trend(value):

    if value == "Bullish":

        return (
            "background-color: #198754; "
            "color: white; "
            "font-weight: bold;"
        )

    if value == "Bearish":

        return (
            "background-color: #dc3545; "
            "color: white; "
            "font-weight: bold;"
        )

    if value == "Neutral":

        return (
            "background-color: #f5b642; "
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

    # --------------------------------------------------------
    # GET NIFTY 100
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
    # DAILY DOWNLOAD
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
    # INTRADAY DOWNLOAD
    # --------------------------------------------------------

    with st.spinner(
        "Downloading latest market prices..."
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
    # INDIA TIME
    # --------------------------------------------------------

    ist = ZoneInfo(
        "Asia/Kolkata"
    )

    now_ist = datetime.now(
        ist
    )

    today = now_ist.date()

    # --------------------------------------------------------
    # CALCULATE ALL STOCKS
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

    df = pd.DataFrame(results)

    # ========================================================
    # IMPORTANT:
    # KEEP THIS COLUMN ORDER
    # ===================
