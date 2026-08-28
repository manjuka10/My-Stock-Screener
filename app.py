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
    "Current price uses the latest available Yahoo Finance "
    "intraday price. Historical calculations use daily data."
)


# ============================================================
# NIFTY 100 CSV
# ============================================================

NIFTY100_URL = (
    "https://www.niftyindices.com/IndexConstituent/"
    "ind_nifty100list.csv"
)


# ============================================================
# GET NIFTY 100 SYMBOLS
# ============================================================

@st.cache_data(ttl=86400)
def get_nifty100_list():

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "Accept": (
            "text/csv,application/csv,text/plain,*/*"
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

    for i, name in enumerate(header):

        if name.lower() == "symbol":
            symbol_index = i
            break

    if symbol_index is None:
        raise ValueError(
            "Symbol column not found."
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


# ============================================================
# DOWNLOAD DAILY DATA
# ============================================================

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


# ============================================================
# DOWNLOAD INTRADAY DATA
# ============================================================

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


# ============================================================
# EXTRACT INDIVIDUAL STOCK DATA
# ============================================================

def get_ticker_data(data, ticker):

    if data is None:
        return pd.DataFrame()

    if not isinstance(data, pd.DataFrame):
        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    try:

        if isinstance(data.columns, pd.MultiIndex):

            level0 = data.columns.get_level_values(0)
            level1 = data.columns.get_level_values(1)

            # TICKER -> OHLC
            if ticker in level0:

                result = data[ticker].copy()

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
            c for c in wanted
            if c in result.columns
        ]

        if "Close" not in available:
            return pd.DataFrame()

        result = result[available].copy()

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


# ============================================================
# CONVERT INDEX TO IST DATE
# ============================================================

def get_date(value):

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
# 28-Aug-2026
#       ↓
# 28-Jul-2026
#
# If target date is holiday/weekend,
# use latest available trading date before/on target.
# ============================================================

def get_previous_month_close(
    history,
    current_date
):

    if history.empty:
        return np.nan

    current_date_ts = pd.Timestamp(
        current_date
    )

    target_date_ts = (
        current_date_ts
        - pd.DateOffset(months=1)
    )

    target_date = target_date_ts.date()

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


# ============================================================
# CALCULATE ONE STOCK
# ============================================================

def calculate_stock(
    symbol,
    daily_data,
    intraday_data,
    today
):

    ticker = symbol + ".NS"

    # --------------------------------------------------------
    # DAILY DATA
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # INTRADAY DATA
    # --------------------------------------------------------

    intraday = get_ticker_data(
        intraday_data,
        ticker
    )

    # --------------------------------------------------------
    # LIVE PRICE
    # --------------------------------------------------------

    live_price = np.nan

    if (
        not intraday.empty
        and "Close" in intraday.columns
    ):

        intraday_prices = (
            intraday["Close"]
            .dropna()
        )

        if not intraday_prices.empty:

            live_price = float(
                intraday_prices.iloc[-1]
            )

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    if (
        not np.isfinite(live_price)
        or live_price <= 0
    ):

        live_price = float(
            daily["Close"].iloc[-1]
        )

    # --------------------------------------------------------
    # DAILY DATES
    # --------------------------------------------------------

    daily_dates = pd.Series(
        [
            get_date(x)
            for x in daily.index
        ],
        index=daily.index
    )

    # --------------------------------------------------------
    # HISTORICAL DATA
    #
    # Today's daily row is excluded because live_price
    # is being used separately.
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
    # 1 DAY RETURN
    #
    # Previous trading day close -> current live price
    # ========================================================

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

    # ========================================================
    # 1 WEEK RETURN
    #
    # 5 trading sessions back -> current price
    #
    # This matches the Friday-to-Friday method we discussed.
    # ========================================================

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

    # ========================================================
    # 1 MONTH RETURN
    #
    # Corresponding date of previous month.
    #
    # Example:
    # 28-Jul close -> 28-Aug live price
    # ========================================================

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

    # ========================================================
    # EMA SERIES
    #
    # Add current live price.
    # ========================================================

    ema_series = historical["Close"].copy()

    live_index = pd.Timestamp.now(
        tz="Asia/Kolkata"
    )

    live_value = pd.Series(
        [live_price],
        index=[live_index]
    )

    ema_series = pd.concat(
        [
            ema_series,
            live_value
        ]
    )

    # ========================================================
    # EMA 21
    # ========================================================

    ema21 = float(
        ema_series
        .ewm(
            span=21,
            adjust=False
        )
        .mean()
        .iloc[-1]
    )

    # ========================================================
    # EMA 50
    # ========================================================

    ema50 = float(
        ema_series
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
        .iloc[-1]
    )

    # ========================================================
    # EMA 200
    # ========================================================

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
    # 52 WEEK ACTUAL HIGH / LOW
    #
    # IMPORTANT:
    #
    # High = actual daily HIGH
    # Low  = actual daily LOW
    #
    # NOT closing prices.
    #
    # NO 220 TRADING-DAY RULE.
    # ========================================================

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

    # --------------------------------------------------------
    # 52 WEEK HIGH
    # --------------------------------------------------------

    if "High" in last_52w.columns:

        high_values = pd.to_numeric(
            last_52w["High"],
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

    # --------------------------------------------------------
    # 52 WEEK LOW
    # --------------------------------------------------------

    if "Low" in last_52w.columns:

        low_values = pd.to_numeric(
            last_52w["Low"],
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

    # ========================================================
    # DISTANCE FROM 52 WEEK HIGH
    # ========================================================

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

    # ========================================================
    # DISTANCE FROM 52 WEEK LOW
    # ========================================================

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

    # ========================================================
    # DISTANCE FROM 21 EMA
    # ========================================================

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

    # ========================================================
    # TREND
    # ========================================================

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

    # ========================================================
    # RETURN
    # ========================================================

    return {
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


# ============================================================
# TREND COLOUR
# ============================================================

def trend_colour(value):

    if value == "Bullish":

        return (
            "background-color: #198754;"
            "color: white;"
            "font-weight: bold;"
            "text-align: center;"
        )

    if value == "Neutral":

        return (
            "background-color: #F5B642;"
            "color: black;"
            "font-weight: bold;"
            "text-align: center;"
        )

    if value == "Bearish":

        return (
            "background-color: #DC3545;"
            "color: white;"
            "font-weight: bold;"
            "text-align: center;"
        )

    return ""


# ============================================================
# SCAN
# ============================================================

if st.button(
    "🔍 Scan Nifty 100",
    type="primary"
):

    # ========================================================
    # GET STOCK LIST
    # ========================================================

    try:

        symbols = get_nifty100_list()

    except Exception as e:

        st.error(
            "Unable to get Nifty 100 list."
        )

        st.error(
            str(e)
        )

        st.stop()

    st.info(
        "Nifty 100 stocks found: "
        + str(len(symbols))
    )

    # ========================================================
    # DAILY DATA
    # ========================================================

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
                str(e)
            )

            st.stop()

    # ========================================================
    # INTRADAY DATA
    # ========================================================

    with st.spinner(
        "Downloading latest live prices..."
    ):

        try:

            intraday_data = (
                download_intraday_data(
                    symbols
                )
            )

        except Exception:

            intraday_data = pd.DataFrame()

            st.warning(
                "Intraday data unavailable. "
                "Latest daily close will be used "
                "only where live price is unavailable."
            )

    # ========================================================
    # CURRENT IST TIME
    # ========================================================

    ist = ZoneInfo(
        "Asia/Kolkata"
    )

    now_ist = datetime.now(
        ist
    )

    today = now_ist.date()

    # ========================================================
    # CALCULATE
    # ========================================================

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

        progress.progress(
            int(
                ((i + 1) / total) * 100
            )
        )

    progress.empty()

    # ========================================================
    # CHECK RESULTS
    # ========================================================

    if len(results) == 0:

        st.error(
            "No stock data was calculated."
        )

        st.stop()

    
