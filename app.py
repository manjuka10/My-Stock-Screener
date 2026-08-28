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
    "During market hours, Yahoo Finance may provide delayed data."
)


# =========================================================
# NIFTY 100 URL
# =========================================================

NIFTY100_URL = (
    "https://www.niftyindices.com/"
    "IndexConstituent/ind_nifty100list.csv"
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
        timeout=20
    )

    response.raise_for_status()

    text = response.content.decode(
        "utf-8-sig",
        errors="replace"
    )

    reader = csv.reader(
        io.StringIO(text)
    )

    rows = list(reader)

    if len(rows) < 2:
        raise ValueError(
            "Nifty 100 CSV returned no data."
        )

    header = [
        str(x).strip()
        for x in rows[0]
    ]

    symbol_index = None

    for i, col in enumerate(header):

        if col.lower() == "symbol":
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

    # Remove duplicates
    symbols = list(
        dict.fromkeys(symbols)
    )

    if len(symbols) < 90:

        raise ValueError(
            f"Only {len(symbols)} stocks found."
        )

    return symbols


# =========================================================
# DOWNLOAD DAILY HISTORICAL DATA
# =========================================================

@st.cache_data(ttl=300)
def download_historical_data(symbols):

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

@st.cache_data(ttl=60)
def download_intraday_data(symbols):

    tickers = [
        symbol + ".NS"
        for symbol in symbols
    ]

    data = yf.download(
        tickers=tickers,
        period="5d",
        interval="5m",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True
    )

    return data


# =========================================================
# GET INDIVIDUAL STOCK HISTORY
# =========================================================

def get_stock_history(
    symbol,
    historical_data
):

    ticker = symbol + ".NS"

    try:

        if isinstance(
            historical_data.columns,
            pd.MultiIndex
        ):

            if ticker not in (
                historical_data
                .columns
                .get_level_values(0)
            ):
                return None

            df = historical_data[
                ticker
            ].copy()

        else:

            df = historical_data.copy()

        if df.empty:
            return None

        if "Close" not in df.columns:
            return None

        df = df.dropna(
            subset=["Close"]
        )

        if df.empty:
            return None

        return df

    except Exception:

        return None


# =========================================================
# GET INTRADAY STOCK DATA
# =========================================================

def get_stock_intraday(
    symbol,
    intraday_data
):

    if intraday_data is None:
        return None

    ticker = symbol + ".NS"

    try:

        if isinstance(
            intraday_data.columns,
            pd.MultiIndex
        ):

            if ticker not in (
                intraday_data
                .columns
                .get_level_values(0)
            ):
                return None

            df = intraday_data[
                ticker
            ].copy()

        else:

            df = intraday_data.copy()

        if df.empty:
            return None

        if "Close" not in df.columns:
            return None

        df = df.dropna(
            subset=["Close"]
        )

        if df.empty:
            return None

        return df

    except Exception:

        return None


# =========================================================
# GET LATEST AVAILABLE PRICE
# =========================================================

def get_latest_price(
    symbol,
    intraday_data,
    historical_df
):

    # -----------------------------------------------------
    # Try intraday first
    # -----------------------------------------------------

    intraday_df = get_stock_intraday(
        symbol,
        intraday_data
    )

    if intraday_df is not None:

        try:

            intraday_close = (
                intraday_df["Close"]
                .dropna()
            )

            if not intraday_close.empty:

                return float(
                    intraday_close.iloc[-1]
                )

        except Exception:
            pass

    # -----------------------------------------------------
    # Fallback to daily data
    # -----------------------------------------------------

    try:

        close = (
            historical_df["Close"]
            .dropna()
        )

        if not close.empty:

            return float(
                close.iloc[-1]
            )

    except Exception:
        pass

    return None


# =========================================================
# GET PREVIOUS TRADING-DAY CLOSE
# =========================================================

def get_previous_close(
    historical_df
):

    try:

        close = (
            historical_df["Close"]
            .dropna()
        )

        if len(close) < 2:
            return None

        # -------------------------------------------------
        # Check whether the latest daily candle is today
        # -------------------------------------------------

        today_ist = datetime.now(
            ZoneInfo("Asia/Kolkata")
        ).date()

        last_date = close.index[-1]

        # Convert timestamp to date safely
        try:

            if hasattr(
                last_date,
                "tz_convert"
            ):

                last_date = (
                    last_date
                    .tz_convert(
                        "Asia/Kolkata"
                    )
                    .date()
                )

            else:

                last_date = (
                    pd.Timestamp(
                        last_date
                    ).date()
                )

        except Exception:

            last_date = (
                pd.Timestamp(
                    last_date
                ).date()
            )

        # -------------------------------------------------
        # If today's candle exists:
        # previous close = second-last candle
        #
        # If today's candle does not exist:
        # latest available candle = previous close
        # -------------------------------------------------

        if last_date == today_ist:

            return float(
                close.iloc[-2]
            )

        else:

            return float(
                close.iloc[-1]
            )

    except Exception:

        return None


# =========================================================
# CALCULATE STOCK DATA
# =========================================================

def calculate_stock_data(
    symbol,
    historical_data,
    intraday_data
):

    try:

        historical_df = get_stock_history(
            symbol,
            historical_data
        )

        if historical_df is None:
            return None

        close = (
            historical_df["Close"]
            .dropna()
        )

        # Need enough data for EMA 200
        if len(close) < 220:
            return None

        # -------------------------------------------------
        # LATEST / LIVE PRICE
        # -------------------------------------------------

        price = get_latest_price(
            symbol,
            intraday_data,
            historical_df
        )

        if price is None:
            return None

        # -------------------------------------------------
        # PREVIOUS TRADING DAY CLOSE
        # -------------------------------------------------

        previous_close = get_previous_close(
            historical_df
        )

        if previous_close is None:
            return None

        # -------------------------------------------------
        # 1D RETURN
        #
        # LIVE PRICE vs PREVIOUS DAY CLOSE
        # -------------------------------------------------

        one_day_return = (
            (price / previous_close) - 1
        ) * 100

        # -------------------------------------------------
        # 1 WEEK RETURN
        #
        # LIVE PRICE vs 5 TRADING SESSIONS AGO
        # -------------------------------------------------

        one_week_return = (
            (price / close.iloc[-6]) - 1
        ) * 100

        # -------------------------------------------------
        # 1 MONTH RETURN
        #
        # LIVE PRICE vs 21 TRADING SESSIONS AGO
        # -------------------------------------------------

        one_month_return = (
            (price / close.iloc[-22]) - 1
        ) * 100

        # -------------------------------------------------
        # LIVE PRICE INCLUDED IN EMA
        # -------------------------------------------------

        live_close = close.copy()

        # Only replace today's candle if today's
        # candle exists.
        today_ist = datetime.now(
            ZoneInfo("Asia/Kolkata")
        ).date()

        last_date = pd.Timestamp(
            live_close.index[-1]
        )

        try:

            if last_date.tzinfo is not None:

                last_date = (
                    last_date
                    .tz_convert(
                        "Asia/Kolkata"
                    )
                    .date()
                )

            else:

                last_date = (
                    last_date.date()
                )

        except Exception:

            last_date = (
                last_date.date()
            )

        if last_date == today_ist:

            live_close.iloc[-1] = price

        else:

            # Add current price as today's value
            new_index = (
                pd.Timestamp(
                    today_ist
                )
            )

            live_close.loc[
                new_index
            ] = price

        # -------------------------------------------------
        # EMA
        # -------------------------------------------------

        ema21 = (
            live_close
            .ewm(
                span=21,
                adjust=False
            )
            .mean()
            .iloc[-1]
        )

        ema50 = (
            live_close
            .ewm(
                span=50,
                adjust=False
            )
            .mean()
            .iloc[-1]
        )

        ema200 = (
            live_close
            .ewm(
                span=200,
                adjust=False
            )
            .mean()
            .iloc[-1]
        )

        # -------------------------------------------------
        # 52 WEEK HIGH / LOW
        # -------------------------------------------------

        last_252 = close.tail(252)

        week52_high = float(
            last_252.max()
        )

        week52_low = float(
            last_252.min()
        )

        # -------------------------------------------------
        # FROM 52 WEEK HIGH
        # -------------------------------------------------

        from_52w_high = (
            (price / week52_high) - 1
        ) * 100

        # -------------------------------------------------
        # FROM 52 WEEK LOW
        # -------------------------------------------------

        from_52w_low = (
            (price / week52_low) - 1
        ) * 100

        # -------------------------------------------------
        # FROM 21 EMA
        # -------------------------------------------------

        from_21_ema = (
            (price / ema21) - 1
        ) * 100

        # -------------------------------------------------
        # TREND
        # -------------------------------------------------

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

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        return {

            "Stock": symbol,

            "Price": price,

            "1D Return %":
                one_day_return,

            "1W Return %":
                one_week_return,

            "1M Return %":
                one_month_return,

            "21 EMA":
                ema21,

            "50 EMA":
                ema50,

            "200 EMA":
                ema200,

            "52W High":
                week52_high,

            "52W Low":
                week52_low,

            "From 52W High %":
                from_52w_high,

            "From 52W Low %":
                from_52w_low,

            "From 21 EMA %":
                from_21_ema,

            "Trend":
                trend
        }

    except Exception:

        return None


# =========================================================
# TREND COLOUR
# =========================================================

def colour_trend(value):

    if value == "Bullish":

        return (
            "background-color: #198754;"
            "color: white;"
            "font-weight: bold;"
        )

    elif value == "Neutral":

        return (
            "background-color: #F5B642;"
            "color: black;"
            "font-weight: bold;"
        )

    elif value == "Bearish":

        return (
            "background-color: #DC3545;"
            "color: white;"
            "font-weight: bold;"
        )

    return ""


# =========================================================
# SCAN BUTTON
# =========================================================

if st.button(
    "🔍 Scan Nifty 100"
):

    # =====================================================
    # GET CURRENT NIFTY 100
    # =====================================================

    try:

        symbols = get_nifty100_list()

        st.info(
            f"Current Nifty 100 list: "
            f"{len(symbols)} stocks"
        )

    except Exception as e:

        st.error(
            "Unable to get the current "
            "Nifty 100 list."
        )

        st.error(str(e))

        st.stop()

    # =====================================================
    # HISTORICAL DATA
    # =====================================================

    with st.spinner(
        "Downloading historical data..."
    ):

        try:

            historical_data = (
                download_historical_data(
                    symbols
                )
            )

        except Exception as e:

            st.error(
                "Unable to download "
                "historical data."
            )

            st.error(str(e))

            st.stop()

    # =====================================================
    # INTRADAY DATA
    # =====================================================

    with st.spinner(
        "Getting latest intraday prices..."
    ):

        try:

            intraday_data = (
                download_intraday_data(
                    symbols
                )
            )

        except Exception:

            intraday_data = None

    # =====================================================
    # CALCULATE ALL STOCKS
    # =====================================================

    results = []

    unavailable = []

    progress = st.progress(0)

    total = len(symbols)

    for i, symbol in enumerate(symbols):

        result = calculate_stock_data(
            symbol,
            historical_data,
            intraday_data
        )

        if result is not None:

            results.append(result)

        else:

            unavailable.append(symbol)

        progress.progress(
            int(
                ((i + 1) / total) * 100
            )
        )

    progress.empty()

    # =====================================================
    # CREATE DATAFRAME
    # =====================================================

    if not results:

        st.error(
            "No stock data could be calculated."
        )

        st.stop()

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
    # REMOVE ANY INDEX
    # =====================================================

    df = df.reset_index(
        drop=True
    )

    # =====================================================
    # SORT BY DISTANCE FROM 21 EMA
    # =====================================================

    df = df.sort_values(
        by="From 21 EMA %",
        ascending=False
    ).reset_index(
        drop=True
    )

    # =====================================================
    # UPDATE TIME
    # =====================================================

    ist = ZoneInfo(
        "Asia/Kolkata"
    )

    updated_time = (
        datetime.now(ist)
        .strftime(
            "%d-%m-%Y %I:%M:%S %p IST"
        )
    )

    st.success(
        f"🕐 Last updated: "
        f"{updated_time}"
    )

    # =====================================================
    # DATA UNAVAILABLE MESSAGE
    # =====================================================

    if unavailable:

        st.warning(
            "Data unavailable for: "
            + ", ".join(unavailable)
        )

    # ===========================
