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

st.title("📊 My Stock Screener")
st.subheader("Nifty 100 Technical Screener")


# =========================================================
# NIFTY 100 LIST
# Cached for 24 hours
# =========================================================

NIFTY100_URL = (
    "https://www.niftyindices.com/"
    "IndexConstituent/ind_nifty100list.csv"
)


@st.cache_data(ttl=86400)
def get_nifty100_list():

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "Accept": "text/csv,application/csv,text/plain,*/*",
        "Referer": "https://www.niftyindices.com/"
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

    symbols = list(
        dict.fromkeys(symbols)
    )

    if len(symbols) < 95:
        raise ValueError(
            f"Only {len(symbols)} stocks found."
        )

    return symbols


# =========================================================
# DAILY DATA
#
# NO CACHE
# Used for historical calculations.
# =========================================================

@st.cache_data(ttl=0)
def get_daily_data(symbols):

    tickers = [
        symbol + ".NS"
        for symbol in symbols
    ]

    return yf.download(
        tickers=tickers,
        period="5y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True
    )


# =========================================================
# INTRADAY PRICE
#
# NO CACHE
# Latest available 5-minute price
# =========================================================

@st.cache_data(ttl=0)
def get_intraday_price(symbol):

    ticker = symbol + ".NS"

    try:

        data = yf.download(
            ticker,
            period="1d",
            interval="5m",
            auto_adjust=False,
            progress=False,
            threads=False
        )

        if data is None or data.empty:
            return None

        # Handle MultiIndex
        if isinstance(
            data.columns,
            pd.MultiIndex
        ):

            if "Close" in data.columns.get_level_values(0):

                close = data.xs(
                    "Close",
                    axis=1,
                    level=0
                )

                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]

            elif "Close" in data.columns.get_level_values(1):

                close = data.xs(
                    "Close",
                    axis=1,
                    level=1
                )

                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]

            else:
                return None

        else:

            if "Close" not in data.columns:
                return None

            close = data["Close"]

        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        close = pd.to_numeric(
            close,
            errors="coerce"
        ).dropna()

        if close.empty:
            return None

        return float(close.iloc[-1])

    except Exception:

        return None


# =========================================================
# EXTRACT DAILY CLOSE
# =========================================================

def get_daily_close(
    symbol,
    data
):

    ticker = symbol + ".NS"

    try:

        if data is None or data.empty:
            return None

        if isinstance(
            data.columns,
            pd.MultiIndex
        ):

            level0 = list(
                data.columns.get_level_values(0)
            )

            level1 = list(
                data.columns.get_level_values(1)
            )

            # Ticker -> OHLC
            if ticker in level0:

                df = data[ticker]

                if "Close" not in df.columns:
                    return None

                close = df["Close"]

            # OHLC -> Ticker
            elif ticker in level1:

                close = data[
                    "Close",
                    ticker
                ]

            else:

                return None

        else:

            if "Close" not in data.columns:
                return None

            close = data["Close"]

        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        close = pd.to_numeric(
            close,
            errors="coerce"
        ).dropna()

        return close

    except Exception:

        return None


# =========================================================
# CALCULATE STOCK
# =========================================================

def calculate_stock(
    symbol,
    daily_close,
    live_price
):

    try:

        if daily_close is None:
            return None

        daily_close = pd.to_numeric(
            daily_close,
            errors="coerce"
        ).dropna()

        if len(daily_close) < 220:
            return None

        # =================================================
        # CURRENT PRICE
        # =================================================

        if live_price is None:

            live_price = float(
                daily_close.iloc[-1]
            )

        else:

            live_price = float(
                live_price
            )

        # =================================================
        # PREVIOUS DAILY CLOSE
        # =================================================

        previous_close = float(
            daily_close.iloc[-1]
        )

        # =================================================
        # RETURNS
        #
        # Current live price is used.
        # =================================================

        one_day_return = (
            live_price
            /
            daily_close.iloc[-1]
            - 1
        ) * 100

        one_week_return = (
            live_price
            /
            daily_close.iloc[-6]
            - 1
        ) * 100

        one_month_return = (
            live_price
            /
            daily_close.iloc[-22]
            - 1
        ) * 100

        # =================================================
        # ADD LIVE PRICE AS TODAY'S LATEST PRICE
        #
        # This makes EMA reflect today's current price.
        # =================================================

        calculation_close = daily_close.copy()

        calculation_close.iloc[-1] = live_price

        # =================================================
        # EMA
        # =================================================

        ema21 = (
            calculation_close
            .ewm(
                span=21,
                adjust=False
            )
            .mean()
            .iloc[-1]
        )

        ema50 = (
            calculation_close
            .ewm(
                span=50,
                adjust=False
            )
            .mean()
            .iloc[-1]
        )

        ema200 = (
            calculation_close
            .ewm(
                span=200,
                adjust=False
            )
            .mean()
            .iloc[-1]
        )

        # =================================================
        # 52 WEEK HIGH / LOW
        #
        # Include current live price.
        # =================================================

        last_252 = calculation_close.tail(252)

        week52_high = float(
            max(
                last_252.max(),
                live_price
            )
        )

        week52_low = float(
            min(
                last_252.min(),
                live_price
            )
        )

        # =================================================
        # DISTANCE FROM 52 WEEK HIGH
        # =================================================

        if week52_high != 0:

            from_52w_high = (
                live_price
                /
                week52_high
                - 1
            ) * 100

        else:

            from_52w_high = np.nan

        # =================================================
        # DISTANCE FROM 52 WEEK LOW
        # =================================================

        if week52_low != 0:

            from_52w_low = (
                live_price
                /
                week52_low
                - 1
            ) * 100

        else:

            from_52w_low = np.nan

        # =================================================
        # DISTANCE FROM 21 EMA
        # =================================================

        if ema21 != 0:

            from_21_ema = (
                live_price
                /
                ema21
                - 1
            ) * 100

        else:

            from_21_ema = np.nan

        # =================================================
        # TREND
        # =================================================

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

        # =================================================
        # RESULT
        # =================================================

        return {

            "Stock": symbol,

            "Price": live_price,

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
    # GET NIFTY 100
    # =====================================================

    try:

        symbols = get_nifty100_list()

        st.info(
            f"Current Nifty 100 list: "
            f"{len(symbols)} stocks"
        )

    except Exception as e:

        st.error(
            "Unable to get current Nifty 100 list."
        )

        st.error(str(e))

        st.stop()

    # =====================================================
    # DAILY DATA
    # =====================================================

    with st.spinner(
        "Downloading daily history..."
    ):

        try:

            daily_data = get_daily_data(
                symbols
            )

        except Exception as e:

            st.error(
                "Unable to download daily data."
            )

            st.error(str(e))

            st.stop()

    # =====================================================
    # PROCESS STOCKS
    # =====================================================

    results = []

    missing_stocks = []

    progress = st.progress(0)

    total = len(symbols)

    for i, symbol in enumerate(symbols):

        # -------------------------------------------------
        # Daily close
        # -------------------------------------------------

        daily_close = get_daily_close(
            symbol,
            daily_data
        )

        # -------------------------------------------------
        # Live / latest intraday price
        # -------------------------------------------------

        live_price = get_intraday_price(
            symbol
        )

        # -------------------------------------------------
        # Calculate
        # -------------------------------------------------

        result = calculate_stock(
            symbol,
            daily_close,
            live_price
        )

        if result is not None:

            results.append(result)

        else:

            missing_stocks.append(symbol)

        progress.progress(
            int(
                ((i + 1) / total) * 100
            )
        )

    progress.empty()

    # =====================================================
    # CHECK RESULTS
    # =====================================================

    if not results:

        st.error(
            "No stock data could be calculated."
        )

        st.stop()

    df = pd.DataFrame(results)

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
    # SORT
    # =====================================================

    df = df.sort_values(
        by="From 21 EMA %",
        ascending=False
    ).reset_index(
        drop=True
    )

    # =====================================================
    # LAST UPDATE
    # =====================================================

    ist = ZoneInfo(
        "Asia/Kolkata"
    )

    updated_time = datetime.now(
        ist
    ).strftime(
        "%d-%m-%Y %I:%M:%S %p IST"
    )

    st.success(
        f"🕐 Latest data request: "
        f"{updated_time}"
    )

    # =====================================================
    # DATA NOTE
    # =====================================================

    st.caption(
        "Price uses the latest available "
        "Yahoo Finance 5-minute intraday data. "
        "Technical calculations incorporate "
        "the current price."
    )

    # =====================================================
    # RESULT COUNT
    # =====================================================

    st.subheader(
        f"📋 Results — {len(df)} stocks"
    )

    # =====================================================
    # MISSING STOCKS
    # =====================================================

    if missing_stocks:

        st.warning(
            "Data unavailable for: "
            + ", ".join(missing_stocks)
        )

    # =====================================================
    # ROUND NUMBERS
    # =====================================================

    display_df = df.copy()

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

        "From 52W High %",
        "From 52W Low %",
        "From 21 EMA %"
    ]

    for col in number_columns:

        display_df[col] = pd.to_numeric(
            display_df[col],
            errors="coerce"
        ).round(2)

    # =====================================================
    # STYLE
    # =====================================================

    styled_df = (

        display_df.style

        .map(
            colour_trend,
            subset=["Trend"]
        )

        .format({

            "Price": "{:.2f}",

            "1D Return %": "{:.2f}",
            "1W Return %": "{:.2f}",
            "1M Return %": "{:.2f}",

            "21 EMA": "{:.2f}",
            "50 EMA": "{:.2f}",
            "200 EMA": "{:.2f}",

            "52W High": "{:.2f}",
            "52W Low": "{:.2f}",

            "From 52W High %": "{:.2f}",
            "From 52W Low %": "{:.2f}",
            "From 21 EMA %": "{:.2f}"
        })
    )

    # =====================================================
    # DISPLAY
    # =====================================================

    st.dataframe(
        styled_df,
        use_container_width=True,
        height=650,
        hide_index=True
    )

    # =====================================================
    # DOWNLOAD CSV
    # =====================================================

    csv_data = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Results CSV",
        data=csv_data,
        file_name="nifty100_live_screener.csv",
        mime="text/csv"
)
