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


# =========================================================
# NIFTY 100 CONSTITUENTS
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

    if len(symbols) < 95:

        raise ValueError(
            f"Only {len(symbols)} stocks found "
            "in Nifty 100 list."
        )

    return symbols


# =========================================================
# DOWNLOAD BULK STOCK DATA
#
# IMPORTANT:
# ttl=0 means NO CACHE.
# Every Scan requests fresh data.
# =========================================================

@st.cache_data(ttl=0)
def download_stock_data(symbols):

    tickers = [
        symbol + ".NS"
        for symbol in symbols
    ]

    data = yf.download(
        tickers=tickers,
        period="5y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True
    )

    return data


# =========================================================
# DOWNLOAD INDIVIDUAL STOCK
#
# NO CACHE
# =========================================================

@st.cache_data(ttl=0)
def download_single_stock(symbol):

    ticker = symbol + ".NS"

    try:

        df = yf.download(
            ticker,
            period="5y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False
        )

        if df is None or df.empty:
            return None

        # Handle MultiIndex
        if isinstance(
            df.columns,
            pd.MultiIndex
        ):

            if "Close" in df.columns.get_level_values(0):

                df = df.xs(
                    "Close",
                    axis=1,
                    level=0
                )

                if isinstance(df, pd.DataFrame):
                    df = df.iloc[:, 0]

            elif "Close" in df.columns.get_level_values(1):

                df = df.xs(
                    "Close",
                    axis=1,
                    level=1
                )

                if isinstance(df, pd.DataFrame):
                    df = df.iloc[:, 0]

        elif "Close" in df.columns:

            df = df["Close"]

        if isinstance(df, pd.DataFrame):
            df = df.iloc[:, 0]

        df = pd.to_numeric(
            df,
            errors="coerce"
        ).dropna()

        if df.empty:
            return None

        return df

    except Exception:

        return None


# =========================================================
# GET CLOSE PRICE FROM BULK DATA
# =========================================================

def get_close_from_bulk(
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

                if "Close" in df.columns:

                    close = df["Close"]

                    return pd.to_numeric(
                        close,
                        errors="coerce"
                    ).dropna()

            # OHLC -> Ticker
            if ticker in level1:

                close = data[
                    "Close",
                    ticker
                ]

                return pd.to_numeric(
                    close,
                    errors="coerce"
                ).dropna()

        elif "Close" in data.columns:

            close = data["Close"]

            return pd.to_numeric(
                close,
                errors="coerce"
            ).dropna()

    except Exception:

        pass

    return None


# =========================================================
# CALCULATE STOCK DATA
# =========================================================

def calculate_stock_data(
    symbol,
    close
):

    try:

        if close is None:
            return None

        close = pd.to_numeric(
            close,
            errors="coerce"
        ).dropna()

        # Minimum history needed
        if len(close) < 22:
            return None

        # -------------------------------------------------
        # CURRENT PRICE
        # -------------------------------------------------

        price = float(
            close.iloc[-1]
        )

        # -------------------------------------------------
        # RETURNS
        # -------------------------------------------------

        one_day_return = (
            close.iloc[-1]
            /
            close.iloc[-2]
            - 1
        ) * 100

        one_week_return = (
            close.iloc[-1]
            /
            close.iloc[-6]
            - 1
        ) * 100

        one_month_return = (
            close.iloc[-1]
            /
            close.iloc[-22]
            - 1
        ) * 100

        # -------------------------------------------------
        # EMA
        # -------------------------------------------------

        ema21 = (
            close
            .ewm(
                span=21,
                adjust=False
            )
            .mean()
            .iloc[-1]
        )

        ema50 = (
            close
            .ewm(
                span=50,
                adjust=False
            )
            .mean()
            .iloc[-1]
        )

        ema200 = (
            close
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
        # DISTANCE FROM 52 WEEK HIGH
        # -------------------------------------------------

        if week52_high != 0:

            from_52w_high = (
                price
                /
                week52_high
                - 1
            ) * 100

        else:

            from_52w_high = np.nan

        # -------------------------------------------------
        # DISTANCE FROM 52 WEEK LOW
        # -------------------------------------------------

        if week52_low != 0:

            from_52w_low = (
                price
                /
                week52_low
                - 1
            ) * 100

        else:

            from_52w_low = np.nan

        # -------------------------------------------------
        # DISTANCE FROM 21 EMA
        # -------------------------------------------------

        if ema21 != 0:

            from_21_ema = (
                price
                /
                ema21
                - 1
            ) * 100

        else:

            from_21_ema = np.nan

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
    # GET CURRENT NIFTY 100 LIST
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
    # FRESH BULK DOWNLOAD
    # =====================================================

    with st.spinner(
        "Downloading fresh Nifty 100 data..."
    ):

        try:

            bulk_data = download_stock_data(
                symbols
            )

        except Exception:

            bulk_data = None

    # =====================================================
    # CALCULATE ALL STOCKS
    # =====================================================

    results = []

    missing_stocks = []

    progress = st.progress(0)

    total = len(symbols)

    for i, symbol in enumerate(symbols):

        # First try bulk data
        close = get_close_from_bulk(
            symbol,
            bulk_data
        )

        # If missing, try individual download
        if close is None or len(close) < 22:

            close = download_single_stock(
                symbol
            )

        result = calculate_stock_data(
            symbol,
            close
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
    # DATAFRAME
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
    # LAST UPDATED TIME
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
        f"🕐 Last updated: {updated_time}"
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
            "Data could not be retrieved for: "
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
    # DISPLAY TABLE
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
        file_name="nifty100_screener.csv",
        mime="text/csv"
)
