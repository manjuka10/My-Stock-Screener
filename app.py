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
        timeout=30
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

    if len(symbols) < 80:

        raise ValueError(
            f"Only {len(symbols)} "
            "Nifty 100 stocks were found."
        )

    return symbols


# =========================================================
# DOWNLOAD DAILY DATA
# =========================================================

@st.cache_data(ttl=900)
def download_daily_data(symbols):

    tickers = [
        symbol + ".NS"
        for symbol in symbols
    ]

    try:

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

    except Exception:

        return pd.DataFrame()


# =========================================================
# DOWNLOAD LIVE / INTRADAY DATA
# =========================================================

@st.cache_data(ttl=300)
def download_live_data(symbols):

    live_data = {}

    for symbol in symbols:

        ticker = symbol + ".NS"

        try:

            data = yf.download(
                tickers=ticker,
                period="5d",
                interval="15m",
                auto_adjust=False,
                progress=False,
                threads=False
            )

            if data is None or data.empty:
                continue

            if isinstance(
                data.columns,
                pd.MultiIndex
            ):

                if "Close" in data.columns:

                    close = data["Close"]

                    if isinstance(
                        close,
                        pd.DataFrame
                    ):

                        close = close.iloc[:, 0]

                else:
                    continue

            else:

                if "Close" not in data.columns:
                    continue

                close = data["Close"]

            close = close.dropna()

            if len(close) > 0:

                live_price = float(
                    close.iloc[-1]
                )

                live_data[symbol] = live_price

        except Exception:

            continue

    return live_data


# =========================================================
# GET SINGLE STOCK DAILY DATA
# FALLBACK WHEN BULK DOWNLOAD FAILS
# =========================================================

@st.cache_data(ttl=900)
def download_single_stock(symbol):

    ticker = symbol + ".NS"

    try:

        data = yf.download(
            tickers=ticker,
            period="2y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False
        )

        if data is None or data.empty:
            return None

        if isinstance(
            data.columns,
            pd.MultiIndex
        ):

            # Flatten single-ticker MultiIndex
            if "Close" in data.columns.get_level_values(0):

                close = data["Close"]

                if isinstance(
                    close,
                    pd.DataFrame
                ):

                    close = close.iloc[:, 0]

                df = pd.DataFrame({
                    "Close": close
                })

            else:

                return None

        else:

            df = data.copy()

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
# EXTRACT STOCK DATA FROM BULK DOWNLOAD
# =========================================================

def get_stock_from_bulk(
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

            level0 = (
                data.columns
                .get_level_values(0)
            )

            if ticker not in level0:
                return None

            df = data[ticker].copy()

        else:

            df = data.copy()

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
# CALCULATE STOCK INDICATORS
# =========================================================

def calculate_stock_data(
    symbol,
    daily_data,
    live_price=None
):

    # -----------------------------------------------------
    # First try bulk data
    # -----------------------------------------------------

    df = get_stock_from_bulk(
        symbol,
        daily_data
    )

    # -----------------------------------------------------
    # If bulk data failed, download individually
    # -----------------------------------------------------

    if df is None:

        df = download_single_stock(
            symbol
        )

    if df is None or df.empty:

        return None

    try:

        close = (
            pd.to_numeric(
                df["Close"],
                errors="coerce"
            )
            .dropna()
        )

        if len(close) < 220:

            return None

        # -------------------------------------------------
        # PREVIOUS TRADING DAY CLOSE
        # -------------------------------------------------

        previous_close = float(
            close.iloc[-1]
        )

        # -------------------------------------------------
        # CURRENT PRICE
        #
        # Prefer intraday/live price.
        # If unavailable, use latest daily close.
        # -------------------------------------------------

        if (
            live_price is not None
            and np.isfinite(live_price)
            and live_price > 0
        ):

            price = float(
                live_price
            )

        else:

            price = previous_close

        # -------------------------------------------------
        # 1 DAY RETURN
        #
        # Current price vs previous trading-day close
        # -------------------------------------------------

        one_day_return = (
            (price / previous_close) - 1
        ) * 100

        # -------------------------------------------------
        # 1 WEEK RETURN
        #
        # Current price vs 5 trading sessions ago
        # -------------------------------------------------

        week_base = float(
            close.iloc[-6]
        )

        one_week_return = (
            (price / week_base) - 1
        ) * 100

        # -------------------------------------------------
        # 1 MONTH RETURN
        #
        # Current price vs 21 trading sessions ago
        # -------------------------------------------------

        month_base = float(
            close.iloc[-22]
        )

        one_month_return = (
            (price / month_base) - 1
        ) * 100

        # -------------------------------------------------
        # EMA
        #
        # Historical daily closing prices are used.
        # Current price is NOT artificially inserted
        # into the EMA calculation.
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
        # RETURN
        # -------------------------------------------------

        return {

            "Stock": symbol,

            "Price": price,

            "1D Return %": one_day_return,

            "1W Return %": one_week_return,

            "1M Return %": one_month_return,

            "21 EMA": ema21,

            "50 EMA": ema50,

            "200 EMA": ema200,

            "52W High": week52_high,

            "52W Low": week52_low,

            "From 52W High %":
                from_52w_high,

            "From 52W Low %":
                from_52w_low,

            "From 21 EMA %":
                from_21_ema,

            "Trend": trend

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

    except Exception as e:

        st.error(
            "Unable to get the current Nifty 100 list."
        )

        st.error(str(e))

        st.stop()

    st.info(
        f"Current Nifty 100 list: "
        f"{len(symbols)} stocks"
    )

    # =====================================================
    # DOWNLOAD DAILY DATA
    # =====================================================

    with st.spinner(
        "Downloading daily market data..."
    ):

        daily_data = (
            download_daily_data(
                symbols
            )
        )

    # =====================================================
    # DOWNLOAD LIVE DATA
    # =====================================================

    with st.spinner(
        "Getting latest available prices..."
    ):

        live_data = (
            download_live_data(
                symbols
            )
        )

    # =====================================================
    # CALCULATE
    # =====================================================

    results = []

    unavailable = []

    progress = st.progress(0)

    total = len(symbols)

    for i, symbol in enumerate(symbols):

        live_price = (
            live_data.get(symbol)
        )

        result = calculate_stock_data(
            symbol,
            daily_data,
            live_price
        )

        # -------------------------------------------------
        # IMPORTANT:
        # Do NOT remove the stock from the list.
        # -------------------------------------------------

        if result is not None:

            results.append(
                result
            )

        else:

            unavailable.append(
                symbol
            )

            # -------------------------------------------------
            # Keep the stock in the table
            # -------------------------------------------------

            results.append({

                "Stock": symbol,

                "Price": np.nan,

                "1D Return %": np.nan,

                "1W Return %": np.nan,

                "1M Return %": np.nan,

                "21 EMA": np.nan,

                "50 EMA": np.nan,

                "200 EMA": np.nan,

                "52W High": np.nan,

                "52W Low": np.nan,

                "From 52W High %": np.nan,

                "From 52W Low %": np.nan,

                "From 21 EMA %": np.nan,

                "Trend": "Data Unavailable"

            })

        progress.progress(
            int(
                ((i + 1) / total) * 100
            )
        )

    progress.empty()

    # =====================================================
    # CREATE DATAFRAME
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
    # SORT
    #
    # Available stocks first
    # Then sort by From 21 EMA %
    # =====================================================

    df["_available"] = (
        df["Trend"] != "Data Unavailable"
    )

    df = df.sort_values(
        by=[
            "_available",
            "From 21 EMA %"
        ],
        ascending=[
            False,
            False
        ],
        na_position="last"
    )

    df = (
        df
        .drop(
            columns=["_available"]
        )
        .reset_index(drop=True)
    )

    # =====================================================
    # LAST UPDATED TIME
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
    # RESULT COUNT
    # =====================================================

    st.subheader(
        f"📋 Results — {len(df)} stocks"
    )

    # =====================================================
    # UNAVAILABLE WARNING
    # =====================================================

    if unavailable:

        st.warning(
            "Intraday/historical data unavailable for: "
            + ", ".join(unavailable)
        )

    # =====================================================
    # FORMAT DISPLAY DATA
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

        display_df[col] = (
            pd.to_numeric(
                display_df[col],
                errors="coerce"
            )
            .round(2)
        )

    # =====================================================
    # TREND COLOUR
    # =====================================================

    def style_row_trend(row):

        styles = [
            ""
            for _ in row
        ]

        trend_index = (
            display_df.columns
            .get_loc("Trend")
        )

        trend = row["Trend"]

        styles[trend_index] = (
            colour_trend(trend)
        )

        return styles

    styled_df = (
        display_df
        .style
        .apply(
            style_row_trend,
            axis=1
        )
        .format(
            {
                "Price": "{:.2f}",

                "1D Return %": "{:.2f}",

                "1W Return %": "{:.2f}",

                "1M Return %": "{:.2f}",

                "21 EMA": "{:.2f}",

                "50 EMA": "{:.2f}",

                "200 EMA": "{:.2f}",

                "52W High": "{:.2f}",

            
