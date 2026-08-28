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

NIFTY100_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv"


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

    # -----------------------------------------------------
    # IMPORTANT:
    # Use csv.reader instead of pandas.read_csv directly.
    # This avoids the tokenizing error seen earlier.
    # -----------------------------------------------------

    text = response.content.decode("utf-8-sig", errors="replace")

    reader = csv.reader(io.StringIO(text))

    rows = list(reader)

    if len(rows) < 2:
        raise ValueError("Nifty 100 CSV returned no data.")

    header = [str(x).strip() for x in rows[0]]

    # Find Symbol column
    symbol_index = None

    for i, col in enumerate(header):
        if col.lower() == "symbol":
            symbol_index = i
            break

    if symbol_index is None:
        raise ValueError("Symbol column not found in Nifty 100 CSV.")

    symbols = []

    for row in rows[1:]:

        if len(row) <= symbol_index:
            continue

        symbol = row[symbol_index].strip()

        if symbol:
            symbols.append(symbol)

    # Remove duplicates while maintaining order
    symbols = list(dict.fromkeys(symbols))

    if len(symbols) < 80:
        raise ValueError(
            f"Only {len(symbols)} Nifty 100 stocks were found."
        )

    return symbols


# =========================================================
# DOWNLOAD STOCK DATA
# =========================================================

@st.cache_data(ttl=900)
def download_stock_data(symbols):

    tickers = [symbol + ".NS" for symbol in symbols]

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
# CALCULATE STOCK INDICATORS
# =========================================================

def calculate_stock_data(symbol, data):

    ticker = symbol + ".NS"

    try:

        # -------------------------------------------------
        # Handle yfinance MultiIndex data
        # -------------------------------------------------

        if isinstance(data.columns, pd.MultiIndex):

            if ticker not in data.columns.get_level_values(0):
                return None

            df = data[ticker].copy()

        else:

            df = data.copy()

        if df.empty:
            return None

        # -------------------------------------------------
        # Make sure required column exists
        # -------------------------------------------------

        if "Close" not in df.columns:
            return None

        close = df["Close"].dropna()

        if len(close) < 220:
            return None

        # -------------------------------------------------
        # Current price
        # -------------------------------------------------

        price = float(close.iloc[-1])

        # -------------------------------------------------
        # RETURNS
        #
        # 1D = previous trading day's close
        # 1W = 5 trading sessions ago
        # 1M = 21 trading sessions ago
        # -------------------------------------------------

        one_day_return = (
            (close.iloc[-1] / close.iloc[-2]) - 1
        ) * 100

        one_week_return = (
            (close.iloc[-1] / close.iloc[-6]) - 1
        ) * 100

        one_month_return = (
            (close.iloc[-1] / close.iloc[-22]) - 1
        ) * 100

        # -------------------------------------------------
        # EMA
        # -------------------------------------------------

        ema21 = close.ewm(
            span=21,
            adjust=False
        ).mean().iloc[-1]

        ema50 = close.ewm(
            span=50,
            adjust=False
        ).mean().iloc[-1]

        ema200 = close.ewm(
            span=200,
            adjust=False
        ).mean().iloc[-1]

        # -------------------------------------------------
        # 52 WEEK HIGH / LOW
        #
        # Approximately 252 trading sessions
        # -------------------------------------------------

        last_252 = close.tail(252)

        week52_high = float(last_252.max())
        week52_low = float(last_252.min())

        # -------------------------------------------------
        # DISTANCE FROM 52 WEEK HIGH
        #
        # Example:
        # Price = 900
        # 52W High = 1000
        # Result = -10%
        # -------------------------------------------------

        from_52w_high = (
            (price / week52_high) - 1
        ) * 100

        # -------------------------------------------------
        # DISTANCE FROM 52 WEEK LOW
        #
        # Example:
        # Price = 120
        # 52W Low = 100
        # Result = +20%
        # -------------------------------------------------

        from_52w_low = (
            (price / week52_low) - 1
        ) * 100

        # -------------------------------------------------
        # DISTANCE FROM 21 EMA
        # -------------------------------------------------

        from_21_ema = (
            (price / ema21) - 1
        ) * 100

        # -------------------------------------------------
        # TREND
        #
        # Bullish:
        # Price > 21 EMA > 50 EMA > 200 EMA
        #
        # Bearish:
        # Price < 21 EMA < 50 EMA < 200 EMA
        #
        # Otherwise Neutral
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
        # RETURN RESULT
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

            "From 52W High %": from_52w_high,
            "From 52W Low %": from_52w_low,

            "From 21 EMA %": from_21_ema,

            "Trend": trend
        }

    except Exception:
        return None


# =========================================================
# TREND COLOUR FUNCTION
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
    "🔍 Scan Nifty 100",
    use_container_width=False
):

    # -----------------------------------------------------
    # Get current Nifty 100 list
    # -----------------------------------------------------

    try:

        symbols = get_nifty100_list()

        st.info(
            f"Current Nifty 100 list: {len(symbols)} stocks"
        )

    except Exception as e:

        st.error(
            "Unable to get the current Nifty 100 list."
        )

        st.error(str(e))

        st.stop()

    # -----------------------------------------------------
    # Download data
    # -----------------------------------------------------

    with st.spinner(
        "Downloading Nifty 100 market data..."
    ):

        try:

            data = download_stock_data(symbols)

        except Exception as e:

            st.error(
                "Unable to download stock data."
            )

            st.error(str(e))

            st.stop()

    # -----------------------------------------------------
    # Calculate indicators
    # -----------------------------------------------------

    results = []

    progress = st.progress(0)

    total = len(symbols)

    for i, symbol in enumerate(symbols):

        result = calculate_stock_data(
            symbol,
            data
        )

        if result is not None:
            results.append(result)

        progress.progress(
            int(((i + 1) / total) * 100)
        )

    progress.empty()

    # -----------------------------------------------------
    # Create DataFrame
    # -----------------------------------------------------

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
    # SORTING
    #
    # Sort by From 21 EMA % descending
    # =====================================================

    df = df.sort_values(
        by="From 21 EMA %",
        ascending=False
    ).reset_index(drop=True)

    # =====================================================
    # LAST UPDATED TIME
    # =====================================================

    ist = ZoneInfo("Asia/Kolkata")

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
    # FORMAT NUMBERS
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

        display_df[col] = display_df[col].round(2)

    # =====================================================
    # COLOUR TREND
    # =====================================================

    styled_df = (
        display_df.style
        .map(
            colour_trend,
            subset=["Trend"]
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
                "52W Low": "{:.2f}",

                "From 52W High %": "{:.2f}",
                "From 52W Low %": "{:.2f}",
                "From 21 EMA %": "{:.2f}"
            }
        )
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
