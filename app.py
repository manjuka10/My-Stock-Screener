import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
from datetime import datetime
from zoneinfo import ZoneInfo

# ---------------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------------

st.set_page_config(
    page_title="My Stock Screener",
    page_icon="📊",
    layout="wide"
)

st.title("📊 My Stock Screener")
st.subheader("Nifty 100 Technical Screener")

# ---------------------------------------------------------
# GET CURRENT NIFTY 100 STOCK LIST
# ---------------------------------------------------------

@st.cache_data(ttl=86400)
def get_nifty100_stocks():

    url = "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv"

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv,application/xhtml+xml,application/xml"
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=20
    )

    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text))

    # NSE CSV normally contains Symbol column
    symbols = df["Symbol"].dropna().astype(str).str.strip().tolist()

    return symbols


# ---------------------------------------------------------
# DOWNLOAD STOCK DATA
# ---------------------------------------------------------

@st.cache_data(ttl=900)
def get_stock_data(symbol):

    ticker = symbol + ".NS"

    try:
        data = yf.download(
            ticker,
            period="1y",
            interval="1d",
            auto_adjust=False,
            progress=False
        )

        if data.empty:
            return None

        # Handle yfinance MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.dropna(subset=["Close"])

        if len(data) < 30:
            return None

        close = data["Close"]

        # Current price
        price = float(close.iloc[-1])

        # -------------------------------------------------
        # RETURNS
        # -------------------------------------------------

        # 1 Day
        if len(close) >= 2:
            return_1d = (price / float(close.iloc[-2]) - 1) * 100
        else:
            return_1d = np.nan

        # 1 Week - approximately 5 trading days
        if len(close) >= 6:
            return_1w = (price / float(close.iloc[-6]) - 1) * 100
        else:
            return_1w = np.nan

        # 1 Month - approximately 21 trading days
        if len(close) >= 22:
            return_1m = (price / float(close.iloc[-22]) - 1) * 100
        else:
            return_1m = np.nan

        # -------------------------------------------------
        # EMA
        # -------------------------------------------------

        ema21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
        ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
        ema200 = close.ewm(span=200, adjust=False).mean().iloc[-1]

        # -------------------------------------------------
        # 52 WEEK LOW
        # -------------------------------------------------

        low_52w = float(close.min())

        distance_52w_low = (
            (price / low_52w) - 1
        ) * 100

        # -------------------------------------------------
        # DISTANCE FROM 21 EMA
        # -------------------------------------------------

        distance_21ema = (
            (price / ema21) - 1
        ) * 100

        # -------------------------------------------------
        # TREND
        # -------------------------------------------------

        if price > ema21 > ema50 > ema200:
            trend = "Bullish"

        elif price < ema21 < ema50 < ema200:
            trend = "Bearish"

        else:
            trend = "Neutral"

        return {
            "Stock": symbol,
            "Price": price,
            "1D Return %": return_1d,
            "1W Return %": return_1w,
            "1M Return %": return_1m,
            "21 EMA": ema21,
            "50 EMA": ema50,
            "200 EMA": ema200,
            "From 52W Low %": distance_52w_low,
            "From 21 EMA %": distance_21ema,
            "Trend": trend
        }

    except Exception:
        return None


# ---------------------------------------------------------
# TREND COLOUR FUNCTION
# ---------------------------------------------------------

def colour_trend(value):

    if value == "Bullish":
        return (
            "background-color: #198754; "
            "color: white; "
            "font-weight: bold;"
        )

    elif value == "Bearish":
        return (
            "background-color: #dc3545; "
            "color: white; "
            "font-weight: bold;"
        )

    elif value == "Neutral":
        return (
            "background-color: #f0ad4e; "
            "color: black; "
            "font-weight: bold;"
        )

    return ""


# ---------------------------------------------------------
# SCAN BUTTON
# ---------------------------------------------------------

if st.button("🔍 Scan Nifty 100", use_container_width=False):

    with st.spinner("Getting current Nifty 100 stocks..."):

        try:
            stocks = get_nifty100_stocks()

        except Exception as e:
            st.error("Unable to get the current Nifty 100 list.")
            st.error(str(e))
            st.stop()

    results = []

    progress = st.progress(0)

    total = len(stocks)

    for i, symbol in enumerate(stocks):

        result = get_stock_data(symbol)

        if result is not None:
            results.append(result)

        progress.progress((i + 1) / total)

    progress.empty()

    if not results:
        st.error("No stock data could be downloaded.")
        st.stop()

    df = pd.DataFrame(results)

    # -----------------------------------------------------
    # SORT BY 1 WEEK RETURN
    # -----------------------------------------------------

    df = df.sort_values(
        by="1W Return %",
        ascending=False
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # ROUND NUMBERS
    # -----------------------------------------------------

    numeric_columns = [
        "Price",
        "1D Return %",
        "1W Return %",
        "1M Return %",
        "21 EMA",
        "50 EMA",
        "200 EMA",
        "From 52W Low %",
        "From 21 EMA %"
    ]

    for col in numeric_columns:
        df[col] = df[col].round(2)

    # -----------------------------------------------------
    # LAST UPDATE TIME - INDIA
    # -----------------------------------------------------

    update_time = datetime.now(
        ZoneInfo("Asia/Kolkata")
    ).strftime("%d-%m-%Y %I:%M:%S %p")

    st.success(
        f"Last updated: {update_time} IST"
    )

    st.write(
        f"📋 **Results — {len(df)} stocks**"
    )

    # -----------------------------------------------------
    # COLOUR TREND COLUMN
    # -----------------------------------------------------

    styled_df = (
        df.style
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
            "From 52W Low %": "{:.2f}",
            "From 21 EMA %": "{:.2f}"
        })
    )

    # -----------------------------------------------------
    # DISPLAY TABLE
    # -----------------------------------------------------

    st.dataframe(
        styled_df,
        use_container_width=True,
        height=650,
        hide_index=True
    )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    bullish_count = (df["Trend"] == "Bullish").sum()
    neutral_count = (df["Trend"] == "Neutral").sum()
    bearish_count = (df["Trend"] == "Bearish").sum()

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🟢 Bullish",
            bullish_count
        )

    with col2:
        st.metric(
            "🟠 Neutral",
            neutral_count
        )

    with col3:
        st.metric(
            "🔴 Bearish",
            bearish_count
        )

else:

    st.info(
        "Click **🔍 Scan Nifty 100** to scan the current Nifty 100 stocks."
    )

    st.caption(
        "The Nifty 100 constituent list is fetched automatically."
)
