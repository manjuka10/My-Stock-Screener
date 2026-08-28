import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
from datetime import datetime

# ---------------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------------

st.set_page_config(
    page_title="My Stock Screener",
    page_icon="📊",
    layout="wide"
)

st.title("📊 My Stock Screener")
st.caption("Nifty 100 Technical Screener")

# ---------------------------------------------------------
# GET CURRENT NIFTY 100 CONSTITUENTS
# ---------------------------------------------------------

@st.cache_data(ttl=21600)
def get_nifty100_stocks():

    url = "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv"

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
        url,
        headers=headers,
        timeout=20
    )

    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text))

    # Find the symbol column automatically
    symbol_column = None

    for col in df.columns:
        if "symbol" in col.lower():
            symbol_column = col
            break

    if symbol_column is None:
        raise Exception("Could not find Symbol column in Nifty 100 file.")

    symbols = (
        df[symbol_column]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    # Yahoo Finance symbols
    symbols = [symbol + ".NS" for symbol in symbols]

    return symbols


# ---------------------------------------------------------
# CALCULATE STOCK DATA
# ---------------------------------------------------------

def analyze_stock(symbol):

    try:

        data = yf.download(
            symbol,
            period="1y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False
        )

        if data.empty:
            return None

        # Handle Yahoo Finance multi-level columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.dropna(subset=["Close"])

        if len(data) < 50:
            return None

        close = data["Close"]

        current_price = float(close.iloc[-1])

        previous_close = float(close.iloc[-2])

        # -------------------------------------------------
        # 1 DAY RETURN
        # -------------------------------------------------

        one_day_return = (
            (current_price / previous_close) - 1
        ) * 100

        # -------------------------------------------------
        # 21 EMA
        # -------------------------------------------------

        ema21 = close.ewm(
            span=21,
            adjust=False
        ).mean()

        current_ema21 = float(ema21.iloc[-1])

        distance_from_ema21 = (
            (current_price / current_ema21) - 1
        ) * 100

        # -------------------------------------------------
        # 52 WEEK HIGH / LOW
        # -------------------------------------------------

        week52_high = float(close.max())
        week52_low = float(close.min())

        distance_from_high = (
            (current_price / week52_high) - 1
        ) * 100

        distance_from_low = (
            (current_price / week52_low) - 1
        ) * 100

        # -------------------------------------------------
        # TREND
        # -------------------------------------------------

        ema50 = close.ewm(
            span=50,
            adjust=False
        ).mean()

        current_ema50 = float(ema50.iloc[-1])

        previous_ema21 = float(ema21.iloc[-2])
        previous_ema50 = float(ema50.iloc[-2])

        if (
            current_price > current_ema21
            and current_ema21 > current_ema50
            and current_ema21 > previous_ema21
        ):
            trend = "Bullish"

        elif (
            current_price < current_ema21
            and current_ema21 < current_ema50
            and current_ema21 < previous_ema21
        ):
            trend = "Bearish"

        else:
            trend = "Neutral"

        return {
            "Stock": symbol.replace(".NS", ""),
            "Price": round(current_price, 2),
            "1D Return %": round(one_day_return, 2),
            "High %": round(distance_from_high, 2),
            "From 52W Low %": round(distance_from_low, 2),
            "From 21 EMA %": round(distance_from_ema21, 2),
            "Trend": trend
        }

    except Exception as e:
        return None


# ---------------------------------------------------------
# SCAN BUTTON
# ---------------------------------------------------------

if st.button("🔍 Scan Nifty 100", type="primary"):

    with st.spinner("Getting latest Nifty 100 constituents..."):

        try:
            stocks = get_nifty100_stocks()

        except Exception as e:

            st.error(
                "Unable to download the latest Nifty 100 constituent list."
            )

            st.write(str(e))

            st.stop()

    st.info(
        f"Latest Nifty 100 list loaded: {len(stocks)} stocks"
    )

    results = []

    progress = st.progress(0)

    for i, stock in enumerate(stocks):

        result = analyze_stock(stock)

        if result is not None:
            results.append(result)

        progress.progress(
            (i + 1) / len(stocks)
        )

    progress.empty()

    if results:

        result_df = pd.DataFrame(results)

        # -------------------------------------------------
        # SORT BY 1 DAY RETURN
        # -------------------------------------------------

        result_df = result_df.sort_values(
            "1D Return %",
            ascending=False
        ).reset_index(drop=True)

        st.subheader(
            f"📋 Results — {len(result_df)} stocks"
        )

        # -------------------------------------------------
        # DISPLAY TABLE
        # -------------------------------------------------

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True
        )

        # -------------------------------------------------
        # DOWNLOAD
        # -------------------------------------------------

        csv = result_df.to_csv(index=False)

        st.download_button(
            label="⬇️ Download Results",
            data=csv,
            file_name="nifty100_screener.csv",
            mime="text/csv"
        )

        st.caption(
            "Data is based on the latest available market prices."
        )

    else:

        st.error(
            "No stock data could be downloaded."
        )


# ---------------------------------------------------------
# INFORMATION
# ---------------------------------------------------------

st.divider()

st.caption(
    "Nifty 100 constituents are fetched automatically from "
    "NSE Indices. Price data is provided by Yahoo Finance."
)
