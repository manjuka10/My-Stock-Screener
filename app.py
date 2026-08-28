import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from io import StringIO
from datetime import datetime
import pytz

# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="My Stock Screener",
    page_icon="📊",
    layout="wide"
)

st.title("📊 My Stock Screener")
st.caption("Automatic Nifty 100 Technical Screener")


# =========================================================
# GET CURRENT NIFTY 100 CONSTITUENTS
# =========================================================

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

    df = pd.read_csv(
        StringIO(response.text)
    )

    symbol_column = None

    for column in df.columns:
        if "symbol" in column.lower():
            symbol_column = column
            break

    if symbol_column is None:
        raise Exception(
            "Symbol column not found in Nifty 100 file."
        )

    symbols = (
        df[symbol_column]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    return [
        symbol + ".NS"
        for symbol in symbols
    ]


# =========================================================
# ANALYZE STOCK
# =========================================================

def analyze_stock(symbol):

    try:

        data = yf.download(
            symbol,
            period="1y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False
        )

        if data.empty:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = (
                data.columns.get_level_values(0)
            )

        data = data.dropna(
            subset=["Close"]
        )

        if len(data) < 200:
            return None

        close = data["Close"]

        # =================================================
        # PRICE
        # =================================================

        price = float(
            close.iloc[-1]
        )

        previous_close = float(
            close.iloc[-2]
        )


        # =================================================
        # RETURNS
        # =================================================

        one_day_return = (
            price / previous_close - 1
        ) * 100

        one_week_return = (
            price / float(close.iloc[-6]) - 1
        ) * 100

        one_month_return = (
            price / float(close.iloc[-22]) - 1
        ) * 100


        # =================================================
        # EMA
        # =================================================

        ema21 = close.ewm(
            span=21,
            adjust=False
        ).mean()

        ema50 = close.ewm(
            span=50,
            adjust=False
        ).mean()

        ema200 = close.ewm(
            span=200,
            adjust=False
        ).mean()

        current_ema21 = float(
            ema21.iloc[-1]
        )

        current_ema50 = float(
            ema50.iloc[-1]
        )

        current_ema200 = float(
            ema200.iloc[-1]
        )


        # =================================================
        # DISTANCE FROM 21 EMA
        # =================================================

        distance_21ema = (
            price / current_ema21 - 1
        ) * 100


        # =================================================
        # 52 WEEK HIGH / LOW
        # =================================================

        high_52w = float(
            close.max()
        )

        low_52w = float(
            close.min()
        )


        # =================================================
        # DISTANCE FROM 52 WEEK HIGH
        # =================================================

        distance_high = (
            price / high_52w - 1
        ) * 100


        # =================================================
        # DISTANCE FROM 52 WEEK LOW
        # =================================================

        distance_low = (
            price / low_52w - 1
        ) * 100


        # =================================================
        # VOLATILITY
        # =================================================

        daily_returns = (
            close.pct_change()
            .dropna()
        )

        volatility = (
            daily_returns.std()
            * (252 ** 0.5)
            * 100
        )


        # =================================================
        # TREND
        # =================================================

        if (
            price > current_ema21
            and current_ema21 > current_ema50
            and current_ema50 > current_ema200
        ):
            trend = "Bullish"

        elif (
            price < current_ema21
            and current_ema21 < current_ema50
            and current_ema50 < current_ema200
        ):
            trend = "Bearish"

        else:
            trend = "Neutral"


        # =================================================
        # RETURN
        # =================================================

        return {

            "Stock":
                symbol.replace(".NS", ""),

            "Price":
                round(price, 2),

            "1D Return %":
                round(one_day_return, 2),

            "1W Return %":
                round(one_week_return, 2),

            "1M Return %":
                round(one_month_return, 2),

            "21 EMA":
                round(current_ema21, 2),

            "50 EMA":
                round(current_ema50, 2),

            "200 EMA":
                round(current_ema200, 2),

            "From 21 EMA %":
                round(distance_21ema, 2),

            "Volatility %":
                round(volatility, 2),

            "52W High":
                round(high_52w, 2),

            "52W Low":
                round(low_52w, 2),

            "From 52W High %":
                round(distance_high, 2),

            "From 52W Low %":
                round(distance_low, 2),

            "Trend":
                trend
        }


    except Exception:
        return None


# =========================================================
# SIDEBAR FILTERS
# =========================================================

st.sidebar.header("🔍 Screener Filters")

min_1d = st.sidebar.number_input(
    "Minimum 1D Return %",
    value=-100.0,
    step=1.0
)

min_1w = st.sidebar.number_input(
    "Minimum 1W Return %",
    value=-100.0,
    step=1.0
)

min_1m = st.sidebar.number_input(
    "Minimum 1M Return %",
    value=-100.0,
