import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from io import StringIO
from datetime import datetime
import pytz

st.set_page_config(
    page_title="My Stock Screener",
    page_icon="📊",
    layout="wide"
)

st.title("📊 My Stock Screener")
st.caption("Automatic Nifty 100 Technical Screener")


@st.cache_data(ttl=21600)
def get_nifty100_stocks():

    url = "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv"

    headers = {
        "User-Agent": "Mozilla/5.0",
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

        price = float(
            close.iloc[-1]
        )

        previous_close = float(
            close.iloc[-2]
        )

        one_day_return = (
            price / previous_close - 1
        ) * 100

        one_week_return = (
            price / float(close.iloc[-6]) - 1
        ) * 100

        one_month_return = (
            price / float(close.iloc[-22]) - 1
        ) * 100

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

        distance_21ema = (
            price / current_ema21 - 1
        ) * 100

        high_52w = float(
            close.max()
        )

        low_52w = float(
            close.min()
        )

        distance_high = (
            price / high_52w - 1
        ) * 100

        distance_low = (
            price / low_52w - 1
        ) * 100

        daily_returns = (
            close.pct_change()
            .dropna()
        )

        volatility = (
            daily_returns.std()
            * (252 ** 0.5)
            * 100
        )

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
    step=1.0
)

min_distance_21 = st.sidebar.number_input(
    "Minimum Distance From 21 EMA %",
    value=-100.0,
    step=1.0
)

max_volatility = st.sidebar.number_input(
    "Maximum Volatility %",
    value=100.0,
    step=5.0
)

min_distance_high = st.sidebar.number_input(
    "Minimum Distance From 52W High %",
    value=-100.0,
    step=1.0
)

min_distance_low = st.sidebar.number_input(
    "Minimum Distance From 52W Low %",
    value=-100.0,
    step=5.0
)

trend_filter = st.sidebar.selectbox(
    "Trend",
    [
        "All",
        "Bullish",
        "Neutral",
        "Bearish"
    ]
)


if st.button(
    "🔍 Scan Nifty 100",
    type="primary"
):

    with st.spinner(
        "Getting latest Nifty 100 constituents..."
    ):

        try:
            stocks = get_nifty100_stocks()

        except Exception as e:

            st.error(
                "Unable to get the latest Nifty 100 list."
            )

            st.exception(e)

            st.stop()

    st.info(
        f"Latest Nifty 100 universe: {len(stocks)} stocks"
    )

    results = []

    progress = st.progress(0)

    status = st.empty()

    for i, stock in enumerate(stocks):

        status.text(
            f"Scanning "
            f"{stock.replace('.NS', '')} "
            f"({i + 1}/{len(stocks)})..."
        )

        result = analyze_stock(stock)

        if result is not None:
            results.append(result)

        progress.progress(
            (i + 1) / len(stocks)
        )

    progress.empty()
    status.empty()

    df = pd.DataFrame(results)

    if not df.empty:

        df = df[
            (df["1D Return %"] >= min_1d)
            &
            (df["1W Return %"] >= min_1w)
            &
            (df["1M Return %"] >= min_1m)
            &
            (df["From 21 EMA %"] >= min_distance_21)
            &
            (df["Volatility %"] <= max_volatility)
            &
            (df["From 52W High %"] >= min_distance_high)
            &
            (df["From 52W Low %"] >= min_distance_low)
        ]

        if trend_filter != "All":

            df = df[
                df["Trend"] == trend_filter
            ]

        df = df.sort_values(
            "1W Return %",
            ascending=False
        ).reset_index(drop=True)

        india_timezone = pytz.timezone(
            "Asia/Kolkata"
        )

        last_update = datetime.now(
            india_timezone
        ).strftime(
            "%d-%b-%Y %I:%M:%S %p IST"
        )

        st.success(
            f"🕒 Last Updated: {last_update}"
        )

        st.subheader(
            f"📋 Results — {len(df)} stocks"
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            height=650
        )

        csv = df.to_csv(
            index=False
        )

        st.download_button(
            label="⬇️ Download Results",
            data=csv,
            file_name="nifty100_screener.csv",
            mime="text/csv"
        )

    else:

        st.warning(
            "No stocks matched your filters."
        )

else:

    st.info(
        "Set your filters and tap "
        "'Scan Nifty 100'."
    )


st.divider()

st.caption(
    "Nifty 100 constituents are fetched automatically "
    "from NSE Indices. Market data is provided by "
    "Yahoo Finance."
)
