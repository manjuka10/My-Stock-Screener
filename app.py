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
    "Intraday data is used when available. "
    "Technical calculations incorporate the latest available price."
)


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

    if len(symbols) < 90:

        raise ValueError(
            f"Only {len(symbols)} Nifty 100 stocks were found."
        )

    return symbols


# =========================================================
# DOWNLOAD HISTORICAL DATA
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
# DOWNLOAD LATEST INTRADAY PRICE
# =========================================================

@st.cache_data(ttl=60)
def download_intraday_data(symbols):

    tickers = [
        symbol + ".NS"
        for symbol in symbols
    ]

    data = yf.download(
        tickers=tickers,
        period="1d",
        interval="5m",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True
    )

    return data


# =========================================================
# GET HISTORICAL STOCK DATA
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
# GET LIVE / LATEST PRICE
# =========================================================

def get_latest_price(
    symbol,
    intraday_data,
    historical_df
):

    ticker = symbol + ".NS"

    # -----------------------------------------------------
    # First try 5-minute intraday data
    # -----------------------------------------------------

    try:

        if (
            isinstance(
                intraday_data.columns,
                pd.MultiIndex
            )
            and ticker in
            intraday_data.columns
            .get_level_values(0)
        ):

            intraday_df = intraday_data[
                ticker
            ].copy()

            if (
                not intraday_df.empty
                and "Close" in intraday_df.columns
            ):

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
    # Fallback to latest historical close
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
# CALCULATE STOCK INDICATORS
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

        if len(close) < 30:
            return None

        # -------------------------------------------------
        # LIVE / LATEST PRICE
        # -------------------------------------------------

        price = get_latest_price(
            symbol,
            intraday_data,
            historical_df
        )

        if price is None:
            return None

        # -------------------------------------------------
        # PREVIOUS DAY CLOSE
        # -------------------------------------------------

        previous_close = float(
            close.iloc[-1]
        )

        # -------------------------------------------------
        # 1D RETURN
        #
        # Live price vs previous trading-day close
        # -------------------------------------------------

        one_day_return = (
            (price / previous_close) - 1
        ) * 100

        # -------------------------------------------------
        # 1 WEEK RETURN
        #
        # Current live price vs 5 sessions ago
        # -------------------------------------------------

        if len(close) >= 6:

            one_week_return = (
                (price / close.iloc[-6]) - 1
            ) * 100

        else:

            one_week_return = np.nan

        # -------------------------------------------------
        # 1 MONTH RETURN
        #
        # Current live price vs 21 sessions ago
        # -------------------------------------------------

        if len(close) >= 22:

            one_month_return = (
                (price / close.iloc[-22]) - 1
            ) * 100

        else:

            one_month_return = np.nan

        # -------------------------------------------------
        # CREATE SERIES INCLUDING CURRENT PRICE
        #
        # This makes today's live price part of EMA
        # calculation.
        # -------------------------------------------------

        live_close = close.copy()

        live_close.iloc[-1] = price

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
        # DISTANCE FROM 52 WEEK HIGH
        # -------------------------------------------------

        from_52w_high = (
            (price / week52_high) - 1
        ) * 100

        # -------------------------------------------------
        # DISTANCE FROM 52 WEEK LOW
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
    # GET NIFTY 100 LIST
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
    # DOWNLOAD HISTORICAL DATA
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
    # DOWNLOAD INTRADAY DATA
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
    # CALCULATE
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
    # RESULTS
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
    # SORT
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
    # UNAVAILABLE STOCKS
    # =====================================================

    if unavailable:

        st.warning(
            "Intraday/historical data "
            "unavailable for: "
            + ", ".join(unavailable)
        )

    # =====================================================
    # RESULT COUNT
    # =====================================================

    st.subheader(
        f"📋 Results — {len(df)} stocks"
    )

    # =====================================================
    # DISPLAY DATA
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
            ).round(2)
        )

    # =====================================================
    # STYLE
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

                "From 52W High %":
                    "{:.2f}",

                "From 52W Low %":
                    "{:.2f}",

                "From 21 EMA %":
                    "{:.2f}"
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

    csv_data = (
        df.to_csv(
            index=False
        ).encode("utf-8")
    )

    st.download_button(
        label="⬇️ Download Results CSV",
        data=csv_data,
        file_name="nifty100_screener.csv",
        mime="text/csv"
)
