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
    "https://www.niftyindices.com/IndexConstituent/"
    "ind_nifty100list.csv"
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

    if len(symbols) < 80:
        raise ValueError(
            f"Only {len(symbols)} Nifty stocks found."
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

@st.cache_data(ttl=300)
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
# GET TICKER DATAFRAME
# =========================================================

def get_ticker_dataframe(data, ticker):

    try:

        if data is None or data.empty:
            return None

        if isinstance(data.columns, pd.MultiIndex):

            level0 = list(
                data.columns.get_level_values(0)
            )

            level1 = list(
                data.columns.get_level_values(1)
            )

            # Format:
            # Ticker -> OHLCV
            if ticker in level0:

                return data[ticker].copy()

            # Format:
            # OHLCV -> Ticker
            if ticker in level1:

                return data.xs(
                    ticker,
                    axis=1,
                    level=1
                ).copy()

            return None

        return data.copy()

    except Exception:

        return None


# =========================================================
# GET LATEST INTRADAY PRICE
# =========================================================

def get_latest_intraday_price(
    symbol,
    intraday_data
):

    ticker = symbol + ".NS"

    try:

        df = get_ticker_dataframe(
            intraday_data,
            ticker
        )

        if df is None or df.empty:
            return None

        if "Close" not in df.columns:
            return None

        close = pd.to_numeric(
            df["Close"],
            errors="coerce"
        ).dropna()

        if close.empty:
            return None

        return float(close.iloc[-1])

    except Exception:

        return None


# =========================================================
# CALCULATE STOCK DATA
# =========================================================

def calculate_stock_data(
    symbol,
    daily_data,
    intraday_data
):

    ticker = symbol + ".NS"

    try:

        # -------------------------------------------------
        # GET DAILY DATA
        # -------------------------------------------------

        df = get_ticker_dataframe(
            daily_data,
            ticker
        )

        if df is None or df.empty:
            return None

        if "Close" not in df.columns:
            return None

        close = pd.to_numeric(
            df["Close"],
            errors="coerce"
        ).dropna()

        # IMPORTANT:
        # NO 220-DAY CRITERIA
        if len(close) < 2:
            return None

        # -------------------------------------------------
        # PREVIOUS TRADING DAY CLOSE
        # -------------------------------------------------

        previous_close = float(
            close.iloc[-2]
        )

        # -------------------------------------------------
        # LATEST AVAILABLE INTRADAY PRICE
        # -------------------------------------------------

        live_price = get_latest_intraday_price(
            symbol,
            intraday_data
        )

        # -------------------------------------------------
        # FALLBACK
        #
        # If Yahoo intraday data isn't available,
        # use latest daily close.
        # -------------------------------------------------

        if live_price is None:

            live_price = float(
                close.iloc[-1]
            )

        price = live_price

        # -------------------------------------------------
        # 1 DAY RETURN
        #
        # LIVE PRICE vs PREVIOUS TRADING DAY CLOSE
        # -------------------------------------------------

        one_day_return = (
            (price / previous_close) - 1
        ) * 100

        # -------------------------------------------------
        # 1 WEEK RETURN
        #
        # LIVE PRICE vs 5 TRADING DAYS AGO
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
        # LIVE PRICE vs 21 TRADING DAYS AGO
        # -------------------------------------------------

        if len(close) >= 22:

            one_month_return = (
                (price / close.iloc[-22]) - 1
            ) * 100

        else:

            one_month_return = np.nan

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
        # Use available history, maximum 252 sessions.
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

            "Price": round(price, 2),

            "1D Return %": round(
                one_day_return, 2
            ),

            "1W Return %": round(
                one_week_return, 2
            ) if not pd.isna(one_week_return) else np.nan,

            "1M Return %": round(
                one_month_return, 2
            ) if not pd.isna(one_month_return) else np.nan,

            "21 EMA": round(
                float(ema21), 2
            ),

            "50 EMA": round(
                float(ema50), 2
            ),

            "200 EMA": round(
                float(ema200), 2
            ),

            "52W High": round(
                week52_high, 2
            ),

            "52W Low": round(
                week52_low, 2
            ),

            "From 52W High %": round(
                from_52w_high, 2
            ),

            "From 52W Low %": round(
                from_52w_low, 2
            ),

            "From 21 EMA %": round(
                from_21_ema, 2
            ),

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
# SCAN
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
        "Downloading historical data..."
    ):

        try:

            daily_data = download_daily_data(
                symbols
            )

        except Exception as e:

            st.error(
                "Unable to download historical data."
            )

            st.error(str(e))

            st.stop()

    # =====================================================
    # INTRADAY DATA
    # =====================================================

    with st.spinner(
        "Getting latest available intraday prices..."
    ):

        try:

            intraday_data = download_intraday_data(
                symbols
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
            daily_data,
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
    # DATAFRAME
    # =====================================================

    df = pd.DataFrame(
        results
    )

    # =====================================================
    # COLUMNS
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

    # =====================================================
    # ADD UNAVAILABLE STOCKS
    # =====================================================

    for symbol in unavailable:

        df.loc[len(df)] = {

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

            "Trend": "Data unavailable"

        }

    df = df[columns]

    # =====================================================
    # SORT
    # =====================================================

    df["_available"] = (
        df["Price"].notna()
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

    df = df.drop(
        columns=["_available"]
    )

    df = df.reset_index(
        drop=True
    )

    # =====================================================
    # LAST UPDATED
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
        f"🕐 Last updated: {updated_time} IST"
    )

    # =====================================================
    # RESULT COUNT
    # =====================================================

    st.subheader(
        f"📋 Results — {len(df)} stocks"
    )

    # =====================================================
    # UNAVAILABLE
    # =====================================================

    if unavailable:

        st.warning(
            "Yahoo data unavailable for: "
            + ", ".join(unavailable)
        )

    # =====================================================
    # DISPLAY
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

                "From 52W High %": "{:.2f}",
                "From 52W Low %": "{:.2f}",
                "From 21 EMA %": "{:.2f}"

            },

            na_rep="—"
        )
    )

    # =====================================================
    # TABLE
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
