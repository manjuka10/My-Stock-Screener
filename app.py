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
    "Latest available Yahoo Finance intraday price is used for calculations. "
    "Yahoo Finance data may be delayed during market hours."
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
        "Accept": (
            "text/csv,application/csv,text/plain,"
            "*/*"
        ),
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

    # Remove duplicates
    symbols = list(
        dict.fromkeys(symbols)
    )

    if len(symbols) < 80:

        raise ValueError(
            f"Only {len(symbols)} stocks found."
        )

    return symbols


# =========================================================
# DOWNLOAD HISTORICAL DATA
# =========================================================

@st.cache_data(ttl=900)
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
# DOWNLOAD LATEST INTRADAY DATA
# =========================================================

@st.cache_data(ttl=300)
def download_intraday_data(symbols):

    tickers = [
        symbol + ".NS"
        for symbol in symbols
    ]

    try:

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

    except Exception:

        return None


# =========================================================
# GET STOCK DATAFRAME FROM YFINANCE
# =========================================================

def get_ticker_dataframe(
    data,
    ticker
):

    try:

        if data is None:
            return None

        if data.empty:
            return None

        # MultiIndex format
        if isinstance(
            data.columns,
            pd.MultiIndex
        ):

            level0 = (
                data.columns
                .get_level_values(0)
            )

            level1 = (
                data.columns
                .get_level_values(1)
            )

            # Standard yfinance format
            if ticker in level0:

                df = data[ticker].copy()

            elif ticker in level1:

                df = data.xs(
                    ticker,
                    axis=1,
                    level=1
                ).copy()

            else:

                return None

        else:

            df = data.copy()

        if df.empty:
            return None

        if "Close" not in df.columns:
            return None

        return df

    except Exception:

        return None


# =========================================================
# GET LATEST PRICE
# =========================================================

def get_latest_price(
    symbol,
    intraday_data,
    historical_data
):

    ticker = symbol + ".NS"

    # -----------------------------------------------------
    # First try intraday data
    # -----------------------------------------------------

    try:

        df_intraday = get_ticker_dataframe(
            intraday_data,
            ticker
        )

        if (
            df_intraday is not None
            and not df_intraday.empty
        ):

            close = (
                pd.to_numeric(
                    df_intraday["Close"],
                    errors="coerce"
                )
                .dropna()
            )

            if not close.empty:

                price = float(
                    close.iloc[-1]
                )

                if (
                    np.isfinite(price)
                    and price > 0
                ):

                    return price

    except Exception:

        pass

    # -----------------------------------------------------
    # Fallback to latest daily data
    # -----------------------------------------------------

    try:

        df_daily = get_ticker_dataframe(
            historical_data,
            ticker
        )

        if (
            df_daily is not None
            and not df_daily.empty
        ):

            close = (
                pd.to_numeric(
                    df_daily["Close"],
                    errors="coerce"
                )
                .dropna()
            )

            if not close.empty:

                price = float(
                    close.iloc[-1]
                )

                if (
                    np.isfinite(price)
                    and price > 0
                ):

                    return price

    except Exception:

        pass

    return None


# =========================================================
# CALCULATE STOCK
# =========================================================

def calculate_stock_data(
    symbol,
    historical_data,
    intraday_data
):

    ticker = symbol + ".NS"

    try:

        # -------------------------------------------------
        # DAILY DATA
        # -------------------------------------------------

        df_daily = get_ticker_dataframe(
            historical_data,
            ticker
        )

        if (
            df_daily is None
            or df_daily.empty
        ):

            return None

        close = pd.to_numeric(
            df_daily["Close"],
            errors="coerce"
        ).dropna()

        if len(close) < 220:

            return None

        # -------------------------------------------------
        # PREVIOUS TRADING DAY CLOSE
        #
        # This is used for 1D return.
        # -------------------------------------------------

        previous_close = float(
            close.iloc[-2]
        )

        # -------------------------------------------------
        # LATEST AVAILABLE PRICE
        #
        # Intraday price is preferred.
        # -------------------------------------------------

        current_price = get_latest_price(
            symbol,
            intraday_data,
            historical_data
        )

        if current_price is None:

            return None

        price = float(
            current_price
        )

        # -------------------------------------------------
        # CREATE CALCULATION SERIES
        #
        # Replace the latest daily close with
        # the latest available intraday price.
        #
        # This makes EMA calculations incorporate
        # the current/latest price.
        # -------------------------------------------------

        calc_close = close.copy()

        calc_close.iloc[-1] = price

        # -------------------------------------------------
        # 1 DAY RETURN
        #
        # Current price vs previous trading day close.
        # -------------------------------------------------

        one_day_return = (
            (price / previous_close) - 1
        ) * 100

        # -------------------------------------------------
        # 1 WEEK RETURN
        #
        # Current price vs 5 trading sessions ago.
        # -------------------------------------------------

        one_week_base = float(
            close.iloc[-6]
        )

        one_week_return = (
            (price / one_week_base) - 1
        ) * 100

        # -------------------------------------------------
        # 1 MONTH RETURN
        #
        # Current price vs 21 trading sessions ago.
        # -------------------------------------------------

        one_month_base = float(
            close.iloc[-22]
        )

        one_month_return = (
            (price / one_month_base) - 1
        ) * 100

        # -------------------------------------------------
        # EMA
        # -------------------------------------------------

        ema21 = (
            calc_close
            .ewm(
                span=21,
                adjust=False
            )
            .mean()
            .iloc[-1]
        )

        ema50 = (
            calc_close
            .ewm(
                span=50,
                adjust=False
            )
            .mean()
            .iloc[-1]
        )

        ema200 = (
            calc_close
            .ewm(
                span=200,
                adjust=False
            )
            .mean()
            .iloc[-1]
        )

        # -------------------------------------------------
        # 52 WEEK HIGH / LOW
        #
        # Use historical prices.
        # Current price is also included.
        # -------------------------------------------------

        last_252 = close.tail(252)

        week52_high = max(
            float(last_252.max()),
            price
        )

        week52_low = min(
            float(last_252.min()),
            price
        )

        # -------------------------------------------------
        # FROM 52W HIGH
        # -------------------------------------------------

        from_52w_high = (
            (price / week52_high) - 1
        ) * 100

        # -------------------------------------------------
        # FROM 52W LOW
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
            "Unable to get the current "
            "Nifty 100 list."
        )

        st.error(str(e))

        st.stop()

    # =====================================================
    # DOWNLOAD DATA
    # =====================================================

    with st.spinner(
        "Downloading market data..."
    ):

        historical_data = (
            download_historical_data(
                symbols
            )
        )

        intraday_data = (
            download_intraday_data(
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
    # CREATE DATAFRAME
    # =====================================================

    df = pd.DataFrame(
        results
    )

    # =====================================================
    # REQUIRED COLUMNS
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
    #
    # This guarantees all Nifty 100 stocks
    # remain visible.
    # =====================================================

    unavailable_rows = []

    for symbol in unavailable:

        unavailable_rows.append({

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

            "Trend": "Unavailable"
        })

    if unavailable_rows:

        df_unavailable = pd.DataFrame(
            unavailable_rows
        )

        df = pd.concat(
            [
                df,
                df_unavailable
            ],
            ignore_index=True
        )

    # =====================================================
    # FORCE COLUMN ORDER
    # =====================================================

    df = df[columns]

    # =====================================================
    # SORT
    #
    # Available stocks first.
    # Then sort by distance from 21 EMA.
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
        ]
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
    # UNAVAILABLE MESSAGE
    # =====================================================

    if unavailable:

        st.warning(
            "Intraday/historical data unavailable for: "
            + ", ".join(unavailable)
        )

    # =====================================================
    # DISPLAY COPY
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
    # STYLE TREND
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
          
