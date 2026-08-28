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
    page_title="Nifty 100 Technical Screener",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Nifty 100 Technical Screener")

st.write(
    "Latest available intraday price is used for calculations. "
    "Yahoo Finance data may be delayed during market hours."
)


# =========================================================
# NIFTY 100 CONSTITUENTS
# =========================================================

NIFTY100_URL = (
    "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv"
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
# DOWNLOAD INTRADAY DATA
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
            period="1d",
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
# GET CURRENT / LIVE PRICE
# =========================================================

def get_current_price(
    symbol,
    historical_data,
    intraday_data
):

    ticker = symbol + ".NS"

    # -----------------------------------------------------
    # FIRST: Try latest 5-minute intraday price
    # -----------------------------------------------------

    try:

        if (
            intraday_data is not None
            and isinstance(
                intraday_data.columns,
                pd.MultiIndex
            )
        ):

            if ticker in (
                intraday_data
                .columns
                .get_level_values(0)
            ):

                intraday_df = (
                    intraday_data[ticker]
                    .copy()
                )

                if (
                    not intraday_df.empty
                    and "Close" in intraday_df.columns
                ):

                    prices = (
                        intraday_df["Close"]
                        .dropna()
                    )

                    if len(prices) > 0:

                        return float(
                            prices.iloc[-1]
                        )

    except Exception:
        pass


    # -----------------------------------------------------
    # SECOND: Try Yahoo fast_info
    # -----------------------------------------------------

    try:

        ticker_obj = yf.Ticker(ticker)

        fast_info = ticker_obj.fast_info

        last_price = fast_info.get(
            "last_price",
            None
        )

        if (
            last_price is not None
            and np.isfinite(float(last_price))
        ):

            return float(last_price)

    except Exception:
        pass


    # -----------------------------------------------------
    # THIRD: Use latest historical close
    # -----------------------------------------------------

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

        if (
            not df.empty
            and "Close" in df.columns
        ):

            close = (
                df["Close"]
                .dropna()
            )

            if len(close) > 0:

                return float(
                    close.iloc[-1]
                )

    except Exception:
        pass

    return None


# =========================================================
# CALCULATE STOCK DATA
# =========================================================

def calculate_stock_data(
    symbol,
    historical_data,
    intraday_data
):

    ticker = symbol + ".NS"

    try:

        # -------------------------------------------------
        # GET HISTORICAL DAILY DATA
        # -------------------------------------------------

        if isinstance(
            historical_data.columns,
            pd.MultiIndex
        ):

            if ticker not in (
                historical_data
                .columns
                .get_level_values(0)
            ):

                return {
                    "Stock": symbol,
                    "Data Status": "Unavailable"
                }

            df = historical_data[
                ticker
            ].copy()

        else:

            df = historical_data.copy()

        if df.empty:

            return {
                "Stock": symbol,
                "Data Status": "Unavailable"
            }

        if "Close" not in df.columns:

            return {
                "Stock": symbol,
                "Data Status": "Unavailable"
            }

        close = (
            df["Close"]
            .dropna()
            .astype(float)
        )

        if len(close) < 220:

            return {
                "Stock": symbol,
                "Data Status": "Insufficient data"
            }


        # -------------------------------------------------
        # CURRENT / INTRADAY PRICE
        # -------------------------------------------------

        current_price = get_current_price(
            symbol,
            historical_data,
            intraday_data
        )

        if current_price is None:

            return {
                "Stock": symbol,
                "Data Status": "Unavailable"
            }


        price = float(current_price)


        # -------------------------------------------------
        # IMPORTANT
        #
        # Replace today's historical close with
        # CURRENT PRICE for calculations.
        #
        # Therefore EMA, 52W High/Low and returns
        # respond to the current price.
        # -------------------------------------------------

        calculation_close = close.copy()

        today = pd.Timestamp.now(
            tz="Asia/Kolkata"
        ).normalize().tz_localize(None)

        last_date = (
            calculation_close.index[-1]
        )

        if hasattr(last_date, "tz"):

            last_date = (
                last_date
                .tz_localize(None)
            )

        last_date = pd.Timestamp(
            last_date
        ).normalize()


        if last_date == today:

            calculation_close.iloc[-1] = price

        else:

            calculation_close.loc[
                today
            ] = price

        calculation_close = (
            calculation_close
            .sort_index()
        )


        # -------------------------------------------------
        # 1 DAY RETURN
        #
        # Current price vs previous trading day close
        # -------------------------------------------------

        previous_close = float(
            close.iloc[-1]
        )

        one_day_return = (
            (price / previous_close) - 1
        ) * 100


        # -------------------------------------------------
        # 1 WEEK RETURN
        #
        # Current price vs 5 trading sessions ago
        # -------------------------------------------------

        if len(close) >= 6:

            one_week_return = (
                (price / float(close.iloc[-6])) - 1
            ) * 100

        else:

            one_week_return = np.nan


        # -------------------------------------------------
        # 1 MONTH RETURN
        #
        # Current price vs 21 trading sessions ago
        # -------------------------------------------------

        if len(close) >= 22:

            one_month_return = (
                (price / float(close.iloc[-22])) - 1
            ) * 100

        else:

            one_month_return = np.nan


        # -------------------------------------------------
        # EMA
        #
        # Current price is included.
        # -------------------------------------------------

        ema21 = (
            calculation_close
            .ewm(
                span=21,
                adjust=False
            )
            .mean()
            .iloc[-1]
        )

        ema50 = (
            calculation_close
            .ewm(
                span=50,
                adjust=False
            )
            .mean()
            .iloc[-1]
        )

        ema200 = (
            calculation_close
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
        # Current price is included.
        # -------------------------------------------------

        last_252 = (
            calculation_close
            .tail(252)
        )

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

            "From 52W High %": from_52w_high,

            "From 52W Low %": from_52w_low,

            "From 21 EMA %": from_21_ema,

            "Trend": trend,

            "Data Status": "OK"
        }


    except Exception as e:

        return {
            "Stock": symbol,
            "Data Status": "Unavailable"
        }


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

    except Exception as e:

        st.error(
            "Unable to get Nifty 100 list."
        )

        st.error(str(e))

        st.stop()


    st.info(
        f"Current Nifty 100 list: {len(symbols)} stocks"
    )


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
                "Historical data download failed."
            )

            st.error(str(e))

            st.stop()


    # =====================================================
    # DOWNLOAD INTRADAY DATA
    # =====================================================

    with st.spinner(
        "Downloading latest intraday prices..."
    ):

        intraday_data = (
            download_intraday_data(
                symbols
            )
        )


    # =====================================================
    # CALCULATE
    # =====================================================

    results = []

    progress = st.progress(0)

    total = len(symbols)

    for i, symbol in enumerate(symbols):

        result = calculate_stock_data(
            symbol,
            historical_data,
            intraday_data
        )

        results.append(result)

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

        "Trend",

        "Data Status"
    ]

    df = df[
        [
            col
            for col in columns
            if col in df.columns
        ]
    ]


    # =====================================================
    # SORT
    #
    # Only stocks with valid calculations first
    # =====================================================

    if "From 21 EMA %" in df.columns:

        df["_sort"] = (
            pd.to_numeric(
                df["From 21 EMA %"],
                errors="coerce"
            )
        )

        df = df.sort_values(
            by="_sort",
            ascending=False,
            na_position="last"
        )

        df = df.drop(
            columns=["_sort"]
        )


    df = df.reset_index(
        drop=True
    )


    # =====================================================
    # UPDATED TIME
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
    # SHOW DATA PROBLEMS
    # =====================================================

    unavailable = df[
        df["Data Status"] != "OK"
    ]

    if len(unavailable) > 0:

        st.warning(
            "Data unavailable for: "
            + ", ".join(
                unavailable["Stock"]
                .astype(str)
                .tolist()
            )
        )


    # =====================================================
    # FORMAT
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

        if col in display_df.columns:

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
        
