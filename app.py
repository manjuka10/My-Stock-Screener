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
    "Yahoo Finance market data may be delayed during market hours."
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

    if len(symbols) < 80:

        raise ValueError(
            f"Only {len(symbols)} Nifty 100 stocks were found."
        )

    return symbols


# =========================================================
# DOWNLOAD MARKET DATA
# =========================================================

@st.cache_data(ttl=300)
def download_stock_data(symbols):

    tickers = [
        symbol + ".NS"
        for symbol in symbols
    ]

    # -----------------------------------------------------
    # 2 YEARS DAILY DATA
    # Used for:
    # EMA
    # 52 week high/low
    # Historical returns
    # -----------------------------------------------------

    daily_data = yf.download(
        tickers=tickers,
        period="2y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True
    )

    # -----------------------------------------------------
    # LATEST INTRADAY DATA
    #
    # 5 minute data gives the latest available price.
    # Yahoo data may be delayed during market hours.
    # -----------------------------------------------------

    intraday_data = yf.download(
        tickers=tickers,
        period="1d",
        interval="5m",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True
    )

    return daily_data, intraday_data


# =========================================================
# GET CLOSE SERIES FOR ONE TICKER
# =========================================================

def get_ticker_close(data, ticker):

    if data is None:
        return pd.Series(
            dtype="float64"
        )

    if data.empty:
        return pd.Series(
            dtype="float64"
        )

    try:

        # -------------------------------------------------
        # MULTIINDEX DATA
        # -------------------------------------------------

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

            # Ticker is first level
            if ticker in level0:

                temp = data[ticker].copy()

                if "Close" in temp.columns:

                    close = pd.to_numeric(
                        temp["Close"],
                        errors="coerce"
                    )

                    return close.dropna()

            # Ticker is second level
            if ticker in level1:

                temp = data.xs(
                    ticker,
                    axis=1,
                    level=1
                )

                if "Close" in temp.columns:

                    close = pd.to_numeric(
                        temp["Close"],
                        errors="coerce"
                    )

                    return close.dropna()

        # -------------------------------------------------
        # NORMAL DATA
        # -------------------------------------------------

        else:

            if "Close" in data.columns:

                close = pd.to_numeric(
                    data["Close"],
                    errors="coerce"
                )

                return close.dropna()

    except Exception:

        pass

    return pd.Series(
        dtype="float64"
    )


# =========================================================
# GET DATE FROM INDEX
# =========================================================

def get_index_date(index_value):

    try:

        timestamp = pd.Timestamp(
            index_value
        )

        if timestamp.tzinfo is not None:

            timestamp = timestamp.tz_convert(
                "Asia/Kolkata"
            )

        return timestamp.date()

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

        daily_close = get_ticker_close(
            daily_data,
            ticker
        )

        # -------------------------------------------------
        # GET INTRADAY DATA
        # -------------------------------------------------

        intraday_close = get_ticker_close(
            intraday_data,
            ticker
        )

        # Need enough historical data
        if len(daily_close) < 220:

            return (
                None,
                "insufficient historical data"
            )

        # Need intraday price
        if len(intraday_close) == 0:

            return (
                None,
                "intraday data unavailable"
            )

        # -------------------------------------------------
        # LATEST AVAILABLE PRICE
        # -------------------------------------------------

        live_price = float(
            intraday_close.iloc[-1]
        )

        if (
            not np.isfinite(live_price)
            or live_price <= 0
        ):

            return (
                None,
                "invalid intraday price"
            )

        # -------------------------------------------------
        # TODAY'S DATE
        # -------------------------------------------------

        ist = ZoneInfo(
            "Asia/Kolkata"
        )

        today = datetime.now(
            ist
        ).date()

        # -------------------------------------------------
        # DETERMINE WHETHER DAILY DATA ALREADY CONTAINS
        # TODAY'S PARTIAL/FINAL CANDLE
        # -------------------------------------------------

        last_daily_date = get_index_date(
            daily_close.index[-1]
        )

        historical_close = daily_close.copy()

        if last_daily_date == today:

            # Today's daily candle may contain the
            # current/partial price.
            #
            # Remove it so it cannot be mistaken
            # for yesterday's closing price.

            historical_close = (
                historical_close.iloc[:-1]
            )

        # -------------------------------------------------
        # CHECK AGAIN
        # -------------------------------------------------

        if len(historical_close) < 220:

            return (
                None,
                "not enough completed daily history"
            )

        # -------------------------------------------------
        # PREVIOUS COMPLETED TRADING DAY CLOSE
        # -------------------------------------------------

        previous_close = float(
            historical_close.iloc[-1]
        )

        if previous_close <= 0:

            return (
                None,
                "invalid previous closing price"
            )

        # =================================================
        # RETURNS
        # =================================================

        # -------------------------------------------------
        # 1 DAY RETURN
        #
        # Current price vs previous completed close
        # -------------------------------------------------

        one_day_return = (
            (
                live_price
                / previous_close
            ) - 1
        ) * 100

        # -------------------------------------------------
        # 1 WEEK RETURN
        #
        # 5 trading sessions before previous close
        # -------------------------------------------------

        if len(historical_close) >= 6:

            one_week_base = float(
                historical_close.iloc[-6]
            )

            one_week_return = (
                (
                    live_price
                    / one_week_base
                ) - 1
            ) * 100

        else:

            one_week_return = np.nan

        # -------------------------------------------------
        # 1 MONTH RETURN
        #
        # Approximately 21 trading sessions
        # -------------------------------------------------

        if len(historical_close) >= 22:

            one_month_base = float(
                historical_close.iloc[-22]
            )

            one_month_return = (
                (
                    live_price
                    / one_month_base
                ) - 1
            ) * 100

        else:

            one_month_return = np.nan

        # =================================================
        # BUILD CALCULATION SERIES
        # =================================================

        # We append today's latest available price.
        #
        # Therefore EMA calculations respond to the
        # current/latest price.

        calc_close = historical_close.copy()

        calc_close = pd.concat(
            [
                calc_close,
                pd.Series(
                    [live_price],
                    index=[
                        pd.Timestamp.now()
                    ]
                )
            ]
        )

        # =================================================
        # EMA
        # =================================================

        ema21 = float(
            calc_close
            .ewm(
                span=21,
                adjust=False
            )
            .mean()
            .iloc[-1]
        )

        ema50 = float(
            calc_close
            .ewm(
                span=50,
                adjust=False
            )
            .mean()
            .iloc[-1]
        )

        ema200 = float(
            calc_close
            .ewm(
                span=200,
                adjust=False
            )
            .mean()
            .iloc[-1]
        )

        # =================================================
        # 52 WEEK HIGH / LOW
        # =================================================

        last_252 = calc_close.tail(
            252
        )

        week52_high = float(
            last_252.max()
        )

        week52_low = float(
            last_252.min()
        )

        # =================================================
        # DISTANCE FROM 52 WEEK HIGH
        # =================================================

        from_52w_high = (
            (
                live_price
                / week52_high
            ) - 1
        ) * 100

        # =================================================
        # DISTANCE FROM 52 WEEK LOW
        # =================================================

        from_52w_low = (
            (
                live_price
                / week52_low
            ) - 1
        ) * 100

        # =================================================
        # DISTANCE FROM 21 EMA
        # =================================================

        from_21_ema = (
            (
                live_price
                / ema21
            ) - 1
        ) * 100

        # =================================================
        # TREND
        # =================================================

        if (
            live_price > ema21
            and ema21 > ema50
            and ema50 > ema200
        ):

            trend = "Bullish"

        elif (
            live_price < ema21
            and ema21 < ema50
            and ema50 < ema200
        ):

            trend = "Bearish"

        else:

            trend = "Neutral"

        # =================================================
        # RESULT
        # =================================================

        result = {

            "Stock": symbol,

            "Price": live_price,

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

        return result, None

    except Exception as e:

        return (
            None,
            str(e)
        )


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
    "🔍 Scan Nifty 100",
    use_container_width=False
):

    # =====================================================
    # GET CURRENT NIFTY 100 LIST
    # =====================================================

    try:

        symbols = get_nifty100_list()

        st.info(
            f"Current Nifty 100 list: "
            f"{len(symbols)} stocks"
        )

    except Exception as e:

        st.error(
            "Unable to get the current Nifty 100 list."
        )

        st.error(
            str(e)
        )

        st.stop()

    # =====================================================
    # DOWNLOAD DATA
    # =====================================================

    with st.spinner(
        "Downloading daily and latest intraday data..."
    ):

        try:

            daily_data, intraday_data = (
                download_stock_data(symbols)
            )

        except Exception as e:

            st.error(
                "Unable to download stock data."
            )

            st.error(
                str(e)
            )

            st.stop()

    # =====================================================
    # CALCULATE INDICATORS
    # =====================================================

    results = []

    unavailable = []

    progress = st.progress(0)

    total = len(symbols)

    for i, symbol in enumerate(symbols):

        result, reason = calculate_stock_data(
            symbol,
            daily_data,
            intraday_data
        )

        if result is not None:

            results.append(
                result
            )

        else:

            unavailable.append(
                f"{symbol} ({reason})"
            )

        progress.progress(
            int(
                ((i + 1) / total) * 100
            )
        )

    progress.empty()

    # =====================================================
    # NO RESULTS
    # =====================================================

    if not results:

        st.error(
            "No stock data could be calculated."
        )

        st.stop()

    # =====================================================
    # DATAFRAME
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

    df = df[
        columns
    ]

    # =====================================================
    # SORT
    # =====================================================

    df = df.sort_values(
        by="From 21 EMA %",
        ascending=False
    )

    df = df.reset_index(
        drop=True
    )

    # =====================================================
    # TIME
    # =====================================================

    ist = ZoneInfo(
        "Asia/Kolkata"
    )

    updated_time = datetime.now(
        ist
    ).strftime(
        "%d-%m-%Y %I:%M:%S %p IST"
    )

    # =====================================================
    # STATUS
    # =====================================================

    st.success(
        f"🕐 Last updated: "
        f"{updated_time}"
    )

    st.info(
        f"📊 Nifty 100 stocks: {len(symbols)} | "
        f"Successfully calculated: {len(df)}"
    )

    # =====================================================
    # UNAVAILABLE STOCKS
    # =====================================================

    if unavailable:

        st.warning(
            "Data unavailable for: "
            + ", ".join(unavailable)
        )

    # =====================================================
    # RESULT COUNT
    # =====================================================

    st.subheader(
        f"📋 Results — {len(df)} stocks"
    )

    # =====================================================
    # DISPLAY DATAFRAME
    # =====================================================

    display_df = df.copy()

    number_columns = [

        "
