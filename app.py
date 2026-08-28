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
    "Yahoo Finance data may be delayed during market hours."
)


# =========================================================
# NIFTY 100 LIST
# =========================================================

NIFTY100_URL = (
    "https://www.niftyindices.com/IndexConstituent/"
    "ind_nifty100list.csv"
)


@st.cache_data(ttl=86400)
def get_nifty100_list():

    headers = {
        "User-Agent": "Mozilla/5.0",
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

    rows = list(
        csv.reader(
            io.StringIO(text)
        )
    )

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

        if len(row) > symbol_index:

            symbol = row[symbol_index].strip()

            if symbol:
                symbols.append(symbol)

    symbols = list(
        dict.fromkeys(symbols)
    )

    if len(symbols) < 80:

        raise ValueError(
            "Nifty 100 list appears incomplete."
        )

    return symbols


# =========================================================
# DOWNLOAD DATA
# =========================================================

@st.cache_data(ttl=300)
def download_stock_data(symbols):

    tickers = [
        symbol + ".NS"
        for symbol in symbols
    ]

    daily_data = yf.download(
        tickers=tickers,
        period="2y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True
    )

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
# GET CLOSE
# =========================================================

def get_close(data, ticker):

    if data is None:
        return pd.Series(dtype=float)

    if data.empty:
        return pd.Series(dtype=float)

    try:

        if isinstance(
            data.columns,
            pd.MultiIndex
        ):

            level0 = data.columns.get_level_values(0)
            level1 = data.columns.get_level_values(1)

            if ticker in level0:

                temp = data[ticker]

                if "Close" in temp.columns:

                    close = pd.to_numeric(
                        temp["Close"],
                        errors="coerce"
                    )

                    return close.dropna()

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

        else:

            if "Close" in data.columns:

                close = pd.to_numeric(
                    data["Close"],
                    errors="coerce"
                )

                return close.dropna()

    except Exception:

        return pd.Series(dtype=float)

    return pd.Series(dtype=float)


# =========================================================
# GET DATE
# =========================================================

def get_date(index_value):

    try:

        ts = pd.Timestamp(index_value)

        if ts.tzinfo is not None:

            ts = ts.tz_convert(
                "Asia/Kolkata"
            )

        return ts.date()

    except Exception:

        return None


# =========================================================
# CALCULATE STOCK
# =========================================================

def calculate_stock(
    symbol,
    daily_data,
    intraday_data
):

    ticker = symbol + ".NS"

    try:

        daily_close = get_close(
            daily_data,
            ticker
        )

        intraday_close = get_close(
            intraday_data,
            ticker
        )

        # Need sufficient history
        if len(daily_close) < 220:

            return None, "Not enough history"

        # Need intraday price
        if len(intraday_close) == 0:

            return None, "No intraday data"

        # -------------------------------------------------
        # CURRENT / LATEST PRICE
        # -------------------------------------------------

        current_price = float(
            intraday_close.iloc[-1]
        )

        if (
            not np.isfinite(current_price)
            or current_price <= 0
        ):

            return None, "Invalid price"

        # -------------------------------------------------
        # TODAY
        # -------------------------------------------------

        ist = ZoneInfo(
            "Asia/Kolkata"
        )

        today = datetime.now(
            ist
        ).date()

        historical = daily_close.copy()

        last_date = get_date(
            historical.index[-1]
        )

        # Remove today's incomplete candle
        if last_date == today:

            historical = historical.iloc[:-1]

        if len(historical) < 220:

            return None, "Insufficient history"

        # -------------------------------------------------
        # PREVIOUS DAY CLOSE
        # -------------------------------------------------

        previous_close = float(
            historical.iloc[-1]
        )

        # -------------------------------------------------
        # 1 DAY RETURN
        # -------------------------------------------------

        return_1d = (
            current_price / previous_close - 1
        ) * 100

        # -------------------------------------------------
        # 1 WEEK RETURN
        # -------------------------------------------------

        week_base = float(
            historical.iloc[-6]
        )

        return_1w = (
            current_price / week_base - 1
        ) * 100

        # -------------------------------------------------
        # 1 MONTH RETURN
        # -------------------------------------------------

        month_base = float(
            historical.iloc[-22]
        )

        return_1m = (
            current_price / month_base - 1
        ) * 100

        # -------------------------------------------------
        # ADD CURRENT PRICE FOR EMA
        # -------------------------------------------------

        calculation_series = pd.concat(
            [
                historical,

                pd.Series(
                    [current_price],
                    index=[
                        pd.Timestamp.now()
                    ]
                )
            ]
        )

        # -------------------------------------------------
        # EMA
        # -------------------------------------------------

        ema21 = float(
            calculation_series
            .ewm(
                span=21,
                adjust=False
            )
            .mean()
            .iloc[-1]
        )

        ema50 = float(
            calculation_series
            .ewm(
                span=50,
                adjust=False
            )
            .mean()
            .iloc[-1]
        )

        ema200 = float(
            calculation_series
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

        last_252 = calculation_series.tail(
            252
        )

        high_52w = float(
            last_252.max()
        )

        low_52w = float(
            last_252.min()
        )

        # -------------------------------------------------
        # DISTANCE FROM 52 WEEK HIGH
        # -------------------------------------------------

        from_high = (
            current_price / high_52w - 1
        ) * 100

        # -------------------------------------------------
        # DISTANCE FROM 52 WEEK LOW
        # -------------------------------------------------

        from_low = (
            current_price / low_52w - 1
        ) * 100

        # -------------------------------------------------
        # DISTANCE FROM 21 EMA
        # -------------------------------------------------

        from_ema21 = (
            current_price / ema21 - 1
        ) * 100

        # -------------------------------------------------
        # TREND
        # -------------------------------------------------

        if (
            current_price > ema21
            and ema21 > ema50
            and ema50 > ema200
        ):

            trend = "Bullish"

        elif (
            current_price < ema21
            and ema21 < ema50
            and ema50 < ema200
        ):

            trend = "Bearish"

        else:

            trend = "Neutral"

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        result = {

            "Stock": symbol,

            "Price": current_price,

            "1D Return %": return_1d,

            "1W Return %": return_1w,

            "1M Return %": return_1m,

            "21 EMA": ema21,

            "50 EMA": ema50,

            "200 EMA": ema200,

            "52W High": high_52w,

            "52W Low": low_52w,

            "From 52W High %": from_high,

            "From 52W Low %": from_low,

            "From 21 EMA %": from_ema21,

            "Trend": trend
        }

        return result, None

    except Exception as e:

        return None, str(e)


# =========================================================
# TREND COLOUR
# =========================================================

def colour_trend(value):

    if value == "Bullish":

        return (
            "background-color: #198754; "
            "color: white; "
            "font-weight: bold;"
        )

    elif value == "Neutral":

        return (
            "background-color: #F5B642; "
            "color: black; "
            "font-weight: bold;"
        )

    elif value == "Bearish":

        return (
            "background-color: #DC3545; "
            "color: white; "
            "font-weight: bold;"
        )

    return ""


# =========================================================
# SCAN
# =========================================================

if st.button(
    "🔍 Scan Nifty 100"
):

    # -----------------------------------------------------
    # GET NIFTY 100
    # -----------------------------------------------------

    try:

        symbols = get_nifty100_list()

        st.info(
            "Current Nifty 100 list: "
            + str(len(symbols))
            + " stocks"
        )

    except Exception as e:

        st.error(
            "Unable to get Nifty 100 list."
        )

        st.error(
            str(e)
        )

        st.stop()

    # -----------------------------------------------------
    # DOWNLOAD
    # -----------------------------------------------------

    with st.spinner(
        "Downloading market data..."
    ):

        try:

            daily_data, intraday_data = (
                download_stock_data(
                    symbols
                )
            )

        except Exception as e:

            st.error(
                "Unable to download data."
            )

            st.error(
                str(e)
            )

            st.stop()

    # -----------------------------------------------------
    # CALCULATE
    # -----------------------------------------------------

    results = []

    unavailable = []

    progress = st.progress(0)

    total = len(symbols)

    for i, symbol in enumerate(symbols):

        result, reason = calculate_stock(
            symbol,
            daily_data,
            intraday_data
        )

        if result is not None:

            results.append(result)

        else:

            unavailable.append(
                symbol + " - " + reason
            )

        progress.progress(
            int(
                ((i + 1) / total) * 100
            )
        )

    progress.empty()

    # -----------------------------------------------------
    # CHECK RESULTS
    # -----------------------------------------------------

    if len(results) == 0:

        st.error(
            "No stock data could be calculated."
        )

        st.stop()

    # -----------------------------------------------------
    # DATAFRAME
    # -----------------------------------------------------

    df = pd.DataFrame(
        results
    )

    # -----------------------------------------------------
    # COLUMN ORDER
    # -----------------------------------------------------

    df = df[
        [
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
    ]

    # -----------------------------------------------------
    # SORT
    # -----------------------------------------------------

    df = df.sort_values(
        "From 21 EMA %",
        ascending=False
    )

    df = df.reset_index(
        drop=True
    )

    # -----------------------------------------------------
    # UPDATE TIME
    # -----------------------------------------------------

    ist = ZoneInfo(
        "Asia/Kolkata"
    )

    update_time = datetime.now(
        ist
    ).strftime(
        "%d-%m-%Y %I:%M:%S %p IST"
    )

    st.success(
        "🕐 Last updated: "
        + update_time
    )

    # -----------------------------------------------------
    # COUNTS
    # -----------------------------------------------------

    st.info(
        "Nifty 100: "
        + str(len(symbols))
        + " | Calculated: "
        + str(len(df))
    )

    # -----------------------------------------------------
    # UNAVAILABLE
    # -----------------------------------------------------

    if len(unavailable) > 0:

        with st.expander(
            "Stocks with unavailable Yahoo data"
        ):

            for item in unavailable:

                st.write(
                    item
                )

    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    st.subheader(
        "📋 Results - "
        + str(len(df))
        + " stocks"
    )

    # -----------------------------------------------------
    # DISPLAY COPY
    # -----------------------------------------------------

    display_df = df.copy()

    numeric_columns = [

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

    for column in numeric_columns:

        display_df[column] = pd.to_numeric(
            display_df[column],
            errors="coerce"
        ).round(2)

    # -----------------------------------------------------
    # STYLE
    # -----------------------------------------------------

    styled_df = display_df.style.map(
        colour_trend,
        subset=["Trend"]
    )

    # -----------------------------------------------------
    # TABLE
    # -----------------------------------------------------

    st.dataframe(
        styled_df,
        use_container_width=True,
        height=650,
        hide_index=True
    )

    # -----------------------------------------------------
    # CSV
    # -----------------------------------------------------

    csv_data = df.to_csv(
        index=False
    ).encode(
        "utf-8"
    )

    st.download_button(
        "⬇️ Download Results CSV",
        csv_data,
        "nifty100_screener.csv",
        "text/csv"
)
