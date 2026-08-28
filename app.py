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
    page_title="Nifty 100 Stock Screener",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("📊 Nifty 100 Stock Screener")

st.caption(
    "Live price + 1D / 1W / 1M returns + 52W High/Low"
)


# =========================================================
# NIFTY 100 URL
# =========================================================

NIFTY_URL = (
    "https://www.niftyindices.com/IndexConstituent/"
    "ind_nifty100list.csv"
)


# =========================================================
# GET NIFTY 100 SYMBOLS
# =========================================================

@st.cache_data(ttl=86400)
def get_symbols():

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.niftyindices.com/"
    }

    response = requests.get(
        NIFTY_URL,
        headers=headers,
        timeout=30
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
            "Nifty 100 list is empty."
        )

    header = [
        str(x).strip()
        for x in rows[0]
    ]

    symbol_col = None

    for i, name in enumerate(header):

        if name.lower() == "symbol":

            symbol_col = i
            break

    if symbol_col is None:

        raise ValueError(
            "Symbol column not found."
        )

    symbols = []

    for row in rows[1:]:

        if len(row) <= symbol_col:

            continue

        symbol = row[symbol_col].strip()

        if symbol:

            symbols.append(symbol)

    return list(
        dict.fromkeys(symbols)
    )


# =========================================================
# DOWNLOAD DAILY DATA
# =========================================================

@st.cache_data(ttl=300)
def get_daily_data(symbols):

    tickers = [
        symbol + ".NS"
        for symbol in symbols
    ]

    return yf.download(
        tickers=tickers,
        period="2y",
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False
    )


# =========================================================
# DOWNLOAD INTRADAY DATA
# =========================================================

@st.cache_data(ttl=120)
def get_intraday_data(symbols):

    tickers = [
        symbol + ".NS"
        for symbol in symbols
    ]

    return yf.download(
        tickers=tickers,
        period="1d",
        interval="5m",
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False
    )


# =========================================================
# EXTRACT ONE TICKER
# =========================================================

def get_one_ticker(
    data,
    ticker
):

    if data is None:

        return pd.DataFrame()

    if data.empty:

        return pd.DataFrame()

    try:

        if isinstance(
            data.columns,
            pd.MultiIndex
        ):

            level0 = list(
                data.columns.get_level_values(0)
            )

            level1 = list(
                data.columns.get_level_values(1)
            )

            if ticker in level0:

                result = data[
                    ticker
                ].copy()

            elif ticker in level1:

                result = data.xs(
                    ticker,
                    axis=1,
                    level=1
                ).copy()

            else:

                return pd.DataFrame()

        else:

            result = data.copy()

        if isinstance(
            result.columns,
            pd.MultiIndex
        ):

            result.columns = [
                str(c[-1])
                for c in result.columns
            ]

        wanted = [
            "Open",
            "High",
            "Low",
            "Close"
        ]

        available = [
            c
            for c in wanted
            if c in result.columns
        ]

        if "Close" not in available:

            return pd.DataFrame()

        result = result[
            available
        ].copy()

        for column in result.columns:

            result[column] = pd.to_numeric(
                result[column],
                errors="coerce"
            )

        return result.dropna(
            how="all"
        )

    except Exception:

        return pd.DataFrame()


# =========================================================
# INDIA DATE
# =========================================================

def to_india_date(value):

    try:

        timestamp = pd.Timestamp(
            value
        )

        if timestamp.tzinfo is not None:

            timestamp = timestamp.tz_convert(
                "Asia/Kolkata"
            )

        return timestamp.date()

    except Exception:

        return None


# =========================================================
# PREVIOUS MONTH CLOSE
# =========================================================

def previous_month_close(
    history,
    today
):

    if history.empty:

        return np.nan

    target = (
        pd.Timestamp(today)
        - pd.DateOffset(months=1)
    )

    target_date = target.date()

    dates = pd.Series(
        [
            to_india_date(x)
            for x in history.index
        ],
        index=history.index
    )

    valid = history.loc[
        dates <= target_date
    ].copy()

    valid = valid.dropna(
        subset=["Close"]
    )

    if valid.empty:

        return np.nan

    return float(
        valid["Close"].iloc[-1]
    )


# =========================================================
# CALCULATE ONE STOCK
# =========================================================

def calculate(
    symbol,
    daily_data,
    intraday_data,
    today
):

    ticker = symbol + ".NS"

    # -----------------------------------------------------
    # DAILY DATA
    # -----------------------------------------------------

    daily = get_one_ticker(
        daily_data,
        ticker
    )

    if daily.empty:

        return None

    daily = daily.dropna(
        subset=["Close"]
    )

    if daily.empty:

        return None

    # -----------------------------------------------------
    # COMPLETED DAILY HISTORY
    # -----------------------------------------------------

    dates = pd.Series(
        [
            to_india_date(x)
            for x in daily.index
        ],
        index=daily.index
    )

    history = daily.loc[
        dates < today
    ].copy()

    if history.empty:

        history = daily.copy()

    # =====================================================
    # LIVE PRICE
    # =====================================================

    live_price = np.nan

    intraday = get_one_ticker(
        intraday_data,
        ticker
    )

    if (
        not intraday.empty
        and
        "Close" in intraday.columns
    ):

        prices = (
            intraday["Close"]
            .dropna()
        )

        if not prices.empty:

            live_price = float(
                prices.iloc[-1]
            )

    # -----------------------------------------------------
    # FALLBACK
    # -----------------------------------------------------

    if (
        not np.isfinite(live_price)
        or
        live_price <= 0
    ):

        live_price = float(
            daily["Close"].iloc[-1]
        )

    # =====================================================
    # 1 DAY RETURN
    # =====================================================

    previous_close = float(
        history["Close"].iloc[-1]
    )

    if previous_close > 0:

        one_day = (
            live_price /
            previous_close -
            1
        ) * 100

    else:

        one_day = np.nan

    # =====================================================
    # 1 WEEK RETURN
    #
    # 5 TRADING SESSIONS
    # =====================================================

    if len(history) >= 6:

        week_close = float(
            history["Close"].iloc[-6]
        )

        if week_close > 0:

            one_week = (
                live_price /
                week_close -
                1
            ) * 100

        else:

            one_week = np.nan

    else:

        one_week = np.nan

    # =====================================================
    # 1 MONTH RETURN
    #
    # PREVIOUS MONTH'S CORRESPONDING DATE
    # =====================================================

    month_close = previous_month_close(
        history,
        today
    )

    if (
        np.isfinite(month_close)
        and
        month_close > 0
    ):

        one_month = (
            live_price /
            month_close -
            1
        ) * 100

    else:

        one_month = np.nan

    # =====================================================
    # EMA
    # =====================================================

    closes = history[
        "Close"
    ].copy()

    closes = pd.concat(
        [
            closes,
            pd.Series([live_price])
        ],
        ignore_index=True
    )

    ema21 = (
        closes
        .ewm(
            span=21,
            adjust=False
        )
        .mean()
        .iloc[-1]
    )

    ema50 = (
        closes
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
        .iloc[-1]
    )

    ema200 = (
        closes
        .ewm(
            span=200,
            adjust=False
        )
        .mean()
        .iloc[-1]
    )

    # =====================================================
    # 52 WEEK HIGH / LOW
    #
    # ACTUAL DAILY HIGH AND LOW
    # =====================================================

    cutoff = (
        pd.Timestamp(today)
        - pd.Timedelta(days=365)
    ).date()

    hist_dates = pd.Series(
        [
            to_india_date(x)
            for x in history.index
        ],
        index=history.index
    )

    last_52w = history.loc[
        hist_dates >= cutoff
    ].copy()

    if last_52w.empty:

        last_52w = history.copy()

    # -----------------------------------------------------
    # ACTUAL HIGH
    # -----------------------------------------------------

    if "High" in last_52w.columns:

        high52 = (
            pd.to_numeric(
                last_52w["High"],
                errors="coerce"
            )
            .max()
        )

    else:

        high52 = np.nan

    # -----------------------------------------------------
    # ACTUAL LOW
    # -----------------------------------------------------

    if "Low" in last_52w.columns:

        low52 = (
            pd.to_numeric(
                last_52w["Low"],
                errors="coerce"
            )
            .min()
        )

    else:

        low52 = np.nan

    # =====================================================
    # DISTANCE FROM 52W HIGH
    # =====================================================

    if (
        np.isfinite(high52)
        and
        high52 > 0
    ):

        from_high = (
            live_price /
            high52 -
            1
        ) * 100

    else:

        from_high = np.nan

    # =====================================================
    # DISTANCE FROM 52W LOW
    # =====================================================

    if (
        np.isfinite(low52)
        and
        low52 > 0
    ):

        from_low = (
            live_price /
            low52 -
            1
        ) * 100

    else:

        from_low = np.nan

    # =====================================================
    # DISTANCE FROM 21 EMA
    # =====================================================

    if (
        np.isfinite(ema21)
        and
        ema21 > 0
    ):

        from_ema = (
            live_price /
            ema21 -
            1
        ) * 100

    else:

        from_ema = np.nan

    # =====================================================
    # TREND
    # =====================================================

    if (
        live_price > ema21
        and
        ema21 > ema50
        and
        ema50 > ema200
    ):

        trend = "Bullish"

    elif (
        live_price < ema21
        and
        ema21 < ema50
        and
        ema50 < ema200
    ):

        trend = "Bearish"

    else:

        trend = "Neutral"

    # =====================================================
    # RESULT
    # =====================================================

    return {

        "Stock": symbol,

        "Price": live_price,

        "1D Return %": one_day,

        "1W Return %": one_week,

        "1M Return %": one_month,

        "21 EMA": ema21,

        "50 EMA": ema50,

        "200 EMA": ema200,

        "52W High": high52,

        "52W Low": low52,

        "From 52W High %": from_high,

        "From 52W Low %": from_low,

        "From 21 EMA %": from_ema,

        "Trend": trend
    }


# =========================================================
# TREND COLOR
# =========================================================

def color_trend(value):

    if value == "Bullish":

        return (
            "background-color: green; "
            "color: white; "
            "font-weight: bold;"
        )

    if value == "Bearish":

        return (
            "background-color: red; "
            "color: white; "
            "font-weight: bold;"
        )

    if value == "Neutral":

        return (
            "background-color: orange; "
            "color: black; "
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
    # GET STOCK LIST
    # =====================================================

    try:

        symbols = get_symbols()

    except Exception as e:

        st.error(
            "Unable to get Nifty 100 stocks."
        )

        st.exception(e)

        st.stop()

    st.info(
        f"Nifty 100 stocks found: {len(symbols)}"
    )

    # =====================================================
    # DAILY DATA
    # =====================================================

    with st.spinner(
        "Downloading daily data..."
    ):

        try:

            daily_data = get_daily_data(
                symbols
            )

        except Exception as e:

            st.error(
                "Daily data download failed."
            )

            st.exception(e)

            st.stop()

    # =====================================================
    # INTRADAY DATA
    # =====================================================

    with st.spinner(
        "Getting latest prices..."
    ):

        try:

            intraday_data = get_intraday_data(
                symbols
            )

        except Exception:

            intraday_data = pd.DataFrame()

            st.warning(
                "Intraday data unavailable. "
                "Latest daily close will be used "
                "as fallback."
            )

    # =====================================================
    # INDIA TIME
    # =====================================================

    ist = ZoneInfo(
        "Asia/Kolkata"
    )

    now = datetime.now(
        ist
    )

    today = now.date()

    # =====================================================
    # CALCULATE
    # =====================================================

    results = []

    failed = []

    progress = st.progress(0)

    total = len(symbols)

    for i, symbol in enumerate(symbols):

        try:

            result = calculate(
                symbol,
                daily_data,
                intraday_data,
                today
            )

            if result is not None:

                results.append(result)

            else:

                failed.append(symbol)

        except Exception as e:

            failed.append(
                symbol
                + " : "
                + str(e)
            )

        progress.progress(
            (i + 1) / total
        )

    progress.empty()

    # =====================================================
    # NO RESULTS
    # =====================================================

    if not results:

        st.error(
            "No stock data was calculated."
        )

        if failed:

            st.code(
                "\n".join(
                    failed[:20]
                )
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
    # NUMBER COLUMNS
    # =====================================================

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

    for column in number_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # =====================================================
    # ROUND TO 2 DECIMALS
    # =====================================================

    df[number_columns] = (
        df[number_columns]
        .round(2)
    )

    # =====================================================
    # SORT
    # =====================================================

    df = df.sort_values(
        by="From 21 EMA %",
        ascending=False,
        na_position="last"
    )

    df = df.reset_index(
        drop=True
    )

    # =====================================================
    # UPDATED TIME
    # =====================================================

    st.success(
        "Updated: "
        + now.strftime(
            "%d-%m-%Y %I:%M:%S %p IST"
        )
    )

    st.write(
        "Stocks calculated: "
        + str(len(df))
        + " / "
        + str(len(symbols))
    )

    # =====================================================
    # COLOUR TREND
    # =====================================================

    styled_df = df.style.applymap(
        color_trend,
        subset=["Trend"]
    )

    # =====================================================
    # DISPLAY TABLE
    # =====================================================

    st.dataframe(
        styled_df,
        use_container_width=True,
        height=700,
        hide_index=True
    )

    # =====================================================
    # DOWNLOAD CSV
    # =====================================================

    csv_data = df.to_csv(
        index=False
    ).encode("utf-8")

    st.downlo
