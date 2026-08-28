import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import csv
import io
from datetime import datetime
from zoneinfo import ZoneInfo


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="My Stock Screener",
    page_icon="📊",
    layout="wide"
)

st.title("📊 My Stock Screener")
st.subheader("Nifty 100 Technical Screener")

st.caption(
    "Latest available intraday price is used as current price."
)


# ============================================================
# NIFTY 100 URL
# ============================================================

NIFTY100_URL = (
    "https://www.niftyindices.com/IndexConstituent/"
    "ind_nifty100list.csv"
)


# ============================================================
# GET NIFTY 100
# ============================================================

@st.cache_data(ttl=86400)
def get_nifty100_list():

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
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

    symbol_index = None

    for i, name in enumerate(header):

        if name.lower() == "symbol":
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

    return list(
        dict.fromkeys(symbols)
    )


# ============================================================
# DOWNLOAD DAILY DATA
# ============================================================

@st.cache_data(ttl=300)
def download_daily_data(symbols):

    tickers = [
        s + ".NS"
        for s in symbols
    ]

    return yf.download(
        tickers=tickers,
        period="2y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True
    )


# ============================================================
# DOWNLOAD INTRADAY DATA
# ============================================================

@st.cache_data(ttl=120)
def download_intraday_data(symbols):

    tickers = [
        s + ".NS"
        for s in symbols
    ]

    return yf.download(
        tickers=tickers,
        period="1d",
        interval="5m",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True
    )


# ============================================================
# GET INDIVIDUAL STOCK DATA
# ============================================================

def get_ticker_data(data, ticker):

    if data is None:
        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    try:

        if isinstance(
            data.columns,
            pd.MultiIndex
        ):

            level0 = data.columns.get_level_values(0)
            level1 = data.columns.get_level_values(1)

            if ticker in level0:

                result = data[ticker].copy()

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

        result = result[available].copy()

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


# ============================================================
# GET DATE
# ============================================================

def get_date(value):

    try:

        ts = pd.Timestamp(value)

        if ts.tzinfo is not None:

            ts = ts.tz_convert(
                "Asia/Kolkata"
            )

        return ts.date()

    except Exception:

        return None


# ============================================================
# PREVIOUS MONTH CLOSE
# ============================================================

def get_previous_month_close(
    history,
    current_date
):

    if history.empty:
        return np.nan

    target = (
        pd.Timestamp(current_date)
        - pd.DateOffset(months=1)
    )

    target_date = target.date()

    dates = pd.Series(
        [
            get_date(x)
            for x in history.index
        ],
        index=history.index
    )

    valid = history.loc[
        dates <= target_date
    ]

    if valid.empty:
        return np.nan

    valid = valid.dropna(
        subset=["Close"]
    )

    if valid.empty:
        return np.nan

    return float(
        valid["Close"].iloc[-1]
    )


# ============================================================
# CALCULATE STOCK
# ============================================================

def calculate_stock(
    symbol,
    daily_data,
    intraday_data,
    today
):

    ticker = symbol + ".NS"

    # --------------------------------------------------------
    # DAILY
    # --------------------------------------------------------

    daily = get_ticker_data(
        daily_data,
        ticker
    )

    if daily.empty:
        return None

    if "Close" not in daily.columns:
        return None

    daily = daily.dropna(
        subset=["Close"]
    ).copy()

    if daily.empty:
        return None

    # --------------------------------------------------------
    # INTRADAY
    # --------------------------------------------------------

    intraday = get_ticker_data(
        intraday_data,
        ticker
    )

    # --------------------------------------------------------
    # CURRENT PRICE
    # --------------------------------------------------------

    current_price = np.nan

    if (
        not intraday.empty
        and "Close" in intraday.columns
    ):

        prices = (
            intraday["Close"]
            .dropna()
        )

        if not prices.empty:

            current_price = float(
                prices.iloc[-1]
            )

    # Fallback

    if (
        not np.isfinite(current_price)
        or current_price <= 0
    ):

        current_price = float(
            daily["Close"].iloc[-1]
        )

    # --------------------------------------------------------
    # DATES
    # --------------------------------------------------------

    dates = pd.Series(
        [
            get_date(x)
            for x in daily.index
        ],
        index=daily.index
    )

    # --------------------------------------------------------
    # HISTORICAL DAILY DATA
    # --------------------------------------------------------

    historical = daily.loc[
        dates < today
    ].copy()

    if historical.empty:

        historical = daily.copy()

    historical = historical.dropna(
        subset=["Close"]
    )

    if historical.empty:
        return None

    # ========================================================
    # 1 DAY RETURN
    # ========================================================

    if len(historical) >= 1:

        previous_close = float(
            historical["Close"].iloc[-1]
        )

        if previous_close > 0:

            one_day = (
                current_price /
                previous_close
                - 1
            ) * 100

        else:

            one_day = np.nan

    else:

        one_day = np.nan

    # ========================================================
    # 1 WEEK RETURN
    # 5 TRADING SESSIONS BACK
    # ========================================================

    if len(historical) >= 5:

        week_close = float(
            historical["Close"].iloc[-5]
        )

        if week_close > 0:

            one_week = (
                current_price /
                week_close
                - 1
            ) * 100

        else:

            one_week = np.nan

    else:

        one_week = np.nan

    # ========================================================
    # 1 MONTH RETURN
    # PREVIOUS MONTH CORRESPONDING DATE
    # ========================================================

    month_close = get_previous_month_close(
        historical,
        today
    )

    if (
        np.isfinite(month_close)
        and month_close > 0
    ):

        one_month = (
            current_price /
            month_close
            - 1
        ) * 100

    else:

        one_month = np.nan

    # ========================================================
    # EMA
    # ========================================================

    ema_data = historical["Close"].copy()

    ema_data.loc[
        pd.Timestamp.now(
            tz="Asia/Kolkata"
        )
    ] = current_price

    ema21 = float(
        ema_data
        .ewm(
            span=21,
            adjust=False
        )
        .mean()
        .iloc[-1]
    )

    ema50 = float(
        ema_data
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
        .iloc[-1]
    )

    ema200 = float(
        ema_data
        .ewm(
            span=200,
            adjust=False
        )
        .mean()
        .iloc[-1]
    )

    # ========================================================
    # 52 WEEK HIGH / LOW
    # ACTUAL HIGH AND LOW
    # NO 220 TRADING-DAY RULE
    # ========================================================

    cutoff = (
        pd.Timestamp(today)
        - pd.Timedelta(days=365)
    ).date()

    historical_dates = pd.Series(
        [
            get_date(x)
            for x in historical.index
        ],
        index=historical.index
    )

    last_52w = historical.loc[
        historical_dates >= cutoff
    ].copy()

    if last_52w.empty:

        last_52w = historical.copy()

    # --------------------------------------------------------
    # 52 WEEK HIGH
    # --------------------------------------------------------

    if "High" in last_52w.columns:

        high_values = pd.to_numeric(
            last_52w["High"],
            errors="coerce"
        ).dropna()

        if not high_values.empty:

            week52_high = float(
                high_values.max()
            )

        else:

            week52_high = np.nan

    else:

        week52_high = np.nan

    # --------------------------------------------------------
    # 52 WEEK LOW
    # --------------------------------------------------------

    if "Low" in last_52w.columns:

        low_values = pd.to_numeric(
            last_52w["Low"],
            errors="coerce"
        ).dropna()

        if not low_values.empty:

            week52_low = float(
                low_values.min()
            )

        else:

            week52_low = np.nan

    else:

        week52_low = np.nan

    # ========================================================
    # DISTANCE FROM 52 WEEK HIGH
    # ========================================================

    if (
        np.isfinite(week52_high)
        and week52_high > 0
    ):

        from_high = (
            current_price /
            week52_high
            - 1
        ) * 100

    else:

        from_high = np.nan

    # ========================================================
    # DISTANCE FROM 52 WEEK LOW
    # ========================================================

    if (
        np.isfinite(week52_low)
        and week52_low > 0
    ):

        from_low = (
            current_price /
            week52_low
            - 1
        ) * 100

    else:

        from_low = np.nan

    # ========================================================
    # DISTANCE FROM 21 EMA
    # ========================================================

    if (
        np.isfinite(ema21)
        and ema21 > 0
    ):

        from_ema21 = (
            current_price /
            ema21
            - 1
        ) * 100

    else:

        from_ema21 = np.nan

    # ========================================================
    # TREND
    # ========================================================

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

    # ========================================================
    # RESULT
    # ========================================================

    return {
        "Stock": symbol,
        "Price": current_price,

        "1D Return %": one_day,
        "1W Return %": one_week,
        "1M Return %": one_month,

        "21 EMA": ema21,
        "50 EMA": ema50,
        "200 EMA": ema200,

        "52W High": week52_high,
        "52W Low": week52_low,

        "From 52W High %": from_high,
        "From 52W Low %": from_low,
        "From 21 EMA %": from_ema21,

        "Trend": trend
    }


# ============================================================
# SCAN BUTTON
# ============================================================

if st.button(
    "🔍 Scan Nifty 100",
    type="primary"
):

    # ========================================================
    # GET NIFTY 100
    # ========================================================

    try:

        symbols = get_nifty100_list()

    except Exception as e:

        st.error(
            "Unable to download Nifty 100 list."
        )

        st.error(
            str(e)
        )

        st.stop()

    st.info(
        "Nifty 100 stocks found: "
        + str(len(symbols))
    )

    # ========================================================
    # DAILY DATA
    # ========================================================

    with st.spinner(
        "Downloading daily data..."
    ):

        try:

            daily_data = download_daily_data(
                symbols
            )

        except Exception as e:

            st.error(
                "Daily data download failed."
            )

            st.error(
                str(e)
            )

            st.stop()

    # ========================================================
    # INTRADAY DATA
    # ========================================================

    with st.spinner(
        "Downloading latest prices..."
    ):

        try:

            intraday_data = (
                download_intraday_data(
                    symbols
                )
            )

        except Exception:

            intraday_data = pd.DataFrame()

            st.warning(
                "Live intraday data is unavailable. "
                "Latest daily close will be used as fallback."
            )

    # ========================================================
    # CURRENT TIME
    # ========================================================

    ist = ZoneInfo(
        "Asia/Kolkata"
    )

    now_ist = datetime.now(
        ist
    )

    today = now_ist.date()

    # ========================================================
    # CALCULATE
    # ========================================================

    results = []

    progress = st.progress(0)

    total = len(symbols)

    for i, symbol in enumerate(symbols):

        try:

            result = calculate_stock(
                symbol,
                daily_data,
                intraday_data,
                today
            )

            if result is not None:

                results.append(
                    result
                )

        except Exception:

            continue

        progress.progress(
            int(
                (i + 1)
                / total
                * 100
            )
        )

    progress.empty()

    # ========================================================
    # CHECK RESULTS
    # ========================================================

    if not results:

        st.error(
            "No stock data was calculated."
        )

        st.stop()

    # ========================================================
    # DATAFRAME
    # ========================================================

    df = pd.DataFrame(
        results
    )

    # ========================================================
    # COLUMN ORDER
    # ========================================================

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

    # ========================================================
    # SORT
    # ========================================================

    df = df.sort_values(
        "From 21 EMA %",
        ascending=False,
        na_position="last"
    )

    df = df.reset_index(
        drop=True
    )

    # ========================================================
    # ROUND TO 2 DECIMALS
    # ========================================================

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

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        ).round(2)

    # ========================================================
    # TIMESTAMP
    # ========================================================

    updated_time = now_ist.strftime(
        "%d-%m-%Y %I:%M:%S %p IST"
    )

    st.success(
        "🕐 Last updated: "
        + updated_time
    )

    st.subheader(
        "📋 Results — "
        + str(len(df))
        + " stocks"
    )

    # ========================================================
    # TREND COLOURS
    #
    # Uses Streamlit's native dataframe styling.
    # Calculations are NOT changed.
    # ========================================================

    def trend_background(value):

        if value == "Bullish":
            return "background-color: green; color: white;"

        if value == "Neutral":
            return "background-color: orange; color: black;"

        if value == "Bearish":
            return "background-color: red; color: white;"

        return ""


    # ========================================================
    # STYLE TABLE
    # ========================================================

    styled_df = df.style.map(
        trend_background,
        subset=["Trend"]
    )

    # ========================================================
    # DISPLAY TABLE
    # ========================================================

    st.dataframe(
        styled_df,
        use_container_width=True,
        height=650,
        hide_index=True
    )

    # ========================================================
    # DOWNLOAD CSV
    # ========================================================

    csv_data = df.to_csv(
        index=False
    ).encode(
        "utf-8"
    )

    st.download_button(
        label="⬇️ Download Results CSV",
        data=csv_data,
        file_name="nifty100_screener.csv",
        mime="text/csv"
    )
