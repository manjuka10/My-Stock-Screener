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
    "Price uses the latest available Yahoo Finance intraday price."
)


# =========================================================
# NIFTY 100 URL
# =========================================================

NIFTY100_URL = (
    "https://www.niftyindices.com/IndexConstituent/"
    "ind_nifty100list.csv"
)


# =========================================================
# GET NIFTY 100
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
            "text/csv,application/csv,"
            "text/plain,*/*"
        ),
        "Referer": (
            "https://www.niftyindices.com/"
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
            "Nifty 100 CSV returned no data."
        )

    header = [
        str(x).strip()
        for x in rows[0]
    ]

    symbol_index = None

    for i, column in enumerate(header):

        if column.lower() == "symbol":

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
# DOWNLOAD DAILY DATA
# =========================================================

@st.cache_data(ttl=300)
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

@st.cache_data(ttl=120)
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
# GET INDIVIDUAL TICKER DATA
# =========================================================

def get_ticker_data(
    data,
    ticker
):

    if data is None:
        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    try:

        # -------------------------------------------------
        # MultiIndex
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

            # ---------------------------------------------
            # Format:
            # TICKER -> OHLC
            # ---------------------------------------------

            if ticker in level0:

                result = data[
                    ticker
                ].copy()

            # ---------------------------------------------
            # Format:
            # OHLC -> TICKER
            # ---------------------------------------------

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

        # -------------------------------------------------
        # Keep required columns
        # -------------------------------------------------

        required = [
            "Open",
            "High",
            "Low",
            "Close"
        ]

        available = [
            c
            for c in required
            if c in result.columns
        ]

        if "Close" not in available:

            return pd.DataFrame()

        result = result[available].copy()

        # -------------------------------------------------
        # Convert to numeric
        # -------------------------------------------------

        for column in result.columns:

            result[column] = pd.to_numeric(
                result[column],
                errors="coerce"
            )

        result = result.dropna(
            how="all"
        )

        return result

    except Exception:

        return pd.DataFrame()


# =========================================================
# GET DATE
# =========================================================

def get_date(index_value):

    try:

        ts = pd.Timestamp(
            index_value
        )

        if ts.tzinfo is not None:

            ts = ts.tz_convert(
                "Asia/Kolkata"
            )

        return ts.date()

    except Exception:

        return None


# =========================================================
# GET PREVIOUS MONTH CLOSE
# =========================================================

def get_previous_month_close(
    history,
    current_date
):

    if history.empty:

        return np.nan

    # -----------------------------------------------------
    # Example:
    #
    # Current date = 28 Aug
    # Target date = 28 Jul
    #
    # If target is holiday/weekend,
    # use latest available trading day before target.
    # -----------------------------------------------------

    current_ts = pd.Timestamp(
        current_date
    )

    target_ts = (
        current_ts -
        pd.DateOffset(months=1)
    )

    target_date = target_ts.date()

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


# =========================================================
# CALCULATE ONE STOCK
# =========================================================

def calculate_stock(
    symbol,
    daily_data,
    intraday_data,
    today
):

    ticker = symbol + ".NS"

    # -----------------------------------------------------
    # DAILY DATA
    # -----------------------------------------------------

    daily = get_ticker_data(
        daily_data,
        ticker
    )

    # -----------------------------------------------------
    # INTRADAY DATA
    # -----------------------------------------------------

    intraday = get_ticker_data(
        intraday_data,
        ticker
    )

    # -----------------------------------------------------
    # DAILY DATA REQUIRED
    # -----------------------------------------------------

    if daily.empty:

        return None

    if "Close" not in daily.columns:

        return None

    # -----------------------------------------------------
    # CLEAN
    # -----------------------------------------------------

    daily = daily.dropna(
        subset=["Close"]
    ).copy()

    if daily.empty:

        return None

    # -----------------------------------------------------
    # LIVE PRICE
    #
    # If intraday is available, use it.
    # Otherwise use latest daily close so the stock
    # can still appear in the table.
    # -----------------------------------------------------

    live_price = np.nan

    if (
        not intraday.empty
        and "Close" in intraday.columns
    ):

        intraday_close = (
            intraday["Close"]
            .dropna()
        )

        if not intraday_close.empty:

            live_price = float(
                intraday_close.iloc[-1]
            )

    # Fallback
    if (
        not np.isfinite(live_price)
        or live_price <= 0
    ):

        live_price = float(
            daily["Close"].iloc[-1]
        )

    # -----------------------------------------------------
    # COMPLETED DAILY DATA
    # -----------------------------------------------------

    dates = pd.Series(
        [
            get_date(x)
            for x in daily.index
        ],
        index=daily.index
    )

    historical = daily.loc[
        dates < today
    ].copy()

    if historical.empty:

        historical = daily.copy()

    # =====================================================
    # 1D RETURN
    # =====================================================

    if len(historical) >= 1:

        previous_close = float(
            historical["Close"].iloc[-1]
        )

        if previous_close > 0:

            one_day_return = (
                (
                    live_price /
                    previous_close
                ) - 1
            ) * 100

        else:

            one_day_return = np.nan

    else:

        one_day_return = np.nan

    # =====================================================
    # 1W RETURN
    #
    # ORIGINAL METHOD:
    # 5 TRADING SESSIONS
    # =====================================================

    if len(historical) >= 6:

        week_base = float(
            historical["Close"].iloc[-6]
        )

        if week_base > 0:

            one_week_return = (
                (
                    live_price /
                    week_base
                ) - 1
            ) * 100

        else:

            one_week_return = np.nan

    else:

        one_week_return = np.nan

    # =====================================================
    # 1M RETURN
    #
    # PREVIOUS MONTH'S CORRESPONDING DATE
    # =====================================================

    month_base = (
        get_previous_month_close(
            historical,
            today
        )
    )

    if (
        np.isfinite(month_base)
        and month_base > 0
    ):

        one_month_return = (
            (
                live_price /
                month_base
            ) - 1
        ) * 100

    else:

        one_month_return = np.nan

    # =====================================================
    # EMA
    # =====================================================

    close_for_ema = historical[
        "Close"
    ].copy()

    # Add live price as today's value
    live_series = pd.Series(
        [live_price],
        index=[
            pd.Timestamp.now()
        ]
    )

    close_for_ema = pd.concat(
        [
            close_for_ema,
            live_series
        ]
    )

    ema21 = float(
        close_for_ema.ewm(
            span=21,
            adjust=False
        ).mean().iloc[-1]
    )

    ema50 = float(
        close_for_ema.ewm(
            span=50,
            adjust=False
        ).mean().iloc[-1]
    )

    ema200 = float(
        close_for_ema.ewm(
            span=200,
            adjust=False
        ).mean().iloc[-1]
    )

    # =====================================================
    # 52 WEEK HIGH / LOW
    #
    # ACTUAL HIGH / LOW
    # NOT CLOSE
    # =====================================================

    cutoff = (
        pd.Timestamp(today)
        - pd.Timedelta(days=365)
    ).date()

    if (
        "High" in historical.columns
        and "Low" in historical.columns
    ):

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

        high_values = (
            pd.to_numeric(
                last_52w["High"],
                errors="coerce"
            )
            .dropna()
        )

        low_values = (
            pd.to_numeric(
                last_52w["Low"],
                errors="coerce"
            )
            .dropna()
        )

        if not high_values.empty:

            week52_high = float(
                high_values.max()
            )

        else:

            week52_high = np.nan

        if not low_values.empty:

            week52_low = float(
                low_values.min()
            )

        else:

            week52_low = np.nan

    else:

        week52_high = np.nan
        week52_low = np.nan

    # =====================================================
    # DISTANCE FROM 52W HIGH
    # =====================================================

    if (
        np.isfinite(week52_high)
        and week52_high > 0
    ):

        from_52w_high = (
            (
                live_price /
                week52_high
            ) - 1
        ) * 100

    else:

        from_52w_high = np.nan

    # =====================================================
    # DISTANCE FROM 52W LOW
    # =====================================================

    if (
        np.isfinite(week52_low)
        and week52_low > 0
    ):

        from_52w_low = (
            (
                live_price /
                week52_low
            ) - 1
        ) * 100

    else:

        from_52w_low = np.nan

    # =====================================================
    # DISTANCE FROM 21 EMA
    # =====================================================

    if (
        np.isfinite(ema21)
        and ema21 > 0
    ):

        from_21_ema = (
            (
                live_price /
                ema21
            ) - 1
        ) * 100

    else:

        from_21_ema = np.nan

    # =====================================================
    # TREND
    # =====================================================

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

    # =====================================================
    # RETURN RESULT
    # =====================================================

    return {

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

    if value == "Neutral":

        return (
            "background-color: #F5B642;"
            "color: black;"
            "font-weight: bold;"
        )

    if value == "Bearish":

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
    # NIFTY 100 LIST
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
        f"Nifty 100 stocks found: {len(symbols)}"
    )

    # =====================================================
    # DOWNLOAD DAILY
    # =====================================================

    with st.spinner(
        "Downloading daily data..."
    ):

        try:

            daily_data = (
                download_daily_data(
                    symbols
                )
            )

        except Exception as e:

            st.error(
                "Daily data download failed."
            )

            st.error(str(e))

            st.stop()

    # =====================================================
    # DOWNLOAD INTRADAY
    # =====================================================

    with st.spinner(
        "Downloading latest intraday prices..."
    ):

        try:

            intraday_data = (
                download_intraday_data(
                    symbols
                )
            )

        except Exception as e:

            # Do NOT stop the application.
            # Daily data will still be used.

            st.warning(
                "Intraday data could not be downloaded. "
                "Latest daily close will be used as fallback."
            )

            intraday_data = pd.DataFrame()

    # =====================================================
    # CURRENT DATE
    # =====================================================

    ist = ZoneInfo(
        "Asia/Kolkata"
    )

    now_ist = datetime.now(
        ist
    )

    today = now_ist.date()

    # =====================================================
    # CALCULATE
    # =====================================================

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

                results.append(result)

        except Exception:

            # One stock should never stop
            # the complete scan.

            pass

        progress.progress(
            int(
                ((i + 1) / total) * 100
            )
        )

    progress.empty()

    # =====================================================
    # CHECK RESULTS
    # =====================================================

    if len(results) == 0:

        st.error(
            "No stock data was calculated."
        )

        st.stop()

    # =====================================================
    # DATAFRAME
    # =====================================================

    df = pd.DataFrame(
        results
    )

    # ==============
