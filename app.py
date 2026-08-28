import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import csv
import io
from datetime import datetime
from dateutil.relativedelta import relativedelta
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
    "Latest available intraday price is used for calculations."
)


# =========================================================
# NIFTY 100 CONSTITUENTS
# =========================================================

NIFTY100_URL = (
    "https://www.niftyindices.com/IndexConstituent/"
    "ind_nifty100list.csv"
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
# DOWNLOAD STOCK DATA
# =========================================================

@st.cache_data(ttl=300)
def download_stock_data(symbols):

    tickers = [
        symbol + ".NS"
        for symbol in symbols
    ]

    # -----------------------------------------------------
    # DAILY DATA
    # Used for:
    # EMA
    # Returns
    # 52W High / Low
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
    # INTRADAY DATA
    # Used for latest available price
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
# GET ONE TICKER OHLC DATA
# =========================================================

def get_ticker_ohlc(
    data,
    ticker
):

    if data is None or data.empty:
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

            # Ticker -> OHLC
            if ticker in level0:

                temp = data[ticker].copy()

                wanted = [
                    c
                    for c in [
                        "Open",
                        "High",
                        "Low",
                        "Close"
                    ]
                    if c in temp.columns
                ]

                if wanted:

                    temp = temp[wanted]

                    temp = temp.apply(
                        pd.to_numeric,
                        errors="coerce"
                    )

                    return temp.dropna(
                        how="all"
                    )

            # OHLC -> Ticker
            if ticker in level1:

                temp = data.xs(
                    ticker,
                    axis=1,
                    level=1
                ).copy()

                wanted = [
                    c
                    for c in [
                        "Open",
                        "High",
                        "Low",
                        "Close"
                    ]
                    if c in temp.columns
                ]

                if wanted:

                    temp = temp[wanted]

                    temp = temp.apply(
                        pd.to_numeric,
                        errors="coerce"
                    )

                    return temp.dropna(
                        how="all"
                    )

        # -------------------------------------------------
        # Normal columns
        # -------------------------------------------------

        else:

            wanted = [
                c
                for c in [
                    "Open",
                    "High",
                    "Low",
                    "Close"
                ]
                if c in data.columns
            ]

            if wanted:

                temp = data[wanted].copy()

                temp = temp.apply(
                    pd.to_numeric,
                    errors="coerce"
                )

                return temp.dropna(
                    how="all"
                )

    except Exception:

        pass

    return pd.DataFrame()


# =========================================================
# GET DATE FROM INDEX
# =========================================================

def get_index_date(
    index_value
):

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
# GET CLOSE ON OR BEFORE A DATE
# =========================================================

def get_close_on_or_before(
    history,
    target_date
):

    if history.empty:
        return None

    dates = pd.Series(
        [
            get_index_date(x)
            for x in history.index
        ],
        index=history.index
    )

    valid = history.loc[
        dates <= target_date,
        "Close"
    ].dropna()

    if valid.empty:
        return None

    return float(
        valid.iloc[-1]
    )


# =========================================================
# PREVIOUS MONTH CORRESPONDING TRADING-DAY CLOSE
# =========================================================

def get_previous_month_close(
    history,
    current_date
):

    if history.empty:
        return None

    # Example:
    #
    # 28-Aug -> 28-Jul
    #
    # If 28-Jul is a holiday/weekend,
    # use the latest available trading-day
    # close on or before 28-Jul.

    current_timestamp = pd.Timestamp(
        current_date
    )

    previous_month_timestamp = (
        current_timestamp
        - pd.DateOffset(months=1)
    )

    target_date = (
        previous_month_timestamp.date()
    )

    return get_close_on_or_before(
        history,
        target_date
    )


# =========================================================
# CALCULATE STOCK INDICATORS
# =========================================================

def calculate_stock_data(
    symbol,
    daily_data,
    intraday_data
):

    ticker = symbol + ".NS"

    try:

        # -------------------------------------------------
        # DAILY DATA
        # -------------------------------------------------

        daily_ohlc = get_ticker_ohlc(
            daily_data,
            ticker
        )

        # -------------------------------------------------
        # INTRADAY DATA
        # -------------------------------------------------

        intraday_ohlc = get_ticker_ohlc(
            intraday_data,
            ticker
        )

        if daily_ohlc.empty:

            return (
                None,
                "daily data unavailable"
            )

        if (
            intraday_ohlc.empty
            or "Close"
            not in intraday_ohlc.columns
        ):

            return (
                None,
                "intraday data unavailable"
            )

        # -------------------------------------------------
        # CLEAN DATA
        # -------------------------------------------------

        daily_ohlc = daily_ohlc.dropna(
            subset=["Close"]
        ).copy()

        intraday_ohlc = intraday_ohlc.dropna(
            subset=["Close"]
        ).copy()

        # IMPORTANT:
        # No 220 trading-day requirement.

        if len(daily_ohlc) < 2:

            return (
                None,
                "insufficient historical data"
            )

        # =================================================
        # LIVE PRICE
        # =================================================

        live_price = float(
            intraday_ohlc["Close"].iloc[-1]
        )

        if (
            not np.isfinite(live_price)
            or live_price <= 0
        ):

            return (
                None,
                "invalid live price"
            )

        # =================================================
        # CURRENT IST DATE
        # =================================================

        ist = ZoneInfo(
            "Asia/Kolkata"
        )

        now_ist = datetime.now(
            ist
        )

        today = now_ist.date()

        # =================================================
        # COMPLETED DAILY DATA
        # =================================================

        daily_dates = pd.Series(
            [
                get_index_date(x)
                for x in daily_ohlc.index
            ],
            index=daily_ohlc.index
        )

        historical = daily_ohlc.loc[
            daily_dates < today
        ].copy()

        if historical.empty:

            return (
                None,
                "no completed daily history"
            )

        # =================================================
        # 1 DAY RETURN
        #
        # CURRENT/LIVE PRICE
        # VS PREVIOUS TRADING DAY CLOSE
        # =================================================

        previous_close = float(
            historical["Close"].iloc[-1]
        )

        one_day_return = (
            (
                live_price /
                previous_close
            ) - 1
        ) * 100

        # =================================================
        # 1 WEEK RETURN
        #
        # KEEP ORIGINAL METHOD
        #
        # 5 TRADING SESSIONS
        # =================================================

        if len(historical) >= 6:

            one_week_base = float(
                historical["Close"].iloc[-6]
            )

            one_week_return = (
                (
                    live_price /
                    one_week_base
                ) - 1
            ) * 100

        else:

            one_week_return = np.nan

        # =================================================
        # 1 MONTH RETURN
        #
        # NEW METHOD
        #
        # PREVIOUS MONTH'S CORRESPONDING
        # TRADING-DAY CLOSE
        #
        # Example:
        #
        # 28-Aug current price
        # 28-Jul closing price
        # =================================================

        one_month_base = (
            get_previous_month_close(
                historical,
                today
            )
        )

        if (
            one_month_base is None
            or one_month_base <= 0
        ):

            one_month_return = np.nan

        else:

            one_month_return = (
                (
                    live_price /
                    one_month_base
                ) - 1
            ) * 100

        # =================================================
        # EMA CALCULATION
        #
        # INCLUDE CURRENT LIVE PRICE
        # =================================================

        calc_close = pd.concat(
            [
                historical["Close"],

                pd.Series(
                    [live_price],
                    index=[
                        pd.Timestamp.now()
                    ]
                )
            ]
        )

        ema21 = float(
            calc_close.ewm(
                span=21,
                adjust=False
            ).mean().iloc[-1]
        )

        ema50 = float(
            calc_close.ewm(
                span=50,
                adjust=False
            ).mean().iloc[-1]
        )

        ema200 = float(
            calc_close.ewm(
                span=200,
                adjust=False
            ).mean().iloc[-1]
        )

        # =================================================
        # 52 WEEK HIGH / LOW
        #
        # ACTUAL HIGH AND LOW
        # NOT CLOSING PRICE
        # =================================================

        cutoff_date = (
            today -
            pd.Timedelta(days=365)
        )

        recent_52w = historical.loc[
            [
                get_index_date(x)
                >= cutoff_date
                for x in historical.index
            ]
        ].copy()

        if recent_52w.empty:

            return (
                None,
                "52-week data unavailable"
            )

        week52_high = float(
            recent_52w["High"].max()
        )

        week52_low = float(
            recent_52w["Low"].min()
        )

        # =================================================
        # DISTANCE FROM 52 WEEK HIGH
        # =================================================

        from_52w_high = (
            (
                live_price /
                week52_high
            ) - 1
        ) * 100

        # =================================================
        # DISTANCE FROM 52 WEEK LOW
        # =================================================

        from_52w_low = (
            (
                live_price /
                week52_low
            ) - 1
        ) * 100

        # =================================================
        # DISTANCE FROM 21 EMA
        # =================================================

        from_21_ema = (
            (
                live_price /
                ema21
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
        # RETURN
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

def colour_trend(
    value
):

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
            "Unable to get the current Nifty 100 list."
        )

        st.error(str(e))

        st.stop()

    # =====================================================
    # DOWNLOAD DATA
    # =====================================================

    with st.spinner(
        "Downloading Nifty 100 market data..."
    ):

        try:

            daily_data, intraday_data = (
                download_stock_data(
                    symbols
                )
            )

        except Exception as e:

            st.error(
                "Unable to download stock data."
            )

            st.error(str(e))

            st.stop()

    # =====================================================
    # CALCULATE INDICATORS
    # =====================================================

    results = []

    unavailable = []

    progress = st.progress(0)

    total = len(symbols)

    for i, symbol in enumerate(symbols):

        result, reason = (
            calculate_stock_data(
                symbol,
                daily_data,
                intraday_data
            )
        )

        if result is not None:

            results.append(result)

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
    # CHECK RESULTS
    # =====================================================

    if not results:

        st.error(
            "No stock data could be calculated."
        )

        st.stop()

    # =====================================================
    # DATAFRAME
    # =============================================
