import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import csv
import io
from datetime import datetime
from zoneinfo import ZoneInfo


st.set_page_config(
    page_title="Nifty 100 Stock Screener",
    layout="wide"
)

st.title("📊 Nifty 100 Stock Screener")
st.caption("Live price + 1D / 1W / 1M returns + 52W High/Low")


NIFTY_URL = (
    "https://www.niftyindices.com/IndexConstituent/"
    "ind_nifty100list.csv"
)


@st.cache_data(ttl=86400)
def get_symbols():

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.niftyindices.com/"
    }

    r = requests.get(
        NIFTY_URL,
        headers=headers,
        timeout=30
    )

    r.raise_for_status()

    text = r.content.decode(
        "utf-8-sig",
        errors="replace"
    )

    rows = list(
        csv.reader(
            io.StringIO(text)
        )
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
            "Symbol column not found in Nifty 100 file."
        )

    symbols = []

    for row in rows[1:]:

        if len(row) > symbol_col:

            symbol = row[symbol_col].strip()

            if symbol:
                symbols.append(symbol)

    return list(dict.fromkeys(symbols))


@st.cache_data(ttl=300)
def get_daily_data(symbols):

    tickers = [
        s + ".NS"
        for s in symbols
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


@st.cache_data(ttl=120)
def get_intraday_data(symbols):

    tickers = [
        s + ".NS"
        for s in symbols
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


def get_one_ticker(data, ticker):

    if data is None or data.empty:
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

                x = data[ticker].copy()

            elif ticker in level1:

                x = data.xs(
                    ticker,
                    axis=1,
                    level=1
                ).copy()

            else:

                return pd.DataFrame()

        else:

            x = data.copy()

        if isinstance(
            x.columns,
            pd.MultiIndex
        ):

            x.columns = [
                str(c[-1])
                for c in x.columns
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
            if c in x.columns
        ]

        if "Close" not in available:
            return pd.DataFrame()

        x = x[available].copy()

        for c in x.columns:

            x[c] = pd.to_numeric(
                x[c],
                errors="coerce"
            )

        return x.dropna(
            how="all"
        )

    except Exception:

        return pd.DataFrame()


def to_india_date(value):

    try:

        x = pd.Timestamp(value)

        if x.tzinfo is not None:

            x = x.tz_convert(
                "Asia/Kolkata"
            )

        return x.date()

    except Exception:

        return None


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

    x = history.loc[
        dates <= target_date
    ].copy()

    x = x.dropna(
        subset=["Close"]
    )

    if x.empty:
        return np.nan

    return float(
        x["Close"].iloc[-1]
    )


def calculate(
    symbol,
    daily_data,
    intraday_data,
    today
):

    ticker = symbol + ".NS"

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

    # -----------------------------------------------------
    # LIVE PRICE
    # -----------------------------------------------------

    live = np.nan

    intraday = get_one_ticker(
        intraday_data,
        ticker
    )

    if (
        not intraday.empty
        and "Close" in intraday.columns
    ):

        prices = (
            intraday["Close"]
            .dropna()
        )

        if not prices.empty:

            live = float(
                prices.iloc[-1]
            )

    # Fallback
    if (
        not np.isfinite(live)
        or live <= 0
    ):

        live = float(
            daily["Close"].iloc[-1]
        )

    # -----------------------------------------------------
    # 1 DAY
    # -----------------------------------------------------

    prev_close = float(
        history["Close"].iloc[-1]
    )

    if prev_close > 0:

        one_day = (
            live / prev_close - 1
        ) * 100

    else:

        one_day = np.nan

    # -----------------------------------------------------
    # 1 WEEK
    #
    # Friday close to Friday close
    # / 5 trading sessions
    # -----------------------------------------------------

    if len(history) >= 6:

        week_close = float(
            history["Close"].iloc[-6]
        )

        if week_close > 0:

            one_week = (
                live / week_close - 1
            ) * 100

        else:

            one_week = np.nan

    else:

        one_week = np.nan

    # -----------------------------------------------------
    # 1 MONTH
    #
    # Previous month's corresponding date
    # -----------------------------------------------------

    month_close = previous_month_close(
        history,
        today
    )

    if (
        np.isfinite(month_close)
        and month_close > 0
    ):

        one_month = (
            live / month_close - 1
        ) * 100

    else:

        one_month = np.nan

    # -----------------------------------------------------
    # EMA
    # -----------------------------------------------------

    closes = history["Close"].copy()

    closes = pd.concat(
        [
            closes,
            pd.Series([live])
        ],
        ignore_index=True
    )

    ema21 = (
        closes
        .ewm(span=21, adjust=False)
        .mean()
        .iloc[-1]
    )

    ema50 = (
        closes
        .ewm(span=50, adjust=False)
        .mean()
        .iloc[-1]
    )

    ema200 = (
        closes
        .ewm(span=200, adjust=False)
        .mean()
        .iloc[-1]
    )

    # -----------------------------------------------------
    # 52 WEEK HIGH / LOW
    #
    # ACTUAL HIGH AND ACTUAL LOW
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # DISTANCE FROM 52W HIGH / LOW
    # -----------------------------------------------------

    if (
        np.isfinite(high52)
        and high52 > 0
    ):

        from_high = (
            live / high52 - 1
        ) * 100

    else:

        from_high = np.nan

    if (
        np.isfinite(low52)
        and low52 > 0
    ):

        from_low = (
            live / low52 - 1
        ) * 100

    else:

        from_low = np.nan

    # -----------------------------------------------------
    # DISTANCE FROM EMA21
    # -----------------------------------------------------

    if (
        np.isfinite(ema21)
        and ema21 > 0
    ):

        from_ema = (
            live / ema21 - 1
        ) * 100

    else:

        from_ema = np.nan

    # -----------------------------------------------------
    # TREND
    # -----------------------------------------------------

    if (
        live > ema21
        and ema21 > ema50
        and ema50 > ema200
    ):

        trend = "Bullish"

    elif (
        live < ema21
        and ema21 < ema50
        and ema50 < ema200
    ):

        trend = "Bearish"

    else:

        trend = "Neutral"

    return {
        "Stock": symbol,
        "Price": live,
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
# SCAN
# =========================================================

if st.button("🔍 Scan Nifty 100"):

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

    with st.spinner(
        "Getting latest prices..."
    ):

        try:

            intraday_data = (
                get_intraday_data(
                    symbols
                )
            )

        except Exception:

            intraday_data = pd.DataFrame()

    ist = ZoneInfo(
        "Asia/Kolkata"
    )

    now = datetime.now(
        ist
    )

    today = now.date()

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

    if not results:

        st.error(
            "No stock data was calculated."
        )

        if failed:

            st.write(
                "Errors:"
            )

            st.code(
                "\n".join(
                    failed[:20]
                )
            )

        st.stop()

    df = pd.DataFrame(
        results
    )

    # -----------------------------------------------------
    # COLUMN ORDER
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # ROUND TO 2 DECIMALS
    # -----------------------------------------------------

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
        ).round(2)

    # -----------------------------------------------------
    # SORT BY DISTANCE FROM EMA21
    # -----------------------------------------------------

    df = df.sort_values(
        "From 21 EMA %",
        ascending=False,
        na_position="last"
    )

    df = df.reset_index(
        drop=True
    )

    # -----------------------------------------------------
    # TIME
    # -----------------------------------------------------

    st.success(
        "Updated: "
        + now.strftime(
            "%d-%m-%Y %I:%M:%S %p IST"
        )
    )

    st.write(
        f"Stocks calculated: {len(df)} / {len(symbols)}"
    )

    # -----------------------------------------------------
    # TABLE
    # -----------------------------------------------------

    st.dataframe(
        df,
        use_container_width=True,
        height=700,
        hide_index=True
    )

    # -----------------------------------------------------
    # DOWNLOAD
    # -----------------------------------------------------

    csv_data = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download CSV",
        csv_data,
        "nifty100_screener.csv",
        "text/csv"
    )

    # -----------------------------------------------------
    # FAILED STOCKS
    # -----------------------------------------------------

    if failed:

        with st.expander(
            "Stocks with unavailable data"
        ):

            st.write(
                failed
        )
