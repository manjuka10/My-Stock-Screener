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
st.caption("Live mode: price, returns, 21/50/200 EMA, EMA distance and 52W High/Low update with the latest intraday data. EMA uses completed daily closes plus the current live price.")

NIFTY100_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv"
IST = ZoneInfo("Asia/Kolkata")


# ============================================================
# NIFTY 100 LIST
# ============================================================

@st.cache_data(ttl=86400)
def get_symbols():
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "Referer": "https://www.niftyindices.com/"
    }

    r = requests.get(NIFTY100_URL, headers=headers, timeout=30)
    r.raise_for_status()

    text = r.content.decode("utf-8-sig", errors="replace")
    rows = list(csv.reader(io.StringIO(text)))

    if len(rows) < 2:
        raise ValueError("Nifty 100 list is empty.")

    header = [str(x).strip().lower() for x in rows[0]]

    if "symbol" not in header:
        raise ValueError("Symbol column not found in Nifty 100 file.")

    idx = header.index("symbol")
    symbols = []

    for row in rows[1:]:
        if len(row) > idx:
            symbol = row[idx].strip().upper()
            if symbol:
                symbols.append(symbol)

    return list(dict.fromkeys(symbols))


# ============================================================
# DOWNLOAD DATA
# ============================================================

@st.cache_data(ttl=300)
def get_daily(symbols):
    tickers = [s + ".NS" for s in symbols]

    return yf.download(
        tickers=tickers,
        period="2y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True
    )


@st.cache_data(ttl=30)
def get_intraday(symbols):
    tickers = [s + ".NS" for s in symbols]

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
# EXTRACT ONE TICKER
# ============================================================

def get_ticker_data(data, ticker):
    if data is None or data.empty:
        return pd.DataFrame()

    try:
        if isinstance(data.columns, pd.MultiIndex):
            level0 = data.columns.get_level_values(0)
            level1 = data.columns.get_level_values(1)

            if ticker in level0:
                result = data[ticker].copy()
            elif ticker in level1:
                result = data.xs(ticker, axis=1, level=1).copy()
            else:
                return pd.DataFrame()
        else:
            result = data.copy()

        wanted = ["Open", "High", "Low", "Close"]
        cols = [c for c in wanted if c in result.columns]

        if "Close" not in cols:
            return pd.DataFrame()

        result = result[cols].copy()

        for c in cols:
            result[c] = pd.to_numeric(result[c], errors="coerce")

        return result.dropna(how="all")

    except Exception:
        return pd.DataFrame()


# ============================================================
# DATE HELPERS
# ============================================================

def index_to_dates(index):
    values = []

    for x in index:
        try:
            ts = pd.Timestamp(x)

            if ts.tzinfo is not None:
                ts = ts.tz_convert(IST)

            values.append(ts.date())
        except Exception:
            values.append(None)

    return pd.Series(values, index=index)


def previous_month_same_or_previous_trading_close(history, current_date):
    """
    Monthly return base:
    current date -> same calendar date in previous month.

    Example:
    28-Aug -> 28-Jul close.

    If that date was not a trading day, use the latest available
    trading-day close on or before the target date.
    """

    if history.empty:
        return np.nan

    current_ts = pd.Timestamp(current_date)
    target_ts = current_ts - pd.DateOffset(months=1)
    target_date = target_ts.date()

    dates = index_to_dates(history.index)

    valid = history.loc[dates <= target_date].copy()
    valid = valid.dropna(subset=["Close"])

    if valid.empty:
        return np.nan

    return float(valid["Close"].iloc[-1])


# ============================================================
# CALCULATE ONE STOCK
# ============================================================

def calculate_stock(symbol, daily_all, intraday_all, today, now_ist):
    ticker = symbol + ".NS"

    daily = get_ticker_data(daily_all, ticker)

    if daily.empty or "Close" not in daily.columns:
        return None

    daily = daily.dropna(subset=["Close"]).copy()

    if daily.empty:
        return None

    intraday = get_ticker_data(intraday_all, ticker)

    # --------------------------------------------------------
    # CURRENT PRICE
    # --------------------------------------------------------

    current_price = np.nan

    if not intraday.empty and "Close" in intraday.columns:
        p = intraday["Close"].dropna()

        if not p.empty:
            current_price = float(p.iloc[-1])

    # Fallback to latest daily close
    if not np.isfinite(current_price) or current_price <= 0:
        current_price = float(daily["Close"].iloc[-1])

    # --------------------------------------------------------
    # COMPLETED DAILY SESSIONS
    # --------------------------------------------------------

    daily_dates = index_to_dates(daily.index)

    historical = daily.loc[daily_dates < today].copy()
    historical = historical.dropna(subset=["Close"])

    if historical.empty:
        historical = daily.copy()

    historical = historical.dropna(subset=["Close"])

    if historical.empty:
        return None

    # --------------------------------------------------------
    # 1D RETURN
    # Current price vs previous completed trading-day close
    # --------------------------------------------------------

    previous_close = float(historical["Close"].iloc[-1])

    if previous_close > 0:
        one_day = (current_price / previous_close - 1.0) * 100.0
    else:
        one_day = np.nan

    # --------------------------------------------------------
    # 1W RETURN
    # Current price vs close 5 completed trading sessions back
    # --------------------------------------------------------

    if len(historical) >= 5:
        week_base = float(historical["Close"].iloc[-5])

        if week_base > 0:
            one_week = (current_price / week_base - 1.0) * 100.0
        else:
            one_week = np.nan
    else:
        one_week = np.nan

    # --------------------------------------------------------
    # 1M RETURN
    # Current price vs previous month's corresponding date close
    # --------------------------------------------------------

    # 1-month return:
    # During market hours, use today's calendar trading date.
    # On a holiday/weekend, use the most recent completed trading date.
    # Example: live Aug-28 price -> Jul-28 close.
    completed_dates = index_to_dates(historical.index)

    if len(historical) >= 1:
        if not intraday.empty:
            reference_date = today
        else:
            reference_date = completed_dates.iloc[-1]

        target_date = (
            pd.Timestamp(reference_date)
            - pd.DateOffset(months=1)
        ).date()

        month_base_rows = historical.loc[
            completed_dates <= target_date
        ].dropna(subset=["Close"])

        if not month_base_rows.empty:
            month_base = float(
                month_base_rows["Close"].iloc[-1]
            )
        else:
            month_base = np.nan
    else:
        month_base = np.nan

    if (
        np.isfinite(month_base)
        and month_base > 0
    ):
        one_month = round(
            (
                current_price /
                month_base
                - 1.0
            ) * 100.0,
            2
        )
    else:
        one_month = np.nan

    # --------------------------------------------------------
    # EMA
    # Use completed daily closes + current live price
    # --------------------------------------------------------

    ema_series = historical["Close"].copy()

    live_index = pd.Timestamp(now_ist)

    ema_series.loc[live_index] = current_price

    ema21 = float(
        ema_series.ewm(span=21, adjust=False).mean().iloc[-1]
    )

    ema50 = float(
        ema_series.ewm(span=50, adjust=False).mean().iloc[-1]
    )

    ema200 = float(
        ema_series.ewm(span=200, adjust=False).mean().iloc[-1]
    )

    # --------------------------------------------------------
    # 52 WEEK HIGH / LOW
    #
    # IMPORTANT:
    # - 365 calendar-day window
    # - Historical HIGH and LOW, not closing prices
    # - Today's intraday HIGH and LOW included
    # - No 220-day approximation
    # --------------------------------------------------------

    cutoff_date = (
        pd.Timestamp(today) - pd.Timedelta(days=365)
    ).date()

    hist_dates = index_to_dates(historical.index)

    last_52w = historical.loc[hist_dates >= cutoff_date].copy()

    if last_52w.empty:
        last_52w = historical.copy()

    historical_high = np.nan
    historical_low = np.nan

    if "High" in last_52w.columns:
        highs = pd.to_numeric(
            last_52w["High"],
            errors="coerce"
        ).dropna()

        if not highs.empty:
            historical_high = float(highs.max())

    if "Low" in last_52w.columns:
        lows = pd.to_numeric(
            last_52w["Low"],
            errors="coerce"
        ).dropna()

        if not lows.empty:
            historical_low = float(lows.min())

    today_high = np.nan
    today_low = np.nan

    if not intraday.empty:
        if "High" in intraday.columns:
            h = pd.to_numeric(
                intraday["High"],
                errors="coerce"
            ).dropna()

            if not h.empty:
                today_high = float(h.max())

        if "Low" in intraday.columns:
            l = pd.to_numeric(
                intraday["Low"],
                errors="coerce"
            ).dropna()

            if not l.empty:
                today_low = float(l.min())

    candidates_high = []

    if np.isfinite(historical_high):
        candidates_high.append(historical_high)

    if np.isfinite(today_high):
        candidates_high.append(today_high)

    week52_high = max(candidates_high) if candidates_high else np.nan

    candidates_low = []

    if np.isfinite(historical_low):
        candidates_low.append(historical_low)

    if np.isfinite(today_low):
        candidates_low.append(today_low)

    week52_low = min(candidates_low) if candidates_low else np.nan

    # --------------------------------------------------------
    # DISTANCES
    # --------------------------------------------------------

    if np.isfinite(week52_high) and week52_high > 0:
        from_high = (current_price / week52_high - 1.0) * 100.0
    else:
        from_high = np.nan

    if np.isfinite(week52_low) and week52_low > 0:
        from_low = (current_price / week52_low - 1.0) * 100.0
    else:
        from_low = np.nan

    # Current price distance from 21 EMA
    if np.isfinite(ema21) and ema21 > 0:
        from_ema21 = (current_price / ema21 - 1.0) * 100.0
    else:
        from_ema21 = np.nan

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

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
# TREND COLOUR
# ============================================================

def colour_trend(value):
    if value == "Bullish":
        return "background-color: #198754; color: white; font-weight: bold;"
    if value == "Bearish":
        return "background-color: #dc3545; color: white; font-weight: bold;"
    if value == "Neutral":
        return "background-color: #f5b642; color: black; font-weight: bold;"
    return ""


# ============================================================
# AUTO REFRESH
# ============================================================

AUTO_REFRESH_SECONDS = 60

refresh_count = 0
if st.checkbox("🔄 Auto refresh every 60 seconds", value=True):
    refresh_count = st_autorefresh(
        interval=AUTO_REFRESH_SECONDS * 1000,
        key="options_screener_refresh"
    )

# ============================================================
# SCAN
# ============================================================

manual_scan = st.button("🔍 Scan Nifty 100", type="primary")
auto_scan = (
    refresh_count > 0
    and st.session_state.get("options_scan_df") is not None
    and refresh_count != st.session_state.get("options_last_refresh_count", -1)
)

if manual_scan or auto_scan:

    now_ist = datetime.now(IST)
    today = now_ist.date()
    # --------------------------------------------------------
    # SYMBOLS
    # --------------------------------------------------------

    try:
        symbols = get_symbols()
    except Exception as e:
        st.error("Unable to download Nifty 100 list.")
        st.error(str(e))
        st.stop()

    st.info(f"Nifty 100 stocks found: {len(symbols)}")

    # --------------------------------------------------------
    # DAILY
    # --------------------------------------------------------

    with st.spinner("Downloading daily data..."):
        try:
            daily_data = get_daily(symbols)
        except Exception as e:
            st.error("Daily data download failed.")
            st.error(str(e))
            st.stop()

    # --------------------------------------------------------
    # INTRADAY
    # --------------------------------------------------------

    with st.spinner("Downloading latest prices..."):
        try:
            intraday_data = get_intraday(symbols)
        except Exception:
            intraday_data = pd.DataFrame()
            st.warning(
                "Intraday data unavailable. Latest daily close will be used."
            )

    # --------------------------------------------------------
    # CALCULATE
    # --------------------------------------------------------

    results = []
    failed = []

    progress = st.progress(0)
    total = len(symbols)

    for i, symbol in enumerate(symbols):
        try:
            result = calculate_stock(
                symbol,
                daily_data,
                intraday_data,
                today,
                now_ist
            )

            if result is not None:
                results.append(result)
            else:
                failed.append(symbol)

        except Exception:
            failed.append(symbol)

        if total:
            progress.progress(int(((i + 1) / total) * 100))

    progress.empty()

    if not results:
        st.error("No stock data was calculated.")
        st.stop()

    # --------------------------------------------------------
    # DATAFRAME
    # --------------------------------------------------------

    df = pd.DataFrame(results)

    # KEEP EXACT COLUMN ORDER
    column_order = [
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

    df = df[column_order]

    # --------------------------------------------------------
    # NUMERIC ROUNDING
    # --------------------------------------------------------

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

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        ).round(2)

    # --------------------------------------------------------
    # SORT
    # Highest current price distance from 21 EMA first
    # --------------------------------------------------------

    df = df.sort_values(
        "From 21 EMA %",
        ascending=False,
        na_position="last"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # LAST UPDATE
    # --------------------------------------------------------

    updated_time = now_ist.strftime(
        "%d-%m-%Y %I:%M:%S %p IST"
    )

    st.success(
        f"🕐 Last updated: {updated_time}"
    )

    st.write(
        f"**Stocks calculated: {len(df)} / {len(symbols)}**"
    )

    if failed:
        st.caption(
            f"{len(failed)} stock(s) could not be calculated from Yahoo Finance data."
        )

    # --------------------------------------------------------
    # TABLE
    #
    # Uses Styler only for the Trend column.
    # The dataframe itself remains unchanged.
    # --------------------------------------------------------

    display_df = df.copy()

    # Explicit two-decimal display for every numeric column.
    display_format = {}
    for col in numeric_columns:
        display_format[col] = "{:.2f}"

    styled_df = display_df.style.format(
        display_format,
        na_rep="—"
    ).map(
        colour_trend,
        subset=["Trend"]
    )

    st.dataframe(
        styled_df,
        use_container_width=True,
        height=650,
        hide_index=True
    )
    st.session_state["options_scan_df"] = df.copy()
    st.session_state["options_symbols"] = symbols
    st.session_state["options_failed"] = failed
    st.session_state["options_last_updated"] = updated_time
    st.session_state["options_last_refresh_count"] = refresh_count

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Results CSV",
        data=csv_data,
        file_name="nifty100_options_screener.csv",
        mime="text/csv"
)
