import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import csv
import io
from datetime import datetime, time
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
    "Live mode: Price, 1D/1W/1M returns, 21/50/200 EMA, "
    "EMA distance and 52W High/Low update from the latest valid intraday data."
)

NIFTY100_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv"
IST = ZoneInfo("Asia/Kolkata")

# ============================================================
# SCAN STATE
# ============================================================

if "scan_started" not in st.session_state:
    st.session_state.scan_started = False

scan_col1, scan_col2 = st.columns([1, 1])
with scan_col1:
    scan_clicked = st.button("🔍 Scan Nifty 100", type="primary")
with scan_col2:
    refresh_clicked = st.button("🔄 Refresh Live Data")

if scan_clicked:
    st.session_state.scan_started = True

# ============================================================
# NIFTY 100 LIST
# ============================================================

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

@st.cache_data(ttl=15, show_spinner=False)
def get_intraday(symbols):
    tickers = [s + ".NS" for s in symbols]

    return yf.download(
        tickers=tickers,
        period="5d",
        interval="5m",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
        prepost=False
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
                result = data.xs(
                    ticker,
                    axis=1,
                    level=1
                ).copy()
            else:
                return pd.DataFrame()
        else:
            result = data.copy()

        wanted = ["Open", "High", "Low", "Close"]
        cols = [c for c in wanted if c in result.columns]

        if "Close" not in cols:
            return pd.DataFrame()

        result = result[cols].copy()

        for col in cols:
            result[col] = pd.to_numeric(
                result[col],
                errors="coerce"
            )

        return result.dropna(how="all")

    except Exception:
        return pd.DataFrame()

# ============================================================
# DATE / MARKET HELPERS
# ============================================================

def index_to_dates(index):
    values = []

    for value in index:
        try:
            ts = pd.Timestamp(value)
            if ts.tzinfo is not None:
                ts = ts.tz_convert(IST)
            values.append(ts.date())
        except Exception:
            values.append(None)

    return pd.Series(values, index=index)


def is_nse_session_now(now_ist):
    """Basic NSE cash-market session check: Mon-Fri, 09:15-15:30 IST.
    Actual intraday data must also contain today's date before it is used.
    """
    if now_ist.weekday() >= 5:
        return False

    market_open = time(9, 15)
    market_close = time(15, 30)
    return market_open <= now_ist.time() <= market_close


def latest_close_on_or_before(history, target_date):
    if history.empty:
        return np.nan

    dates = index_to_dates(history.index)
    valid = history.loc[dates <= target_date].dropna(subset=["Close"])

    if valid.empty:
        return np.nan

    return float(valid["Close"].iloc[-1])

# ============================================================
# CALCULATE ONE STOCK
# ============================================================

def calculate_stock_data(symbol, daily_all, intraday_all, now_ist):
    ticker = symbol + ".NS"

    try:
        daily = get_ticker_data(daily_all, ticker)
        if daily.empty or "Close" not in daily.columns:
            return None, "no daily data"

        daily = daily.dropna(subset=["Close"]).copy()
        if daily.empty:
            return None, "no daily close"

        today = now_ist.date()
        intraday = get_ticker_data(intraday_all, ticker)

        # Identify current-day intraday data and the newest session returned.
        today_intraday = pd.DataFrame()
        latest_intraday = pd.DataFrame()
        latest_intraday_date = None

        if not intraday.empty and "Close" in intraday.columns:
            intraday_dates = index_to_dates(intraday.index)

            today_intraday = intraday.loc[
                intraday_dates == today
            ].copy().dropna(subset=["Close"])

            valid_dates = intraday_dates.dropna()
            if not valid_dates.empty:
                latest_intraday_date = valid_dates.max()
                latest_intraday = intraday.loc[
                    intraday_dates == latest_intraday_date
                ].copy().dropna(subset=["Close"])

        valid_today_intraday = not today_intraday.empty

        # During NSE hours, current-day intraday data is mandatory.
        live_session = (
            is_nse_session_now(now_ist)
            and valid_today_intraday
        )

        # Completed daily history: exclude today's incomplete daily candle.
        daily_dates = index_to_dates(daily.index)
        historical = daily.loc[daily_dates < today].copy()
        historical = historical.dropna(subset=["Close"])

        if historical.empty:
            historical = daily.copy().dropna(subset=["Close"])
        if historical.empty:
            return None, "no completed daily history"

        previous_close = float(historical["Close"].iloc[-1])

        # Current price:
        # - market open + today's intraday => latest today's bar
        # - market closed => latest session returned by intraday feed
        # - market open without today's intraday => unavailable, never stale
        if live_session:
            prices = today_intraday["Close"].dropna()
            current_price = float(prices.iloc[-1])
            price_date = today
        elif not is_nse_session_now(now_ist) and not latest_intraday.empty:
            prices = latest_intraday["Close"].dropna()
            current_price = float(prices.iloc[-1])
            price_date = latest_intraday_date
        elif is_nse_session_now(now_ist):
            return None, "today intraday price unavailable"
        else:
            current_price = previous_close
            price_date = get_index_date(historical.index[-1])

        if not np.isfinite(current_price) or current_price <= 0:
            return None, "invalid current price"

        # 1D return: current price vs previous completed close.
        one_day_return = (
            (current_price / previous_close) - 1.0
        ) * 100.0 if previous_close > 0 else np.nan

        # 1W return: current price vs five completed trading sessions ago.
        if len(historical) >= 5:
            week_base = float(historical["Close"].iloc[-5])
            one_week_return = (
                (current_price / week_base) - 1.0
            ) * 100.0 if week_base > 0 else np.nan
        else:
            one_week_return = np.nan

        # 1M return: current price vs same/previous trading date one month ago.
        month_base = previous_month_same_or_previous_trading_close(
            historical, price_date
        )
        one_month_return = (
            (current_price / month_base) - 1.0
        ) * 100.0 if np.isfinite(month_base) and month_base > 0 else np.nan

        # EMA: completed daily closes + current price.
        ema_series = historical["Close"].copy()
        ema_series.loc[pd.Timestamp(now_ist)] = current_price

        ema21 = float(
            ema_series.ewm(span=21, adjust=False).mean().iloc[-1]
        )
        ema50 = float(
            ema_series.ewm(span=50, adjust=False).mean().iloc[-1]
        )
        ema200 = float(
            ema_series.ewm(span=200, adjust=False).mean().iloc[-1]
        )

        # 52W High/Low: daily High/Low + today's intraday High/Low when available.
        cutoff_date = (
            pd.Timestamp(price_date) - pd.Timedelta(days=365)
        ).date()
        hist_dates = index_to_dates(historical.index)
        last_52w = historical.loc[hist_dates >= cutoff_date].copy()
        if last_52w.empty:
            last_52w = historical.copy()

        week52_high = np.nan
        week52_low = np.nan

        if "High" in last_52w.columns:
            highs = pd.to_numeric(
                last_52w["High"], errors="coerce"
            ).dropna()
            if not highs.empty:
                week52_high = float(highs.max())

        if "Low" in last_52w.columns:
            lows = pd.to_numeric(
                last_52w["Low"], errors="coerce"
            ).dropna()
            if not lows.empty:
                week52_low = float(lows.min())

        if valid_today_intraday:
            if "High" in today_intraday.columns:
                highs = pd.to_numeric(
                    today_intraday["High"], errors="coerce"
                ).dropna()
                if not highs.empty:
                    today_high = float(highs.max())
                    week52_high = (
                        today_high if not np.isfinite(week52_high)
                        else max(week52_high, today_high)
                    )

            if "Low" in today_intraday.columns:
                lows = pd.to_numeric(
                    today_intraday["Low"], errors="coerce"
                ).dropna()
                if not lows.empty:
                    today_low = float(lows.min())
                    week52_low = (
                        today_low if not np.isfinite(week52_low)
                        else min(week52_low, today_low)
                    )

        from_52w_high = (
            (current_price / week52_high - 1.0) * 100.0
            if np.isfinite(week52_high) and week52_high > 0 else np.nan
        )
        from_52w_low = (
            (current_price / week52_low - 1.0) * 100.0
            if np.isfinite(week52_low) and week52_low > 0 else np.nan
        )
        from_21_ema = (
            (current_price / ema21 - 1.0) * 100.0
            if np.isfinite(ema21) and ema21 > 0 else np.nan
        )

        if current_price > ema21 and ema21 > ema50 and ema50 > ema200:
            trend = "Bullish"
        elif current_price < ema21 and ema21 < ema50 and ema50 < ema200:
            trend = "Bearish"
        else:
            trend = "Neutral"

        return {
            "Stock": symbol,
            "Price": current_price,
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
        }, None

    except Exception as exc:
        return None, str(exc)

def get_index_date(index_value):
    try:
        timestamp = pd.Timestamp(index_value)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert(IST)
        return timestamp.date()
    except Exception:
        return None

# ============================================================
# TREND COLOUR
# ============================================================

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

if refresh_clicked:
    get_intraday.clear()
    st.rerun()

# ============================================================
# LIVE SCAN / AUTO REFRESH
# ============================================================

@st.fragment(run_every="30s")
def live_scan():
    if not st.session_state.scan_started:
        return

    now_ist = datetime.now(IST)

    try:
        symbols = get_nifty100_list()
        st.info(
            f"Current Nifty 100 list: {len(symbols)} stocks"
        )
    except Exception as e:
        st.error("Unable to get the current Nifty 100 list.")
        st.error(str(e))
        st.stop()

    with st.spinner("Downloading daily and latest intraday data..."):
        try:
            daily_data = get_daily(tuple(symbols))
            intraday_data = get_intraday(tuple(symbols))
        except Exception as e:
            st.error("Unable to download stock data.")
            st.error(str(e))
            st.stop()

    results = []
    unavailable = []
    progress = st.progress(0)
    total = len(symbols)

    for i, symbol in enumerate(symbols):
        result, reason = calculate_stock_data(
            symbol,
            daily_data,
            intraday_data,
            now_ist
        )

        if result is not None:
            results.append(result)
        else:
            # Keep every Nifty 100 constituent in the table so the
            # screener doesn't unexpectedly become 99/98/etc. Metrics
            # are blank where Yahoo has insufficient history/data.
            results.append({
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
            unavailable.append(f"{symbol} ({reason})")

        progress.progress(
            int(((i + 1) / total) * 100)
        )

    progress.empty()

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

    df = pd.DataFrame(results)[columns]

    df = df.sort_values(
        by="From 21 EMA %",
        ascending=False,
        na_position="last"
    ).reset_index(drop=True)

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
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        ).round(2)

    updated_time = now_ist.strftime(
        "%d-%m-%Y %I:%M:%S %p IST"
    )

    st.success(
        f"🕐 Last updated: {updated_time}"
    )

    st.info(
        f"Nifty 100: {len(symbols)} stocks | "
        f"Displayed: {len(df)} stocks"
    )

    if unavailable:
        st.warning(
            "Data unavailable for: "
            + ", ".join(unavailable)
        )

    st.subheader(
        f"📋 Results — {len(df)} stocks"
    )

    display_df = df.copy()

    styled_df = (
        display_df.style
        .map(
            colour_trend,
            subset=["Trend"]
        )
        .format({
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
        }, na_rep="—")
    )

    st.dataframe(
        styled_df,
        use_container_width=True,
        height=650,
        hide_index=True
    )

    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Results CSV",
        data=csv_data,
        file_name="nifty100_screener.csv",
        mime="text/csv"
    )

live_scan()
