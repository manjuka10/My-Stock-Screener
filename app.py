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

scan_clicked = st.button("🔍 Scan Nifty 100", type="primary")
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

@st.cache_data(ttl=45)
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

        # ----------------------------------------------------
        # Determine whether intraday data is genuinely today's
        # data. Yahoo may return the last session on holidays/
        # weekends, so date validation is essential.
        # ----------------------------------------------------
        valid_today_intraday = False
        if not intraday.empty:
            intraday_dates = index_to_dates(intraday.index)
            valid_today_intraday = bool(
                (intraday_dates == today).any()
            )

        live_session = (
            is_nse_session_now(now_ist)
            and valid_today_intraday
        )

        # ----------------------------------------------------
        # Completed daily history
        # ----------------------------------------------------
        daily_dates = index_to_dates(daily.index)
        historical = daily.loc[daily_dates < today].copy()
        historical = historical.dropna(subset=["Close"])

        # On a normal trading day, today's daily candle may be
        # incomplete. On a holiday, the latest daily row is the
        # previous completed session and must be retained.
        if historical.empty:
            historical = daily.copy()

        historical = historical.dropna(subset=["Close"])

        if historical.empty:
            return None, "no completed daily history"

        previous_close = float(historical["Close"].iloc[-1])

        # ----------------------------------------------------
        # CURRENT PRICE
        # ----------------------------------------------------
        if live_session:
            today_intraday = intraday.loc[
                index_to_dates(intraday.index) == today
            ].copy()
            prices = today_intraday["Close"].dropna()

            if not prices.empty:
                current_price = float(prices.iloc[-1])
            else:
                current_price = previous_close
                live_session = False
        else:
            # Market closed / holiday / stale intraday data:
            # use the latest completed daily close.
            current_price = previous_close

        if not np.isfinite(current_price) or current_price <= 0:
            return None, "invalid current price"

        # ----------------------------------------------------
        # 1D RETURN
        # Live session: live price vs previous completed close.
        # Closed session: latest completed close vs the prior
        # completed trading-day close. This preserves the latest
        # actual 1D return on weekends/holidays instead of showing
        # 0.00% for every stock.
        # ----------------------------------------------------
        if live_session:
            if previous_close > 0:
                one_day_return = (
                    current_price / previous_close - 1.0
                ) * 100.0
            else:
                one_day_return = np.nan
        else:
            if len(historical) >= 2:
                prior_close = float(historical["Close"].iloc[-2])
                if prior_close > 0:
                    one_day_return = (
                        previous_close / prior_close - 1.0
                    ) * 100.0
                else:
                    one_day_return = np.nan
            else:
                one_day_return = np.nan

        # ----------------------------------------------------
        # 1W RETURN
        # Same-date previous week, falling back to the previous
        # trading day on/before the target date.
        # ----------------------------------------------------
        reference_date = today if live_session else historical.index[-1]
        reference_date = get_index_date(reference_date)

        if reference_date is not None:
            week_target = (
                pd.Timestamp(reference_date)
                - pd.Timedelta(days=7)
            ).date()
            week_base = latest_close_on_or_before(
                historical,
                week_target
            )
        else:
            week_base = np.nan

        if np.isfinite(week_base) and week_base > 0:
            one_week_return = (
                current_price / week_base - 1.0
            ) * 100.0
        else:
            one_week_return = np.nan

        # ----------------------------------------------------
        # 1M RETURN
        # Same calendar date previous month, falling back to
        # previous trading day on/before that target date.
        # ----------------------------------------------------
        if reference_date is not None:
            month_target = (
                pd.Timestamp(reference_date)
                - pd.DateOffset(months=1)
            ).date()
            month_base = latest_close_on_or_before(
                historical,
                month_target
            )
        else:
            month_base = np.nan

        if np.isfinite(month_base) and month_base > 0:
            one_month_return = (
                current_price / month_base - 1.0
            ) * 100.0
        else:
            one_month_return = np.nan

        # ----------------------------------------------------
        # EMA
        # During a live session: completed daily closes + live
        # current price as today's observation.
        # When closed: completed daily closes only.
        # ----------------------------------------------------
        calc_close = historical["Close"].copy()

        if live_session:
            calc_close = pd.concat([
                calc_close,
                pd.Series(
                    [current_price],
                    index=[pd.Timestamp(now_ist)]
                )
            ])

        # Need enough data for a meaningful 200 EMA.
        if len(calc_close) < 200:
            return None, "insufficient historical data for 200 EMA"

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

        # ----------------------------------------------------
        # 52 WEEK HIGH / LOW
        # Actual High / Low, not closing prices.
        # Today's intraday High/Low is included only when it is
        # actually today's trading data.
        # ----------------------------------------------------
        cutoff = (
            pd.Timestamp(reference_date)
            - pd.Timedelta(days=365)
        ).date()

        hist_dates = index_to_dates(historical.index)
        last_52w = historical.loc[
            hist_dates >= cutoff
        ].copy()

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

        week52_high = historical_high
        week52_low = historical_low

        if live_session and not intraday.empty:
            today_intraday = intraday.loc[
                index_to_dates(intraday.index) == today
            ].copy()

            if "High" in today_intraday.columns:
                highs = pd.to_numeric(
                    today_intraday["High"],
                    errors="coerce"
                ).dropna()
                if not highs.empty:
                    today_high = float(highs.max())
                    if np.isfinite(week52_high):
                        week52_high = max(
                            week52_high,
                            today_high
                        )
                    else:
                        week52_high = today_high

            if "Low" in today_intraday.columns:
                lows = pd.to_numeric(
                    today_intraday["Low"],
                    errors="coerce"
                ).dropna()
                if not lows.empty:
                    today_low = float(lows.min())
                    if np.isfinite(week52_low):
                        week52_low = min(
                            week52_low,
                            today_low
                        )
                    else:
                        week52_low = today_low

        # ----------------------------------------------------
        # DISTANCES
        # ----------------------------------------------------
        from_52w_high = (
            (current_price / week52_high - 1.0) * 100.0
            if np.isfinite(week52_high) and week52_high > 0
            else np.nan
        )

        from_52w_low = (
            (current_price / week52_low - 1.0) * 100.0
            if np.isfinite(week52_low) and week52_low > 0
            else np.nan
        )

        from_21_ema = (
            (current_price / ema21 - 1.0) * 100.0
            if np.isfinite(ema21) and ema21 > 0
            else np.nan
        )

        # ----------------------------------------------------
        # TREND
        # ----------------------------------------------------
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

# ============================================================
# LIVE SCAN / AUTO REFRESH
# ============================================================

@st.fragment(run_every="60s")
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
