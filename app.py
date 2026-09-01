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
    page_title="My Stock Screener",
    page_icon="📊",
    layout="wide"
)

st.title("📊 My Stock Screener")
st.subheader("Nifty 100 Technical Screener")
st.caption(
    "Latest available intraday price is used for calculations. "
    "Yahoo Finance data may be delayed during market hours."
)

NIFTY100_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv"


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
    rows = list(csv.reader(io.StringIO(text)))

    if len(rows) < 2:
        raise ValueError("Nifty 100 CSV returned no data.")

    header = [str(x).strip() for x in rows[0]]
    symbol_index = None

    for i, col in enumerate(header):
        if col.lower() == "symbol":
            symbol_index = i
            break

    if symbol_index is None:
        raise ValueError("Symbol column not found in Nifty 100 CSV.")

    symbols = []
    for row in rows[1:]:
        if len(row) > symbol_index:
            symbol = row[symbol_index].strip()
            if symbol:
                symbols.append(symbol)

    symbols = list(dict.fromkeys(symbols))

    if len(symbols) < 80:
        raise ValueError(
            f"Only {len(symbols)} Nifty 100 stocks were found."
        )

    return symbols


@st.cache_data(ttl=21600, show_spinner=False)
def get_daily_data(symbols):
    tickers = [symbol + ".NS" for symbol in symbols]
    return yf.download(
        tickers=tickers,
        period="2y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=False,
        prepost=False,
        ignore_tz=False
    )


@st.cache_data(ttl=20, show_spinner=False)
def get_intraday_data(symbols):
    tickers = [symbol + ".NS" for symbol in symbols]
    return yf.download(
        tickers=tickers,
        period="5d",
        interval="5m",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=False,
        prepost=False,
        ignore_tz=False
    )


def get_ticker_close(data, ticker):
    if data is None or data.empty:
        return pd.Series(dtype="float64")

    try:
        if isinstance(data.columns, pd.MultiIndex):
            levels = [
                list(data.columns.get_level_values(i).unique())
                for i in range(data.columns.nlevels)
            ]

            # Normal yfinance layouts: (Ticker, Price) or (Price, Ticker)
            for level_no in range(data.columns.nlevels):
                if ticker in levels[level_no]:
                    temp = data.xs(ticker, axis=1, level=level_no)
                    if isinstance(temp.columns, pd.MultiIndex):
                        # Flatten remaining levels and find Close.
                        close_cols = [
                            col for col in temp.columns
                            if "Close" in tuple(str(x) for x in col)
                        ]
                        if close_cols:
                            s = temp[close_cols[0]]
                        else:
                            continue
                    elif "Close" in temp.columns:
                        s = temp["Close"]
                    else:
                        continue

                    s = pd.to_numeric(s, errors="coerce").dropna()
                    if not s.empty:
                        return s

        elif "Close" in data.columns:
            s = pd.to_numeric(data["Close"], errors="coerce").dropna()
            if not s.empty:
                return s

    except Exception:
        pass

    return pd.Series(dtype="float64")


def get_index_date(index_value):
    try:
        timestamp = pd.Timestamp(index_value)

        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert("Asia/Kolkata")

        return timestamp.date()
    except Exception:
        return None


def calculate_stock_data(symbol, daily_data, intraday_data):
    ticker = symbol + ".NS"

    try:
        daily_close = get_ticker_close(daily_data, ticker)
        if len(daily_close) < 220:
            return None, "insufficient historical daily data"

        ist = ZoneInfo("Asia/Kolkata")
        now = datetime.now(ist)
        today = now.date()

        historical_close = daily_close.copy()
        last_daily_date = get_index_date(historical_close.index[-1])

        if last_daily_date == today:
            historical_close = historical_close.iloc[:-1]

        if len(historical_close) < 220:
            return None, "not enough completed daily history"

        previous_close = float(historical_close.iloc[-1])
        if previous_close <= 0:
            return None, "invalid previous close"

        intraday_close = get_ticker_close(intraday_data, ticker)

        # Convert intraday timestamps to IST dates and keep today's bars.
        today_intraday = pd.Series(dtype="float64")
        if not intraday_close.empty:
            dates = []
            for idx in intraday_close.index:
                try:
                    ts = pd.Timestamp(idx)
                    if ts.tzinfo is not None:
                        ts = ts.tz_convert(ist)
                    dates.append(ts.date())
                except Exception:
                    dates.append(None)

            date_series = pd.Series(dates, index=intraday_close.index)
            today_intraday = intraday_close.loc[
                date_series == today
            ].dropna()

        market_open = time(9, 15)
        market_close = time(15, 30)
        market_is_open = (
            now.weekday() < 5
            and market_open <= now.time() <= market_close
        )

        if market_is_open:
            if today_intraday.empty:
                # Never pretend an old close is a live price.
                return None, "today intraday price unavailable"
            live_price = float(today_intraday.iloc[-1])
            price_status = "LIVE"
        else:
            # Outside market hours, the latest completed daily close is the
            # correct reference price.
            live_price = previous_close
            price_status = "CLOSED"

        if not np.isfinite(live_price) or live_price <= 0:
            return None, "invalid price"

        one_day_return = (live_price / previous_close - 1) * 100

        one_week_base = float(historical_close.iloc[-6])
        one_week_return = (live_price / one_week_base - 1) * 100

        one_month_base = float(historical_close.iloc[-22])
        one_month_return = (live_price / one_month_base - 1) * 100

        calc_close = pd.concat([
            historical_close,
            pd.Series([live_price], index=[pd.Timestamp(now)])
        ])

        ema21 = float(calc_close.ewm(span=21, adjust=False).mean().iloc[-1])
        ema50 = float(calc_close.ewm(span=50, adjust=False).mean().iloc[-1])
        ema200 = float(calc_close.ewm(span=200, adjust=False).mean().iloc[-1])

        # Use daily high/low history for 52W. Include today's intraday range
        # only when today's data actually exists.
        hist_ohlc = None
        if isinstance(daily_data.columns, pd.MultiIndex):
            try:
                temp = daily_data[ticker] if ticker in daily_data.columns.get_level_values(0) else daily_data.xs(ticker, axis=1, level=1)
                hist_ohlc = temp
            except Exception:
                pass

        if hist_ohlc is not None and "High" in hist_ohlc.columns and "Low" in hist_ohlc.columns:
            hist_dates = pd.Series(
                [get_index_date(x) for x in hist_ohlc.index],
                index=hist_ohlc.index
            )
            hist_52 = hist_ohlc.loc[hist_dates >= (today - pd.Timedelta(days=365))]
            highs = pd.to_numeric(hist_52["High"], errors="coerce").dropna()
            lows = pd.to_numeric(hist_52["Low"], errors="coerce").dropna()
            week52_high = float(highs.max()) if not highs.empty else float(calc_close.tail(252).max())
            week52_low = float(lows.min()) if not lows.empty else float(calc_close.tail(252).min())
        else:
            week52_high = float(calc_close.tail(252).max())
            week52_low = float(calc_close.tail(252).min())

        if not today_intraday.empty:
            # intraday_close contains close only; current price itself is at
            # least included in the 52W range.
            week52_high = max(week52_high, live_price)
            week52_low = min(week52_low, live_price)

        from_52w_high = (live_price / week52_high - 1) * 100
        from_52w_low = (live_price / week52_low - 1) * 100
        from_21_ema = (live_price / ema21 - 1) * 100

        if live_price > ema21 and ema21 > ema50 and ema50 > ema200:
            trend = "Bullish"
        elif live_price < ema21 and ema21 < ema50 and ema50 < ema200:
            trend = "Bearish"
        else:
            trend = "Neutral"

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
        return None, str(e)


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


if "scan_started" not in st.session_state:
    st.session_state.scan_started = False

scan_clicked = st.button("🔍 Scan Nifty 100")
if scan_clicked:
    st.session_state.scan_started = True

@st.fragment(run_every='30s')
def render_scan():
    if not st.session_state.scan_started:
        return
    try:
        symbols = get_nifty100_list()
        st.info(
            f"Current Nifty 100 list: {len(symbols)} stocks"
        )
    except Exception as e:
        st.error("Unable to get the current Nifty 100 list.")
        st.error(str(e))
        st.stop()

    with st.spinner(
        "Downloading daily and latest intraday data..."
    ):
        try:
            daily_data = get_daily_data(symbols)
            intraday_data = get_intraday_data(symbols)
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
            intraday_data
        )

        if result is not None:
            results.append(result)
        else:
            unavailable.append(
                f"{symbol} ({reason})"
            )

        progress.progress(
            int(((i + 1) / total) * 100)
        )

    progress.empty()

    if not results:
        st.error("No stock data could be calculated.")
        st.stop()

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

    df = pd.DataFrame(results)
    df = df[columns]

    df = df.sort_values(
        by="From 21 EMA %",
        ascending=False
    ).reset_index(drop=True)

    ist = ZoneInfo("Asia/Kolkata")
    updated_time = datetime.now(ist).strftime(
        "%d-%m-%Y %I:%M:%S %p IST"
    )

    st.success(
        f"🕐 Last updated: {updated_time}"
    )
    now_check = datetime.now(ZoneInfo("Asia/Kolkata"))
    market_is_open_now = (
        now_check.weekday() < 5
        and time(9, 15) <= now_check.time() <= time(15, 30)
    )
    if market_is_open_now:
        st.info("🟢 Market open: current-day intraday prices are used.")
    else:
        st.info("🔵 Market closed: latest completed daily close is used.")

    st.info(
        f"Nifty 100: {len(symbols)} stocks | "
        f"Calculated: {len(df)} stocks"
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
        display_df[col] = pd.to_numeric(
            display_df[col],
            errors="coerce"
        ).round(2)

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
        })
    )

    st.dataframe(
        styled_df,
        use_container_width=True,
        height=650,
        hide_index=True
    )

    csv_data = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Results CSV",
        data=csv_data,
        file_name="nifty100_screener.csv",
        mime="text/csv"
    )

render_scan()
