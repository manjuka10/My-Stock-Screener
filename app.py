import streamlit as st
import pandas as pd
import yfinance as yf

st.set_page_config(
    page_title="My Stock Screener",
    layout="wide"
)

st.title("📊 My Stock Screener")
st.write("Nifty 100 Stock Screener")

# Stocks for initial testing
stocks = [
    "RELIANCE.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "BHARTIARTL.NS",
    "INFY.NS",
    "TCS.NS",
    "ITC.NS",
    "SBIN.NS",
    "LT.NS",
    "AXISBANK.NS"
]

# Sidebar filters
st.sidebar.header("Filters")

min_weekly_return = st.sidebar.number_input(
    "Minimum 1 Week Return (%)",
    value=10.0
)

min_monthly_return = st.sidebar.number_input(
    "Minimum 1 Month Return (%)",
    value=10.0
)

# Scan button
if st.button("🔍 Scan Stocks"):

    results = []

    with st.spinner("Fetching market data..."):

        for ticker in stocks:

            try:
                data = yf.download(
                    ticker,
                    period="1y",
                    interval="1d",
                    auto_adjust=True,
                    progress=False
                )

                if data.empty:
                    continue

                close = data["Close"].squeeze()

                price = float(close.iloc[-1])

                weekly_return = (
                    price / float(close.iloc[-6]) - 1
                ) * 100

                monthly_return = (
                    price / float(close.iloc[-22]) - 1
                ) * 100

                ema21 = float(
                    close.ewm(span=21).mean().iloc[-1]
                )

                ema50 = float(
                    close.ewm(span=50).mean().iloc[-1]
                )

                ema200 = float(
                    close.ewm(span=200).mean().iloc[-1]
                )

                # Trend classification
                if price > ema50 and ema50 > ema200:
                    trend = "🟢 Bullish"

                elif price < ema50 and ema50 < ema200:
                    trend = "🔴 Bearish"

                else:
                    trend = "🟡 Neutral"

                results.append({
                    "Stock": ticker.replace(".NS", ""),
                    "Price": round(price, 2),
                    "1W Return %": round(weekly_return, 2),
                    "1M Return %": round(monthly_return, 2),
                    "21 EMA": round(ema21, 2),
                    "50 EMA": round(ema50, 2),
                    "200 EMA": round(ema200, 2),
                    "Trend": trend
                })

            except Exception:
                pass

    df = pd.DataFrame(results)

    if not df.empty:

        # Apply filters
        df = df[
            (df["1W Return %"] >= min_weekly_return)
            &
            (df["1M Return %"] >= min_monthly_return)
        ]

        # Sort by weekly return
        df = df.sort_values(
            "1W Return %",
            ascending=False
        )

        st.subheader("Screening Results")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "No stocks matched the selected filters."
)
