import streamlit as st
import pandas as pd
import yfinance as yf

st.set_page_config(
    page_title="My Stock Screener",
    page_icon="📊",
    layout="wide"
)

st.title("📊 My Stock Screener")
st.caption("Nifty 100 Technical Screener")


# ==================================================
# NIFTY 100 STOCKS
# ==================================================

stocks = [
    "RELIANCE", "BHARTIARTL", "HDFCBANK", "ICICIBANK",
    "SBIN", "TCS", "BAJFINANCE", "LT", "HINDUNILVR",
    "SUNPHARMA", "INFY", "TITAN", "MARUTI", "ADANIENT",
    "M&M", "ITC", "AXISBANK", "KOTAKBANK", "NTPC",
    "TATASTEEL", "BEL", "SHRIRAMFIN", "ULTRACEMCO",
    "HCLTECH", "POWERGRID", "HINDALCO", "JSWSTEEL",
    "ONGC", "COALINDIA", "ADANIPORTS", "BAJAJFINSV",
    "ETERNAL", "NESTLEIND", "TECHM", "TRENT",
    "WIPRO", "CIPLA", "ADANIGREEN", "TATAMOTORS",
    "DRREDDY", "APOLLOHOSP", "DIVISLAB", "BPCL",
    "GRASIM", "EICHERMOT", "TATACONSUM", "BRITANNIA",
    "HEROMOTOCO", "HINDPETRO", "INDUSINDBK", "SHREECEM",
    "IOC", "BAJAJ-AUTO", "SBILIFE", "HDFCLIFE",
    "VEDL", "DLF", "ICICIPRULI", "PIDILITIND",
    "HAL", "INDHOTEL", "LODHA", "MOTHERSON",
    "MAXHEALTH", "TVSMOTOR", "SIEMENS", "ABB",
    "DABUR", "AMBUJACEM", "ACC", "BANKBARODA",
    "PNB", "CANBK", "UNIONBANK", "RECLTD",
    "PFC", "IRCTC", "JINDALSTEL", "JSWENERGY",
    "TORNTPHARM", "ZYDUSLIFE", "FORTIS", "BOSCHLTD",
    "COLPAL", "MARICO", "GODREJCP", "VBL",
    "UNITDSPR", "SOLARINDS", "POLYCAB", "CUMMINSIND",
    "CGPOWER", "BHEL", "NHPC", "TATAPOWER",
    "GAIL", "INDUSTOWER", "LICI", "PAYTM"
]


# ==================================================
# SIDEBAR FILTERS
# ==================================================

st.sidebar.header("🔍 Screener Filters")

min_weekly = st.sidebar.number_input(
    "Minimum 1 Week Return (%)",
    value=0.0,
    step=1.0
)

min_monthly = st.sidebar.number_input(
    "Minimum 1 Month Return (%)",
    value=0.0,
    step=1.0
)

max_volatility = st.sidebar.number_input(
    "Maximum Volatility (%)",
    value=100.0,
    step=5.0
)

min_distance_high = st.sidebar.number_input(
    "Minimum Distance From 52W High (%)",
    value=-100.0,
    step=1.0
)

trend_filter = st.sidebar.selectbox(
    "Trend",
    [
        "All",
        "Bullish",
        "Neutral",
        "Bearish"
    ]
)


# ==================================================
# SCAN BUTTON
# ==================================================

if st.button("🔍 Scan Nifty 100", type="primary"):

    results = []

    progress = st.progress(0)

    for i, symbol in enumerate(stocks):

        try:

            ticker = symbol + ".NS"

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

            if len(close) < 200:
                continue

            # ------------------------------------------
            # CURRENT PRICE
            # ------------------------------------------

            price = float(close.iloc[-1])


            # ------------------------------------------
            # RETURNS
            # ------------------------------------------

            weekly_return = (
                price / float(close.iloc[-6]) - 1
            ) * 100

            monthly_return = (
                price / float(close.iloc[-22]) - 1
            ) * 100


            # ------------------------------------------
            # EMA
            # ------------------------------------------

            ema21 = float(
                close.ewm(
                    span=21,
                    adjust=False
                ).mean().iloc[-1]
            )

            ema50 = float(
                close.ewm(
                    span=50,
                    adjust=False
                ).mean().iloc[-1]
            )

            ema200 = float(
                close.ewm(
                    span=200,
                    adjust=False
                ).mean().iloc[-1]
            )


            # ------------------------------------------
            # VOLATILITY
            # ------------------------------------------

            daily_returns = close.pct_change().dropna()

            volatility = (
                daily_returns.std()
                * (252 ** 0.5)
            ) * 100


            # ------------------------------------------
            # 52 WEEK HIGH / LOW
            # ------------------------------------------

            high_52w = float(close.max())

            low_52w = float(close.min())


            # ------------------------------------------
            # DISTANCE FROM 52W HIGH
            # ------------------------------------------

            distance_high = (
                (price / high_52w) - 1
            ) * 100


            # ------------------------------------------
            # DISTANCE FROM 52W LOW
            # ------------------------------------------

            distance_low = (
                (price / low_52w) - 1
            ) * 100


            # ------------------------------------------
            # DISTANCE FROM 21 EMA
            # ------------------------------------------

            distance_21ema = (
                (price / ema21) - 1
            ) * 100


            # ------------------------------------------
            # TREND CLASSIFICATION
            # ------------------------------------------

            if (
                price > ema21
                and ema21 > ema50
                and ema50 > ema200
            ):

                trend = "Bullish"

            elif (
                price < ema21
                and ema21 < ema50
                and ema50 < ema200
            ):

                trend = "Bearish"

            else:

                trend = "Neutral"


            # ------------------------------------------
            # STORE RESULTS
            # ------------------------------------------

            results.append({

                "Stock": symbol,

                "Price": round(
                    price, 2
                ),

                "1W Return %": round(
                    weekly_return, 2
                ),

                "1M Return %": round(
                    monthly_return, 2
                ),

                "21 EMA": round(
                    ema21, 2
                ),

                "50 EMA": round(
                    ema50, 2
                ),

                "200 EMA": round(
                    ema200, 2
                ),

                "Volatility %": round(
                    volatility, 2
                ),

                "52W High": round(
                    high_52w, 2
                ),

                "52W Low": round(
                    low_52w, 2
                ),

                "From 52W High %": round(
                    distance_high, 2
                ),

                "From 52W Low %": round(
                    distance_low, 2
                ),

                "From 21 EMA %": round(
                    distance_21ema, 2
                ),

                "Trend": trend
            })


        except Exception:
            continue


        progress.progress(
            (i + 1) / len(stocks)
        )


    progress.empty()


    # ==================================================
    # RESULTS
    # ==================================================

    df = pd.DataFrame(results)


    if not df.empty:

        # ----------------------------------------------
        # APPLY FILTERS
        # ----------------------------------------------

        df = df[
            (df["1W Return %"] >= min_weekly)
            &
            (df["1M Return %"] >= min_monthly)
            &
            (df["Volatility %"] <= max_volatility)
            &
            (df["From 52W High %"] >= min_distance_high)
        ]


        # ----------------------------------------------
        # TREND FILTER
        # ----------------------------------------------

        if trend_filter != "All":

            df = df[
                df["Trend"] == trend_filter
            ]


        # ----------------------------------------------
        # SORT
        # ----------------------------------------------

        df = df.sort_values(
            "1W Return %",
            ascending=False
        )


        # ----------------------------------------------
        # DISPLAY
        # ----------------------------------------------

        st.subheader(
            f"📋 Results — {len(df)} stocks"
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )


    else:

        st.warning(
            "No stocks matched the selected filters."
        )


else:

    st.info(
        "Set your filters on the left and tap "
        "'Scan Nifty 100'."
)
