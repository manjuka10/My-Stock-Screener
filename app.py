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
    "Latest available intraday price is used for calculations. "
    "Yahoo Finance data may be delayed during market hours."
)


# =========================================================
# NIFTY 100 CONSTITUENTS
# =========================================================

NIFTY100_URL = (
    "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv"
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
# DOWNLOAD MARKET DATA
# =========================================================

@st.cache_data(ttl=300)
def download_stock_data(symbols):

    tickers = [
        symbol + ".NS"
        for symbol in symbols
    ]

    # -----------------------------------------------------
    # DAILY DATA
    # Used for historical calculations
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
    # 5 MINUTE INTRADAY DATA
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
# GET CLOSE SERIES
# =========================================================

def get_ticker_close(data, ticker):

    if data is None or data.empty:

        return pd.Series(
            dtype="float64"
        )

    try:

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

            # -------------------------------------------------
            # TICKER ON FIRST LEVEL
            # -------------------------------------------------

            if ticker in level0:

                temp = data[ticker].copy()

                if "Close" in temp.columns:

                    return pd.to_numeric(
                        temp["Close"],
                        errors="coerce"
                    ).dropna()

            # -------------------------------------------------
            # TICKER ON SECOND LEVEL
            # -------------------------------------------------

            if ticker in level1:

                temp = data.xs(
                    ticker,
                    axis=1,
                    level=1
                )

                if "Close" in temp.columns:

                    return pd.to_numeric(
                        temp["Close"],
                        errors="coerce"
                    ).dropna()

        else:

            if "Close" in data.columns:

                return pd.to_numeric(
                    data["Close"],
                    errors="coerce"
                ).dropna()

    except Exception:

        pass

    return pd.Series(
        dtype="float64"
    )


# =========================================================
# GET DATE FROM INDEX
# =========================================================

def get_index_date(index_value):

    try:

        timestamp = pd.Timestamp(
            index_value
        )

        if timestamp.tzinfo is not None:

            timestamp = timestamp.tz_convert(
                "
