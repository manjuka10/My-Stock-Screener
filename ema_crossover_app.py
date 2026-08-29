import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests, csv, io
from datetime import datetime
from zoneinfo import ZoneInfo

st.set_page_config(page_title='21/50 EMA Crossover Screener', page_icon='📈', layout='wide')
st.title('📈 21/50 EMA Crossover Screener')
st.subheader('Trend Following Screener — Nifty 100')
st.caption('EMA, crossover and slope use completed daily candles. Price uses the latest available market price.')

URL='https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv'

@st.cache_data(ttl=86400)
def symbols():
    r=requests.get(URL,headers={'User-Agent':'Mozilla/5.0','Referer':'https://www.niftyindices.com/'},timeout=30)
    r.raise_for_status(); rows=list(csv.reader(io.StringIO(r.content.decode('utf-8-sig',errors='replace'))))
    h=[str(x).strip().lower() for x in rows[0]]; i=h.index('symbol')
    return list(dict.fromkeys(row[i].strip() for row in rows[1:] if len(row)>i and row[i].strip()))

@st.cache_data(ttl=300)
def daily(syms):
    return yf.download(tickers=[s+'.NS' for s in syms],period='2y',interval='1d',auto_adjust=False,progress=False,group_by='ticker',threads=True)

@st.cache_data(ttl=120)
def intra(syms):
    return yf.download(tickers=[s+'.NS' for s in syms],period='1d',interval='5m',auto_adjust=False,progress=False,group_by='ticker',threads=True)

def ticker_data(data,ticker):
    if data is None or data.empty: return pd.DataFrame()
    try:
        if isinstance(data.columns,pd.MultiIndex):
            if ticker in data.columns.get_level_values(0): x=data[ticker].copy()
            elif ticker in data.columns.get_level_values(1): x=data.xs(ticker,axis=1,level=1).copy()
            else: return pd.DataFrame()
        else: x=data.copy()
        cols=[c for c in ['Open','High','Low','Close'] if c in x.columns]
        if 'Close' not in cols: return pd.DataFrame()
        x=x[cols].copy()
        for c in cols: x[c]=pd.to_numeric(x[c],errors='coerce')
        return x.dropna(how='all')
    except Exception: return pd.DataFrame()

def dates(index):
    out=[]
    for v in index:
        try:
            t=pd.Timestamp(v)
            if t.tzinfo is not None: t=t.tz_convert('Asia/Kolkata')
            out.append(t.date())
        except Exception: out.append(None)
    return pd.Series(out,index=index)

def pct(a,b): return ((a/b)-1)*100 if np.isfinite(a) and np.isfinite(b) and b!=0 else np.nan

def calc(sym,daily_all,intra_all,today):
    d=ticker_data(daily_all,sym+'.NS')
    if d.empty: return None
    d=d.dropna(subset=['Close']); ds=dates(d.index)
    d=d.loc[ds<today].copy().dropna(subset=['Close'])
    if len(d)<55: return None
    c=d['Close']; e21=c.ewm(span=21,adjust=False,min_periods=21).mean(); e50=c.ewm(span=50,adjust=False,min_periods=50).mean()
    v=pd.DataFrame({'Close':c,'EMA21':e21,'EMA50':e50}).dropna()
    if len(v)<5: return None
    E21=float(v.EMA21.iloc[-1]); E50=float(v.EMA50.iloc[-1]); prev=float(v.EMA21.iloc[-2])
    it=ticker_data(intra_all,sym+'.NS'); price=np.nan
    if not it.empty:
        p=it['Close'].dropna()
        if not p.empty: price=float(p.iloc[-1])
    if not np.isfinite(price) or price<=0: price=float(v.Close.iloc[-1])
    sep=pct(E21,E50); p21=pct(price,E21); p50=pct(price,E50)
    trend='Bullish' if E21>E50 else ('Bearish' if E21<E50 else 'Neutral')
    slope='Rising' if E21>prev else ('Falling' if E21<prev else 'Flat')
    cross='None'; crossdate=None; days=np.nan
    for i in range(1,len(v)):
        a,b=float(v.EMA21.iloc[i-1]),float(v.EMA50.iloc[i-1]); x,y=float(v.EMA21.iloc[i]),float(v.EMA50.iloc[i])
        if a<=b and x>y: cross='Bullish'; crossdate=pd.Timestamp(v.index[i])
        elif a>=b and x<y: cross='Bearish'; crossdate=pd.Timestamp(v.index[i])
    if crossdate is not None:
        if crossdate.tzinfo is not None: crossdate=crossdate.tz_convert('Asia/Kolkata')
        days=(today-crossdate.date()).days
    last=v.tail(2)
    bull2=len(last)==2 and (last.Close>last.EMA21).all(); bear2=len(last)==2 and (last.Close<last.EMA21).all()
    confirmed_bull=(trend=='Bullish' and slope=='Rising' and p21>=0.5 and p50>=0.5 and bull2 and sep>=1.0)
    confirmed_bear=(trend=='Bearish' and slope=='Falling' and p21<=-0.5 and p50<=-0.5 and bear2 and sep<=-1.0)
    if confirmed_bull: signal='Confirmed Bullish'
    elif confirmed_bear: signal='Confirmed Bearish'
    elif cross in ('Bullish','Bearish') and np.isfinite(days) and days<=3: signal='Fresh Crossover'
    elif (trend=='Bullish' and slope=='Rising') or (trend=='Bearish' and slope=='Falling'): signal='Waiting Confirmation'
    else: signal='No Signal'
    return {'Stock':sym,'Price (Live)':round(price,2),'21 EMA':round(E21,2),'50 EMA':round(E50,2),'EMA Separation %':round(sep,2),'Price vs 21 EMA %':round(p21,2),'Price vs 50 EMA %':round(p50,2),'Trend':trend,'Signal':signal,'Crossover Date':crossdate.strftime('%d-%m-%Y') if crossdate is not None else '-','Days Since Crossover':int(days) if np.isfinite(days) else np.nan,'21 EMA Slope':slope}

def sig_color(x):
    if x=='Confirmed Bullish': return 'background-color:#198754;color:white;font-weight:bold'
    if x=='Confirmed Bearish': return 'background-color:#dc3545;color:white;font-weight:bold'
    if x=='Fresh Crossover': return 'background-color:#0d6efd;color:white;font-weight:bold'
    if x=='Waiting Confirmation': return 'background-color:#ffc107;color:black;font-weight:bold'
    return ''

def trend_color(x):
    if x=='Bullish': return 'background-color:#d1e7dd;color:#0f5132;font-weight:bold'
    if x=='Bearish': return 'background-color:#f8d7da;color:#842029;font-weight:bold'
    return ''

if st.button('🔍 Scan Nifty 100',type='primary'):
    try: syms=symbols()
    except Exception as e: st.error('Unable to download Nifty 100 list.'); st.error(str(e)); st.stop()
    st.info(f'Nifty 100 stocks found: {len(syms)}')
    with st.spinner('Downloading daily data...'):
        try: dd=daily(syms)
        except Exception as e: st.error('Daily data download failed.'); st.error(str(e)); st.stop()
    with st.spinner('Downloading latest market prices...'):
        try: ii=intra(syms)
        except Exception: ii=pd.DataFrame(); st.warning('Intraday data unavailable. Latest traded daily price will be used.')
    now=datetime.now(ZoneInfo('Asia/Kolkata')); today=now.date(); rows=[]; bar=st.progress(0)
    for n,sym in enumerate(syms):
        try:
            r=calc(sym,dd,ii,today)
            if r is not None: rows.append(r)
        except Exception: pass
        bar.progress(int((n+1)/len(syms)*100))
    bar.empty()
    if not rows: st.error('No stock data was calculated.'); st.stop()
    df=pd.DataFrame(rows)
    st.session_state['ema_scan_df'] = df.copy()
    st.session_state['ema_scan_time'] = now
    cols=['Stock','Price (Live)','21 EMA','50 EMA','EMA Separation %','Price vs 21 EMA %','Price vs 50 EMA %','Trend','Signal','Crossover Date','Days Since Crossover','21 EMA Slope']
    df=df[cols]
st.markdown("### Trend Filter")

df = st.session_state.get(
    "ema_scan_df",
    pd.DataFrame()
).copy()

if df.empty:
    st.info("Click 🔍 Scan Nifty 100 to load the latest results.")
    st.stop()

cols = [
    "Stock",
    "Price (Live)",
    "21 EMA",
    "50 EMA",
    "EMA Separation %",
    "Price vs 21 EMA %",
    "Price vs 50 EMA %",
    "Trend",
    "Signal",
    "Crossover Date",
    "Days Since Crossover",
    "21 EMA Slope"
]

df = df[cols]

choice = st.radio(
    "Show",
    [
        "All",
        "Confirmed Bullish",
        "Waiting Confirmation",
        "Confirmed Bearish",
        "Fresh Crossover"
    ],
    horizontal=True,
    label_visibility="collapsed"
)

if choice == "All":
    show = df.copy()
else:
    show = df[
        df["Signal"] == choice
    ].copy()

show = (
    show
    .assign(
        _d=pd.to_numeric(
            show["Days Since Crossover"],
            errors="coerce"
        )
    )
    .sort_values(
        "_d",
        na_position="last"
    )
    .drop(columns="_d")
    .reset_index(drop=True)
)

for c in [
    "Price (Live)",
    "21 EMA",
    "50 EMA",
    "EMA Separation %",
    "Price vs 21 EMA %",
    "Price vs 50 EMA %"
]:
    show[c] = pd.to_numeric(
        show[c],
        errors="coerce"
    ).round(2)

a,b,c,d,e = st.columns(5)

a.metric(
    "Confirmed Bullish",
    int((df["Signal"] == "Confirmed Bullish").sum())
)

b.metric(
    "Waiting Confirmation",
    int((df["Signal"] == "Waiting Confirmation").sum())
)

c.metric(
    "Confirmed Bearish",
    int((df["Signal"] == "Confirmed Bearish").sum())
)

d.metric(
    "Fresh Crossover",
    int((df["Signal"] == "Fresh Crossover").sum())
)

e.metric(
    "Total Stocks",
    len(df)
)

with st.expander(
    "ℹ️ Confirmation rules",
    expanded=False
):
    st.markdown(
        "**Bullish:** 21 EMA > 50 EMA, 21 EMA rising, "
        "live price above both EMAs, last 2 completed daily "
        "closes above 21 EMA, and EMA separation ≥ 1%.\n\n"
        "**Bearish:** exact opposite.\n\n"
        "A fresh crossover alone is **not** treated as a "
        "confirmed entry."
    )

updated_time = st.session_state.get(
    "ema_scan_time"
)

if updated_time is None:
    updated_time = datetime.now(
        ZoneInfo("Asia/Kolkata")
    )

st.success(
    "🕐 Last updated: "
    + updated_time.strftime(
        "%d-%m-%Y %I:%M:%S %p IST"
    )
)

st.subheader(
    f"📋 EMA Crossover Results — {len(show)} stocks"
)

styled = (
    show.style
    .map(
        sig_color,
        subset=["Signal"]
    )
    .map(
        trend_color,
        subset=["Trend"]
    )
    .format(
        {
            "Price (Live)": "{:.2f}",
            "21 EMA": "{:.2f}",
            "50 EMA": "{:.2f}",
            "EMA Separation %": "{:.2f}",
            "Price vs 21 EMA %": "{:.2f}",
            "Price vs 50 EMA %": "{:.2f}",
            "Days Since Crossover": "{:.0f}"
        },
        na_rep="-"
    )
)

st.dataframe(
    styled,
    use_container_width=True,
    height=650,
    hide_index=True
)

st.download_button(
    "⬇️ Download Results CSV",
    show.to_csv(index=False).encode("utf-8"),
    "nifty100_21_50_ema_confirmation.csv",
    "text/csv"
)

st.caption(
    "EMA and crossover use completed daily candles. "
    "Price uses the latest available intraday market price. "
    "The confirmation filter is designed to reduce false "
    "crossover entries."
)
