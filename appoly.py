import streamlit as st
import pandas as pd
import ta
from polygon import RESTClient
from datetime import datetime

# Configuration de l'API Polygon
API_KEY = "VOTRE_CLE_API_POLYGON"
client = RESTClient(API_KEY)

# Liste des paires Forex supportées par Polygon (format C:xxx)
FOREX_PAIRS = [
    'C:EURUSD', 'C:GBPUSD', 'C:USDJPY', 'C:USDCHF',
    'C:AUDUSD', 'C:USDCAD', 'C:NZDUSD', 'C:EURJPY'
]

# Fonction de récupération des données depuis Polygon
@st.cache_data(ttl=300)
def get_data_polygon(symbol, multiplier=1, timespan='hour', from_date='2024-05-01', to_date='2024-05-29'):
    try:
        bars = client.get_aggs(
            symbol=symbol,
            multiplier=multiplier,
            timespan=timespan,
            from_=from_date,
            to=to_date,
            limit=500
        )
        df = pd.DataFrame([{
            'Open': bar.o,
            'High': bar.h,
            'Low': bar.l,
            'Close': bar.c,
            'Volume': bar.v,
            'Datetime': pd.to_datetime(bar.t, unit='ms')
        } for bar in bars])

        df.set_index('Datetime', inplace=True)
        return df
    except Exception as e:
        st.error(f"Erreur récupération Polygon pour {symbol} : {e}")
        return None

# Fonction pour calculer la confluence
def calculate_confluence(df):
    df['HMA'] = ta.trend.hull_moving_average(df['Close'], window=12)
    df['EMA'] = ta.trend.ema_indicator(df['Close'], window=20)
    df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df.dropna(inplace=True)

    confluence = []
    for i in range(len(df)):
        signal = 0
        if df['HMA'].iloc[i] > df['EMA'].iloc[i]:
            signal += 1
        if df['ADX'].iloc[i] > 20:
            signal += 1
        if df['RSI'].iloc[i] > 50:
            signal += 1

        if signal == 3:
            confluence.append('⭐⭐⭐⭐⭐')
        elif signal == 2:
            confluence.append('⭐⭐⭐⭐')
        elif signal == 1:
            confluence.append('⭐⭐⭐')
        else:
            confluence.append('—')

    df['Confluence'] = confluence
    return df[['Close', 'Confluence']]

# Interface Streamlit
st.title("Scanner Forex - Confluence 5⭐ et 6⭐ avec Polygon.io")

selected_pairs = st.multiselect("Choisissez les paires Forex à analyser :", FOREX_PAIRS, default=FOREX_PAIRS[:4])

if selected_pairs:
    for pair in selected_pairs:
        df = get_data_polygon(pair)
        if df is not None:
            df_result = calculate_confluence(df)
            latest_signal = df_result.iloc[-1]['Confluence']
            st.write(f"**{pair}** : Dernier signal = {latest_signal}")
            st.line_chart(df_result['Close'])
            st.dataframe(df_result.tail(5))
