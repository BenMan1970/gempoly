import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

# Remplacez par votre clé API Polygon.io
POLYGON_API_KEY = "YOUR_POLYGON_API_KEY_HERE"

# Liste des paires Forex supportées par Polygon.io (exemple)
FOREX_PAIRS = [
    "C:EURUSD", "C:USDJPY", "C:GBPUSD", "C:USDCHF", "C:USDCAD",
    "C:AUDUSD", "C:NZDUSD", "C:XAUUSD", "C:USDSEK", "C:USDNOK", "C:USDZAR"
]

def get_h1_data(pair, api_key):
    url = f"https://api.polygon.io/v2/aggs/ticker/{pair}/range/1/hour/{(datetime.now(timezone.utc) - timedelta(hours=250)).date()}/{datetime.now(timezone.utc).date()}?adjusted=true&sort=desc&limit=250&apiKey={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        if not results:
            return None
        df = pd.DataFrame(results)
        df["t"] = pd.to_datetime(df["t"], unit="ms")
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close"})
        df = df[["t", "open", "high", "low", "close"]]
        return df[::-1].reset_index(drop=True)
    except Exception as e:
        st.error(f"Erreur récupération {pair}: {e}")
        return None

def calculate_confluence(df):
    try:
        df["ema_20"] = df["close"].ewm(span=20).mean()
        df["ema_50"] = df["close"].ewm(span=50).mean()
        df["signal"] = 0
        df.loc[df["ema_20"] > df["ema_50"], "signal"] += 1
        df.loc[df["close"] > df["ema_20"], "signal"] += 1
        df.loc[df["close"] > df["ema_50"], "signal"] += 1
        df["confluence"] = df["signal"]
        return df.iloc[-1]["confluence"]
    except Exception as e:
        st.error(f"Erreur calcul confluence : {e}")
        return 0

# Interface utilisateur
st.set_page_config(page_title="Scanner Confluence Forex Premium (Polygon.io)", layout="wide")
st.title("🔍 Scanner Confluence Forex Premium (Polygon.io)")
st.markdown("_Version adaptée pour Polygon.io_")

st.sidebar.header("🔧 Paramètres")
min_conf = st.sidebar.selectbox("Confluence minimum (0-6)", list(range(7)), index=3)
show_all = st.sidebar.checkbox("Voir toutes les paires (ignorer filtre confluence)", value=False)

results = []

with st.spinner("🔄 Scan en cours..."):
    for pair in FOREX_PAIRS:
        df = get_h1_data(pair, POLYGON_API_KEY)
        if df is None:
            continue
        score = calculate_confluence(df)
        if score >= min_conf or show_all:
            results.append({
                "Paire": pair.replace("C:", ""),
                "Confluence H1": int(score)
            })

df_result = pd.DataFrame(results).sort_values(by="Confluence H1", ascending=False)

if df_result.empty:
    st.warning("Aucune paire ne correspond aux critères de confluence.")
else:
    st.success(f"{len(df_result)} paires trouvées.")
    st.dataframe(df_result, use_container_width=True)
