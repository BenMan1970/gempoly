import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# --- Configuration de la page ---
st.set_page_config(
    page_title="Scanner Confluence Forex",
    page_icon="⭐",
    layout="wide"
)
st.title("🔍 Scanner Confluence Forex Premium (Polygon.io)")
st.markdown("*Version adaptée pour Polygon.io*")

# --- Liste des paires Forex ---
FOREX_PAIRS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF',
    'AUDUSD', 'USDCAD', 'NZDUSD', 'EURJPY',
    'GBPJPY', 'EURGBP', 'XAUUSD'
]

# --- Clé API Polygon.io ---
API_KEY = "VOTRE_CLE_API_POLYGON"

# --- Fonctions d'indicateurs techniques (inchangées) ---
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rma(series, period):
    return series.ewm(alpha=1/period, adjust=False).mean()

def hull_ma_pine(data_close, period=20):
    half_length = int(period / 2)
    sqrt_length = int(np.sqrt(period))
    wma_half_period = data_close.rolling(window=half_length).apply(lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), raw=True)
    wma_full_period = data_close.rolling(window=period).apply(lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), raw=True)
    diff_wma = 2 * wma_half_period - wma_full_period
    hma_series = diff_wma.rolling(window=sqrt_length).apply(lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), raw=True)
    return hma_series

def rsi_pine(prices_ohlc4, period=10):
    deltas = prices_ohlc4.diff()
    gains = deltas.where(deltas > 0, 0.0)
    losses = -deltas.where(deltas < 0, 0.0)
    avg_gains = rma(gains, period)
    avg_losses = rma(losses, period)
    rs = avg_gains / avg_losses.replace(0, 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def adx_pine(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = rma(tr, period)
    up_move = high.diff()
    down_move = low.shift(1) - low
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)
    safe_atr = atr.replace(0, 1e-9)
    plus_di = 100 * (rma(plus_dm, period) / safe_atr)
    minus_di = 100 * (rma(minus_dm, period) / safe_atr)
    dx_denominator = (plus_di + minus_di).replace(0, 1e-9)
    dx = 100 * (abs(plus_di - minus_di) / dx_denominator)
    adx_series = rma(dx, period)
    return adx_series.fillna(0)

def heiken_ashi_pine(df_ohlc):
    ha_df = pd.DataFrame(index=df_ohlc.index)
    if df_ohlc.empty:
        ha_df['HA_Open'] = pd.Series(dtype=float)
        ha_df['HA_Close'] = pd.Series(dtype=float)
        return ha_df['HA_Open'], ha_df['HA_Close']
    ha_df['HA_Close'] = (df_ohlc['Open'] + df_ohlc['High'] + df_ohlc['Low'] + df_ohlc['Close']) / 4
    ha_df['HA_Open'] = np.nan
    if not df_ohlc.empty:
        ha_df.iloc[0, ha_df.columns.get_loc('HA_Open')] = (df_ohlc['Open'].iloc[0] + df_ohlc['Close'].iloc[0]) / 2
        for i in range(1, len(df_ohlc)):
            ha_df.iloc[i, ha_df.columns.get_loc('HA_Open')] = \
                (ha_df.iloc[i-1, ha_df.columns.get_loc('HA_Open')] + 
                 ha_df.iloc[i-1, ha_df.columns.get_loc('HA_Close')]) / 2
    return ha_df['HA_Open'], ha_df['HA_Close']

def smoothed_heiken_ashi_pine(df_ohlc, len1=10, len2=10):
    ema_open = ema(df_ohlc['Open'], len1)
    ema_high = ema(df_ohlc['High'], len1)
    ema_low = ema(df_ohlc['Low'], len1)
    ema_close = ema(df_ohlc['Close'], len1)
    ha_intermediate_df = pd.DataFrame(index=df_ohlc.index)
    ha_intermediate_df['Open'] = ema_open
    ha_intermediate_df['High'] = ema_high
    ha_intermediate_df['Low'] = ema_low
    ha_intermediate_df['Close'] = ema_close
    ha_open_intermediate, ha_close_intermediate = heiken_ashi_pine(ha_intermediate_df)
    smoothed_ha_open = ema(ha_open_intermediate, len2)
    smoothed_ha_close = ema(ha_close_intermediate, len2)
    return smoothed_ha_open, smoothed_ha_close

def ichimoku_pine_signal(df_high, df_low, df_close, tenkan_p=9, kijun_p=26, senkou_b_p=52):
    tenkan_sen = (df_high.rolling(window=tenkan_p).max() + df_low.rolling(window=tenkan_p).min()) / 2
    kijun_sen = (df_high.rolling(window=kijun_p).max() + df_low.rolling(window=kijun_p).min()) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    senkou_span_b = (df_high.rolling(window=senkou_b_p).max() + df_low.rolling(window=senkou_b_p).min()) / 2
    current_close = df_close.iloc[-1]
    current_ssa = senkou_span_a.iloc[-1]
    current_ssb = senkou_span_b.iloc[-1]
    if pd.isna(current_ssa) or pd.isna(current_ssb) or pd.isna(current_close):
        return 0
    cloud_top_now = max(current_ssa, current_ssb)
    cloud_bottom_now = min(current_ssa, current_ssb)
    signal = 0
    if current_close > cloud_top_now:
        signal = 1
    elif current_close < cloud_bottom_now:
        signal = -1
    return signal

# --- Fonction pour récupérer les données depuis Polygon.io ---
@st.cache_data(ttl=300)
def get_data(symbol, interval='1h'):
    # Polygon symbol format: "C:EURUSD"
    polygon_symbol = f"C:{symbol.upper()}"
    timespan = 'hour' if interval == '1h' else 'day'
    multiplier = 1
    end_date = datetime.utcnow()
    if interval == '1h':
        start_date = end_date - timedelta(days=30)
    else:
        start_date = end_date - timedelta(days=365)
    from_str = start_date.strftime('%Y-%m-%d')
    to_str = end_date.strftime('%Y-%m-%d')
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{polygon_symbol}/range/"
        f"{multiplier}/{timespan}/{from_str}/{to_str}"
        f"?adjusted=true&sort=asc&limit=5000&apiKey={API_KEY}"
    )
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
        if 'results' not in data or not data['results']:
            return None
        df = pd.DataFrame(data['results'])
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close'}, inplace=True)
        return df[['Open', 'High', 'Low', 'Close']].dropna()
    except Exception as e:
        print(f"Erreur Polygon pour {symbol}: {str(e)}")
        return None

# --- Calcul des signaux (inchangé) ---
def calculate_all_signals_pine(data):
    if data is None or len(data) < 60:
        return None
    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in data.columns for col in required_cols):
        print(f"Colonnes manquantes dans les données: {required_cols} pour un symbole.")
        return None
    close = data['Close']
    high = data['High']
    low = data['Low']
    open_price = data['Open']
    ohlc4 = (open_price + high + low + close) / 4
    bull_confluences = 0
    bear_confluences = 0
    signal_details_pine = {}
    # HMA
    try:
        hma_series = hull_ma_pine(close, 20)
        if len(hma_series) >= 2 and not hma_series.iloc[-2:].isna().any():
            hma_val = hma_series.iloc[-1]
            hma_prev = hma_series.iloc[-2]
            if hma_val > hma_prev:
                bull_confluences += 1; signal_details_pine['HMA'] = "▲"
            elif hma_val < hma_prev:
                bear_confluences += 1; signal_details_pine['HMA'] = "▼"
            else: signal_details_pine['HMA'] = "─"
        else: signal_details_pine['HMA'] = "N/A"
    except Exception as e: signal_details_pine['HMA'] = f"Err({type(e).__name__})"
    # RSI
    try:
        rsi_series = rsi_pine(ohlc4, 10)
        if len(rsi_series) >=1 and not pd.isna(rsi_series.iloc[-1]):
            rsi_val = rsi_series.iloc[-1]
            signal_details_pine['RSI_val'] = f"{rsi_val:.0f}"
            if rsi_val > 50:
                bull_confluences += 1; signal_details_pine['RSI'] = f"▲({rsi_val:.0f})"
            elif rsi_val < 50:
                bear_confluences += 1; signal_details_pine['RSI'] = f"▼({rsi_val:.0f})"
            else: signal_details_pine['RSI'] = f"─({rsi_val:.0f})"
        else: signal_details_pine['RSI'] = "N/A"
    except Exception as e:
        signal_details_pine['RSI'] = f"Err({type(e).__name__})"
        signal_details_pine['RSI_val'] = "N/A"
    # ADX
    try:
        adx_series = adx_pine(high, low, close, 14)
        if len(adx_series) >= 1 and not pd.isna(adx_series.iloc[-1]):
            adx_val = adx_series.iloc[-1]
            signal_details_pine['ADX_val'] = f"{adx_val:.0f}"
            if adx_val >= 20:
                bull_confluences += 1
                bear_confluences += 1
                signal_details_pine['ADX'] = f"✔({adx_val:.0f})"
            else:
                signal_details_pine['ADX'] = f"✖({adx_val:.0f})"
        else: signal_details_pine['ADX'] = "N/A"
    except Exception as e:
        signal_details_pine['ADX'] = f"Err({type(e).__name__})"
        signal_details_pine['ADX_val'] = "N/A"
    # Heiken Ashi
    try:
        ha_open, ha_close = heiken_ashi_pine(data)
        if len(ha_open) >=1 and len(ha_close) >=1 and \
           not pd.isna(ha_open.iloc[-1]) and not pd.isna(ha_close.iloc[-1]):
            if ha_close.iloc[-1] > ha_open.iloc[-1]:
                bull_confluences += 1; signal_details_pine['HA'] = "▲"
            elif ha_close.iloc[-1] < ha_open.iloc[-1]:
                bear_confluences += 1; signal_details_pine['HA'] = "▼"
            else: signal_details_pine['HA'] = "─"
        else: signal_details_pine['HA'] = "N/A"
    except Exception as e: signal_details_pine['HA'] = f"Err({type(e).__name__})"
    # Smoothed Heiken Ashi
    try:
        sha_open, sha_close = smoothed_heiken_ashi_pine(data, len1=10, len2=10)
        if len(sha_open) >=1 and len(sha_close) >=1 and \
           not pd.isna(sha_open.iloc[-1]) and not pd.isna(sha_close.iloc[-1]):
            if sha_close.iloc[-1] > sha_open.iloc[-1]:
                bull_confluences += 1; signal_details_pine['SHA'] = "▲"
            elif sha_close.iloc[-1] < sha_open.iloc[-1]:
                bear_confluences += 1; signal_details_pine['SHA'] = "▼"
            else: signal_details_pine['SHA'] = "─"
        else: signal_details_pine['SHA'] = "N/A"
    except Exception as e: signal_details_pine['SHA'] = f"Err({type(e).__name__})"
    # Ichimoku
    try:
        if len(high) >= 52 and len(low) >= 52 and len(close) >= 52:
            ichi_signal = ichimoku_pine_signal(high, low, close, tenkan_p=9, kijun_p=26, senkou_b_p=52)
            if ichi_signal == 1:
                bull_confluences += 1; signal_details_pine['Ichi'] = "▲"
            elif ichi_signal == -1:
                bear_confluences += 1; signal_details_pine['Ichi'] = "▼"
            else:
                signal_details_pine['Ichi'] = "─"
        else:
            signal_details_pine['Ichi'] = "N/D"
    except Exception as e: signal_details_pine['Ichi'] = f"Err({type(e).__name__})"
    confluence_value = max(bull_confluences, bear_confluences)
    direction = "NEUTRE"
    if bull_confluences > bear_confluences: direction = "HAUSSIER"
    elif bear_confluences > bull_confluences: direction = "BAISSIER"
    elif bull_confluences == bear_confluences and bull_confluences > 0:
        direction = "CONFLIT"
    return {
        'confluence_P': confluence_value,
        'direction_P': direction,
        'bull_P': bull_confluences,
        'bear_P': bear_confluences,
        'rsi_P': signal_details_pine.get('RSI_val', "N/A"),
        'adx_P': signal_details_pine.get('ADX_val', "N/A"),
        'signals_P': signal_details_pine
    }

def get_stars_pine(confluence_value):
    if confluence_value == 6: return "⭐⭐⭐⭐⭐⭐"
    elif confluence_value == 5: return "⭐⭐⭐⭐⭐"
    elif confluence_value == 4: return "⭐⭐⭐⭐"
    elif confluence_value == 3: return "⭐⭐⭐"
    elif confluence_value == 2: return "⭐⭐"
    elif confluence_value == 1: return "⭐"
    else: return "WAIT"

# --- Interface Utilisateur ---
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("⚙️ Paramètres")
    min_confluence_filter = st.selectbox(
        "Confluence minimum (0-6)",
        options=[0, 1, 2, 3, 4, 5, 6],
        index=3,
        format_func=lambda x: f"{x} (confluence)"
    )
    show_all_pairs = st.checkbox("Voir toutes les paires (ignorer filtre confluence)")
    scan_button = st.button("🔍 Scanner (Logique Pine H1)", type="primary", use_container_width=True)

with col2:
    if scan_button:
        st.info("🔄 Scan en cours (logique Pine Script H1)...")
        processed_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i, symbol in enumerate(FOREX_PAIRS):
            current_progress = (i + 1) / len(FOREX_PAIRS)
            progress_bar.progress(current_progress)
            pair_name = symbol if symbol != "XAUUSD" else "XAU/USD"
            status_text.text(f"Analyse (H1): {pair_name} ({i+1}/{len(FOREX_PAIRS)})")
            data_h1 = get_data(symbol, interval='1h')
            if data_h1 is not None:
                signals = calculate_all_signals_pine(data_h1)
                if signals:
                    stars_str = get_stars_pine(signals['confluence_P'])
                    result_data = {
                        'Paire': pair_name,
                        'Direction': signals['direction_P'],
                        'Conf. (0-6)': signals['confluence_P'],
                        'Étoiles': stars_str,
                        'RSI': signals['rsi_P'],
                        'ADX': signals['adx_P'],
                        'Détails': str(signals['signals_P'])
                    }
                    if show_all_pairs or signals['confluence_P'] >= min_confluence_filter:
                        processed_results.append(result_data)
        if processed_results:
            df_results = pd.DataFrame(processed_results)
            st.dataframe(df_results, use_container_width=True)
        else:
            st.warning("Aucune paire ne correspond aux critères de confluence.")

                    
