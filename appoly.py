import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import time
import traceback 
import requests 
import logging

# Configuration de la page
st.set_page_config(
    page_title="Scanner Confluence Forex (Polygon)", 
    page_icon="⭐", 
    layout="wide"
)

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Titre et description
st.title("🔍 Scanner Confluence Forex Premium (Données Polygon.io)")
st.markdown("*Utilisation de l'API Polygon.io pour les données de marché H1*")

# Configuration de la clé API
POLYGON_API_KEY = None
try:
    POLYGON_API_KEY = st.secrets["POLYGON_API_KEY"]
    st.sidebar.success("✅ Clé API Polygon.io chargée avec succès")
except KeyError:
    st.error("❌ Erreur: Secret 'POLYGON_API_KEY' non défini dans les secrets Streamlit")
    st.info("💡 Ajoutez votre clé API Polygon dans les secrets Streamlit")
    st.stop()
except Exception as e:
    st.error(f"❌ Erreur lors du chargement de la clé API: {e}")
    st.stop()

# Vérification finale de la clé API
if not POLYGON_API_KEY or POLYGON_API_KEY.strip() == "":
    st.error("❌ Clé API Polygon.io vide ou invalide")
    st.stop()

# Paires de devises Polygon
FOREX_PAIRS_POLYGON = [
    "C:EURUSD", "C:GBPUSD", "C:USDJPY", "C:USDCHF", "C:AUDUSD", 
    "C:USDCAD", "C:NZDUSD", "C:EURJPY", "C:GBPJPY", "C:EURGBP"
]

# Constantes pour les calculs
MIN_DATA_POINTS = 60
DEFAULT_RSI_PERIOD = 14
DEFAULT_ADX_PERIOD = 14
DEFAULT_HMA_PERIOD = 20
ADX_THRESHOLD = 20

# ==================== FONCTIONS D'INDICATEURS TECHNIQUES ====================

def calculate_ema(series, period):
    """Calcule l'EMA (Exponential Moving Average)"""
    try:
        if len(series.dropna()) < period:
            return pd.Series(np.nan, index=series.index)
        return series.ewm(span=period, adjust=False).mean()
    except Exception as e:
        logger.error(f"Erreur EMA: {e}")
        return pd.Series(np.nan, index=series.index)

def calculate_rma(series, period):
    """Calcule le RMA (Relative Moving Average)"""
    try:
        if len(series.dropna()) < period:
            return pd.Series(np.nan, index=series.index)
        return series.ewm(alpha=1/period, adjust=False).mean()
    except Exception as e:
        logger.error(f"Erreur RMA: {e}")
        return pd.Series(np.nan, index=series.index)

def calculate_wma(series, period):
    """Calcule le WMA (Weighted Moving Average)"""
    try:
        weights = np.arange(1, period + 1)
        def wma_func(x):
            if len(x) == period:
                return np.sum(x * weights) / np.sum(weights)
            return np.nan
        
        return series.rolling(window=period).apply(wma_func, raw=True)
    except Exception as e:
        logger.error(f"Erreur WMA: {e}")
        return pd.Series(np.nan, index=series.index)

def calculate_hull_ma(close_prices, period=20):
    """Calcule le Hull Moving Average"""
    try:
        if len(close_prices.dropna()) < period + int(np.sqrt(period)):
            return pd.Series(np.nan, index=close_prices.index)
        
        half_period = max(1, int(period / 2))
        sqrt_period = max(1, int(np.sqrt(period)))
        
        # Calcul des WMA
        wma1 = calculate_wma(close_prices, half_period)
        wma2 = calculate_wma(close_prices, period)
        
        # Hull MA brut
        raw_hma = 2 * wma1 - wma2
        
        # Hull MA final
        hma = calculate_wma(raw_hma, sqrt_period)
        return hma
        
    except Exception as e:
        logger.error(f"Erreur Hull MA: {e}")
        return pd.Series(np.nan, index=close_prices.index)

def calculate_rsi(prices, period=14):
    """Calcule le RSI (Relative Strength Index)"""
    try:
        if len(prices.dropna()) < period + 1:
            return pd.Series(50.0, index=prices.index)
        
        delta = prices.diff()
        gains = delta.where(delta > 0, 0.0)
        losses = -delta.where(delta < 0, 0.0)
        
        avg_gains = calculate_rma(gains, period)
        avg_losses = calculate_rma(losses, period)
        
        # Éviter la division par zéro
        avg_losses_safe = avg_losses.replace(0, 1e-9)
        rs = avg_gains / avg_losses_safe
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)
        
    except Exception as e:
        logger.error(f"Erreur RSI: {e}")
        return pd.Series(50.0, index=prices.index)

def calculate_adx(high, low, close, period=14):
    """Calcule l'ADX (Average Directional Index)"""
    try:
        if len(high.dropna()) < period * 2:
            return pd.Series(0.0, index=high.index)
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = calculate_rma(tr, period)
        
        # Directional Movement
        up_move = high.diff()
        down_move = low.shift(1) - low
        
        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)
        
        # Directional Indicators
        atr_safe = atr.replace(0, 1e-9)
        plus_di = 100 * (calculate_rma(plus_dm, period) / atr_safe)
        minus_di = 100 * (calculate_rma(minus_dm, period) / atr_safe)
        
        # ADX
        dx_denominator = (plus_di + minus_di).replace(0, 1e-9)
        dx = 100 * (abs(plus_di - minus_di) / dx_denominator)
        adx = calculate_rma(dx, period)
        
        return adx.fillna(0)
        
    except Exception as e:
        logger.error(f"Erreur ADX: {e}")
        return pd.Series(0.0, index=high.index)

def calculate_heiken_ashi(ohlc_data):
    """Calcule les bougies Heiken Ashi"""
    try:
        if len(ohlc_data.dropna()) < 1:
            return pd.Series(dtype=float, index=ohlc_data.index), pd.Series(dtype=float, index=ohlc_data.index)
        
        ha_close = (ohlc_data['Open'] + ohlc_data['High'] + ohlc_data['Low'] + ohlc_data['Close']) / 4
        ha_open = pd.Series(index=ohlc_data.index, dtype=float)
        
        # Première valeur
        ha_open.iloc[0] = (ohlc_data['Open'].iloc[0] + ohlc_data['Close'].iloc[0]) / 2
        
        # Calcul récursif
        for i in range(1, len(ohlc_data)):
            ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
        
        return ha_open, ha_close
        
    except Exception as e:
        logger.error(f"Erreur Heiken Ashi: {e}")
        return pd.Series(dtype=float, index=ohlc_data.index), pd.Series(dtype=float, index=ohlc_data.index)

def calculate_smoothed_heiken_ashi(ohlc_data, ema1_period=10, ema2_period=10):
    """Calcule les bougies Heiken Ashi lissées"""
    try:
        if len(ohlc_data.dropna()) < max(ema1_period, ema2_period) + 1:
            return pd.Series(dtype=float, index=ohlc_data.index), pd.Series(dtype=float, index=ohlc_data.index)
        
        # Lissage des prix OHLC
        smoothed_open = calculate_ema(ohlc_data['Open'], ema1_period)
        smoothed_high = calculate_ema(ohlc_data['High'], ema1_period)
        smoothed_low = calculate_ema(ohlc_data['Low'], ema1_period)
        smoothed_close = calculate_ema(ohlc_data['Close'], ema1_period)
        
        smoothed_ohlc = pd.DataFrame({
            'Open': smoothed_open,
            'High': smoothed_high,
            'Low': smoothed_low,
            'Close': smoothed_close
        }).dropna()
        
        if smoothed_ohlc.empty:
            return pd.Series(dtype=float, index=ohlc_data.index), pd.Series(dtype=float, index=ohlc_data.index)
        
        # Calcul Heiken Ashi sur données lissées
        ha_open_raw, ha_close_raw = calculate_heiken_ashi(smoothed_ohlc)
        
        # Lissage final
        sha_open = calculate_ema(ha_open_raw, ema2_period)
        sha_close = calculate_ema(ha_close_raw, ema2_period)
        
        return sha_open, sha_close
        
    except Exception as e:
        logger.error(f"Erreur Smoothed Heiken Ashi: {e}")
        return pd.Series(dtype=float, index=ohlc_data.index), pd.Series(dtype=float, index=ohlc_data.index)

def calculate_ichimoku_signal(high, low, close, tenkan_period=9, kijun_period=26, senkou_b_period=52):
    """Calcule le signal Ichimoku"""
    try:
        min_required = max(tenkan_period, kijun_period, senkou_b_period)
        if len(high.dropna()) < min_required:
            return 0
        
        # Lignes Ichimoku
        tenkan_sen = (high.rolling(window=tenkan_period).max() + low.rolling(window=tenkan_period).min()) / 2
        kijun_sen = (high.rolling(window=kijun_period).max() + low.rolling(window=kijun_period).min()) / 2
        senkou_span_a = (tenkan_sen + kijun_sen) / 2
        senkou_span_b = (high.rolling(window=senkou_b_period).max() + low.rolling(window=senkou_b_period).min()) / 2
        
        # Vérification des valeurs finales
        current_close = close.iloc[-1]
        current_span_a = senkou_span_a.iloc[-1]
        current_span_b = senkou_span_b.iloc[-1]
        
        if pd.isna(current_close) or pd.isna(current_span_a) or pd.isna(current_span_b):
            return 0
        
        # Détermination du signal
        cloud_top = max(current_span_a, current_span_b)
        cloud_bottom = min(current_span_a, current_span_b)
        
        if current_close > cloud_top:
            return 1  # Signal haussier
        elif current_close < cloud_bottom:
            return -1  # Signal baissier
        else:
            return 0  # Neutre (dans le nuage)
            
    except Exception as e:
        logger.error(f"Erreur Ichimoku: {e}")
        return 0

# ==================== FONCTION DE RÉCUPÉRATION DES DONNÉES ====================

@st.cache_data(ttl=300)  # Cache pendant 5 minutes
def get_polygon_data(symbol, timespan='hour', multiplier=1, days_history=30):
    """Récupère les données depuis l'API Polygon.io"""
    global POLYGON_API_KEY
    
    if not POLYGON_API_KEY:
        logger.error("Clé API Polygon non disponible")
        return None
    
    try:
        # Calcul des dates
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days_history)
        
        # URL de l'API
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 5000,
            "apikey": POLYGON_API_KEY
        }
        
        logger.info(f"Récupération des données pour {symbol}")
        
        # Requête API
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Vérification de la réponse
        if data.get('status') != 'OK' or not data.get('results'):
            logger.warning(f"Pas de données pour {symbol}: {data.get('error', 'Erreur inconnue')}")
            return None
        
        # Conversion en DataFrame
        df = pd.DataFrame(data['results'])
        
        # Renommage des colonnes
        column_mapping = {
            'o': 'Open',
            'h': 'High', 
            'l': 'Low',
            'c': 'Close',
            'v': 'Volume',
            't': 'timestamp'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Conversion de l'index en datetime
        df.index = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        
        # Sélection des colonnes OHLCV
        ohlcv_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df[ohlcv_columns].dropna()
        
        # Vérification de la quantité de données
        if len(df) < MIN_DATA_POINTS:
            logger.warning(f"Données insuffisantes pour {symbol}: {len(df)} points (minimum: {MIN_DATA_POINTS})")
            return None
        
        logger.info(f"Données récupérées avec succès pour {symbol}: {len(df)} points")
        return df
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Erreur de requête pour {symbol}: {e}")
        return None
    except Exception as e:
        logger.error(f"Erreur inattendue pour {symbol}: {e}")
        return None

# ==================== FONCTION DE CALCUL DES SIGNAUX ====================

def calculate_all_signals(data):
    """Calcule tous les signaux de confluence"""
    if data is None or len(data) < MIN_DATA_POINTS:
        logger.error(f"Données insuffisantes: {len(data) if data is not None else 'None'} points")
        return None
    
    # Vérification des colonnes requises
    required_columns = ['Open', 'High', 'Low', 'Close']
    if not all(col in data.columns for col in required_columns):
        logger.error("Colonnes OHLC manquantes")
        return None
    
    # Extraction des séries de prix
    open_price = data['Open']
    high_price = data['High']
    low_price = data['Low']
    close_price = data['Close']
    
    # Prix OHLC4 pour le RSI
    ohlc4 = (open_price + high_price + low_price + close_price) / 4
    
    # Initialisation des compteurs
    bull_signals = 0
    bear_signals = 0
    signal_details = {}
    
    # 1. Hull Moving Average (HMA)
    try:
        hma = calculate_hull_ma(close_price, DEFAULT_HMA_PERIOD)
        if len(hma.dropna()) >= 2:
            current_hma = hma.iloc[-1]
            previous_hma = hma.iloc[-2]
            
            if not pd.isna(current_hma) and not pd.isna(previous_hma):
                if current_hma > previous_hma:
                    bull_signals += 1
                    signal_details['HMA'] = "▲ Haussier"
                elif current_hma < previous_hma:
                    bear_signals += 1
                    signal_details['HMA'] = "▼ Baissier"
                else:
                    signal_details['HMA'] = "─ Neutre"
            else:
                signal_details['HMA'] = "N/A"
        else:
            signal_details['HMA'] = "N/A"
    except Exception as e:
        signal_details['HMA'] = f"Erreur: {str(e)[:20]}"
        logger.error(f"Erreur HMA: {e}")
    
    # 2. RSI
    try:
        rsi = calculate_rsi(ohlc4, DEFAULT_RSI_PERIOD)
        if len(rsi.dropna()) >= 1:
            current_rsi = rsi.iloc[-1]
            
            if not pd.isna(current_rsi):
                signal_details['RSI_value'] = f"{current_rsi:.1f}"
                
                if current_rsi > 50:
                    bull_signals += 1
                    signal_details['RSI'] = f"▲ {current_rsi:.1f}"
                elif current_rsi < 50:
                    bear_signals += 1
                    signal_details['RSI'] = f"▼ {current_rsi:.1f}"
                else:
                    signal_details['RSI'] = f"─ {current_rsi:.1f}"
            else:
                signal_details['RSI'] = "N/A"
                signal_details['RSI_value'] = "N/A"
        else:
            signal_details['RSI'] = "N/A"
            signal_details['RSI_value'] = "N/A"
    except Exception as e:
        signal_details['RSI'] = f"Erreur: {str(e)[:20]}"
        signal_details['RSI_value'] = "N/A"
        logger.error(f"Erreur RSI: {e}")
    
    # 3. ADX
    try:
        adx = calculate_adx(high_price, low_price, close_price, DEFAULT_ADX_PERIOD)
        if len(adx.dropna()) >= 1:
            current_adx = adx.iloc[-1]
            
            if not pd.isna(current_adx):
                signal_details['ADX_value'] = f"{current_adx:.1f}"
                
                if current_adx >= ADX_THRESHOLD:
                    # ADX fort = tendance forte (compte pour les deux directions)
                    bull_signals += 1
                    bear_signals += 1
                    signal_details['ADX'] = f"✓ Fort ({current_adx:.1f})"
                else:
                    signal_details['ADX'] = f"✗ Faible ({current_adx:.1f})"
            else:
                signal_details['ADX'] = "N/A"
                signal_details['ADX_value'] = "N/A"
        else:
            signal_details['ADX'] = "N/A"
            signal_details['ADX_value'] = "N/A"
    except Exception as e:
        signal_details['ADX'] = f"Erreur: {str(e)[:20]}"
        signal_details['ADX_value'] = "N/A"
        logger.error(f"Erreur ADX: {e}")
    
    # 4. Heiken Ashi
    try:
        ha_open, ha_close = calculate_heiken_ashi(data)
        if len(ha_open.dropna()) >= 1 and len(ha_close.dropna()) >= 1:
            current_ha_open = ha_open.iloc[-1]
            current_ha_close = ha_close.iloc[-1]
            
            if not pd.isna(current_ha_open) and not pd.isna(current_ha_close):
                if current_ha_close > current_ha_open:
                    bull_signals += 1
                    signal_details['HA'] = "▲ Haussier"
                elif current_ha_close < current_ha_open:
                    bear_signals += 1
                    signal_details['HA'] = "▼ Baissier"
                else:
                    signal_details['HA'] = "─ Neutre"
            else:
                signal_details['HA'] = "N/A"
        else:
            signal_details['HA'] = "N/A"
    except Exception as e:
        signal_details['HA'] = f"Erreur: {str(e)[:20]}"
        logger.error(f"Erreur HA: {e}")
    
    # 5. Smoothed Heiken Ashi
    try:
        sha_open, sha_close = calculate_smoothed_heiken_ashi(data, 10, 10)
        if len(sha_open.dropna()) >= 1 and len(sha_close.dropna()) >= 1:
            current_sha_open = sha_open.iloc[-1]
            current_sha_close = sha_close.iloc[-1]
            
            if not pd.isna(current_sha_open) and not pd.isna(current_sha_close):
                if current_sha_close > current_sha_open:
                    bull_signals += 1
                    signal_details['SHA'] = "▲ Haussier"
                elif current_sha_close < current_sha_open:
                    bear_signals += 1
                    signal_details['SHA'] = "▼ Baissier"
                else:
                    signal_details['SHA'] = "─ Neutre"
            else:
                signal_details['SHA'] = "N/A"
        else:
            signal_details['SHA'] = "N/A"
    except Exception as e:
        signal_details['SHA'] = f"Erreur: {str(e)[:20]}"
        logger.error(f"Erreur SHA: {e}")
    
    # 6. Ichimoku
    try:
        ichimoku_signal = calculate_ichimoku_signal(high_price, low_price, close_price)
        
        if ichimoku_signal == 1:
            bull_signals += 1
            signal_details['Ichimoku'] = "▲ Haussier"
        elif ichimoku_signal == -1:
            bear_signals += 1
            signal_details['Ichimoku'] = "▼ Baissier"
        else:  # ichimoku_signal == 0
            signal_details['Ichimoku'] = "─ Neutre"
            
    except Exception as e:
        signal_details['Ichimoku'] = f"Erreur: {str(e)[:20]}"
        logger.error(f"Erreur Ichimoku: {e}")
    
    # Calcul de la confluence et direction
    confluence_score = max(bull_signals, bear_signals)
    
    if bull_signals > bear_signals:
        direction = "HAUSSIER"
    elif bear_signals > bull_signals:
        direction = "BAISSIER"
    elif bull_signals == bear_signals and bull_signals > 0:
        direction = "MIXTE"
    else:
        direction = "NEUTRE"
    
    return {
        'confluence': confluence_score,
        'direction': direction,
        'bull_count': bull_signals,
        'bear_count': bear_signals,
        'rsi_value': signal_details.get('RSI_value', 'N/A'),
        'adx_value': signal_details.get('ADX_value', 'N/A'),
        'signals': signal_details
    }

def get_star_rating(confluence_score):
    """Convertit le score de confluence en étoiles"""
    star_map = {
        6: "⭐⭐⭐⭐⭐⭐",
        5: "⭐⭐⭐⭐⭐",
        4: "⭐⭐⭐⭐",
        3: "⭐⭐⭐",
        2: "⭐⭐",
        1: "⭐",
        0: "ATTENDRE"
    }
    return star_map.get(confluence_score, "ATTENDRE")

# ==================== INTERFACE UTILISATEUR ====================

# Colonnes pour la mise en page
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("⚙️ Paramètres de Scan")
    
    # Filtre de confluence minimum
    min_confluence = st.selectbox(
        "Confluence minimum",
        options=[0, 1, 2, 3, 4, 5, 6],
        index=3,
        help="Nombre minimum de signaux concordants requis"
    )
    
    # Option pour afficher toutes les paires
    show_all_pairs = st.checkbox(
        "Afficher toutes les paires",
        help="Ignorer le filtre de confluence et afficher tous les résultats"
    )
    
    # Historique des données
    days_history = st.slider("Jours d'historique", 7, 60, 30)
    
    # Bouton de scan
    scan_button = st.button(
        "🔍 Lancer le Scan",
        type="primary",
        use_container_width=True,
        help="Scanner toutes les paires de devises avec l'API Polygon.io"
    )

with col2:
    if scan_button:
        st.info("🔄 Scan en cours... Veuillez patienter")
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_pairs = len(FOREX_PAIRS_POLYGON)
        
        for i, pair in enumerate(FOREX_PAIRS_POLYGON):
            # Mise à jour du statut
            progress = (i + 1) / total_pairs
            progress_bar.progress(progress)
            status_text.text(f"Analyse de {pair} ({i + 1}/{total_pairs})")
            
            try:
                # Récupération des données
                data = get_polygon_data(pair, 'hour', 1, days_history)
                
                if data is not None:
                    # Calcul des signaux
                    signals = calculate_all_signals(data)
                    
                    if signals:
                        result = {
                            'Paire': pair,
                            'Direction': signals['direction'],
                            'Confluence': signals['confluence'],
                            'Étoiles': get_star_rating(signals['confluence']),
                            'RSI': signals['rsi_value'],
                            'ADX': signals['adx_value'],
                            'Bull': signals['bull_count'],
                            'Bear': signals['bear_count'],
                            'Signaux': signals['signals']
                        }
                    else:
                        result = {
                            'Paire': pair,
                            'Direction': 'ERREUR CALCUL',
                            'Confluence': 0,
                            'Étoiles': 'N/A',
                            'RSI': 'N/A',
                            'ADX': 'N/A',
                            'Bull': 0,
                            'Bear': 0,
                            'Signaux': {'Erreur': 'Échec du calcul des signaux'}
                        }
                else:
                    result = {
                        'Paire': pair,
                        'Direction': 'ERREUR DONNÉES',
                        'Confluence': 0,
                        'Étoiles': 'N/A',
                        'RSI': 'N/A',
                        'ADX': 'N/A',
                        'Bull': 0,
                        'Bear': 0,
                        'Signaux': {'Erreur': 'Données non disponibles'}
                    }
                
                results.append(result)
                
                # Pause pour respecter les limites de taux de l'API
                if i < total_pairs - 1:  # Pas de pause après la dernière paire
                    time.sleep(12)  # 12 secondes entre les requêtes
                    
            except Exception as e:
                logger.error(f"Erreur lors du traitement de {pair}: {e}")
                results.append({
                    'Paire': pair,
                    'Direction': 'ERREUR',
                    'Confluence': 0,
                    'Étoiles': 'N/A',
                    'RSI': 'N/A',
                    'ADX': 'N/A',
                    '
