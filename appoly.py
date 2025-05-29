import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import pytz

# Configuration de l'application
st.set_page_config(
    page_title="Forex Scanner (Polygon.io)",
    page_icon="💹",
    layout="wide"
)

# Titre et description
st.title("📊 Forex Scanner avec Polygon.io")
st.markdown("""
Analyse des principales paires Forex utilisant les données de Polygon.io  
*Fonctionne avec les plans Basic (gratuits)*
""")

# Initialisation des secrets
if 'POLYGON_API_KEY' not in st.session_state:
    st.session_state.POLYGON_API_KEY = None

# Sidebar pour la configuration
with st.sidebar:
    st.header("🔧 Configuration")
    api_key = st.text_input("Entrez votre clé Polygon.io", type="password")
    if api_key:
        st.session_state.POLYGON_API_KEY = api_key
        st.success("Clé API configurée!")
    
    st.markdown("---")
    st.markdown("""
    ### Comment obtenir une clé API ?
    1. Créez un compte sur [polygon.io](https://polygon.io)
    2. Trouvez votre clé dans le dashboard
    3. Les plans gratuits ont des limites
    """)

# Liste des paires Forex avec les codes Polygon
FOREX_PAIRS = {
    "EUR/USD": "C:EURUSD",
    "USD/JPY": "C:USDJPY", 
    "GBP/USD": "C:GBPUSD",
    "USD/CHF": "C:USDCHF",
    "USD/CAD": "C:USDCAD",
    "AUD/USD": "C:AUDUSD",
    "NZD/USD": "C:NZDUSD",
    "XAU/USD": "C:XAUUSD"
}

# Fonction pour récupérer les données quotidiennes
@st.cache_data(ttl=3600, show_spinner="Récupération des données...")
def get_daily_forex_data(pair, days=30):
    if not st.session_state.POLYGON_API_KEY:
        st.error("Veuillez entrer une clé API valide")
        return None
    
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{pair}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}?adjusted=true&sort=asc&apiKey={st.session_state.POLYGON_API_KEY}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if response.status_code != 200:
            st.error(f"Erreur {data.get('status')}: {data.get('error', 'Unknown error')}")
            return None
            
        if data.get('resultsCount', 0) == 0:
            st.warning("Aucune donnée disponible pour cette période")
            return None
            
        df = pd.DataFrame(data['results'])
        df['date'] = pd.to_datetime(df['t'], unit='ms').dt.tz_localize('UTC')
        df = df.rename(columns={
            'o': 'Open',
            'h': 'High', 
            'l': 'Low',
            'c': 'Close',
            'v': 'Volume'
        })
        
        return df[['date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données: {str(e)}")
        return None

# Fonction pour calculer les indicateurs techniques
def calculate_technical_indicators(df):
    try:
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = compute_rsi(df['Close'], 14)
        return df
    except Exception as e:
        st.error(f"Erreur dans les calculs: {str(e)}")
        return df

# Calcul du RSI
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# Interface principale
if st.session_state.POLYGON_API_KEY:
    selected_pair = st.selectbox(
        "Sélectionnez une paire Forex:",
        options=list(FOREX_PAIRS.keys())
    
    days_to_fetch = st.slider(
        "Nombre de jours à analyser:",
        min_value=5,
        max_value=365,
        value=90)
    
    if st.button("Analyser"):
        with st.spinner(f"Récupération des données pour {selected_pair}..."):
            pair_code = FOREX_PAIRS[selected_pair]
            df = get_daily_forex_data(pair_code, days_to_fetch)
            
            if df is not None:
                df = calculate_technical_indicators(df)
                
                # Affichage des données
                st.subheader(f"Dernières données pour {selected_pair}")
                st.dataframe(df.tail(10), use_container_width=True)
                
                # Graphique des prix
                st.subheader("Évolution des prix")
                st.line_chart(df.set_index('date')[['Close', 'SMA_20', 'SMA_50']])
                
                # Graphique RSI
                st.subheader("Indicateur RSI (14 jours)")
                st.line_chart(df.set_index('date')['RSI'])
                
                # Dernière valeur
                last_close = df.iloc[-1]['Close']
                prev_close = df.iloc[-2]['Close']
                change_pct = ((last_close - prev_close) / prev_close) * 100
                
                st.metric(
                    label=f"Dernier cours {selected_pair}",
                    value=f"{last_close:.5f}",
                    delta=f"{change_pct:.2f}%"
                )
else:
    st.warning("Veuillez entrer votre clé API Polygon.io dans la sidebar")

# Pied de page
st.markdown("---")
st.markdown("""
**Remarques:**
- Les données peuvent être retardées de 15 minutes sur les plans gratuits
- Limite de 5 requêtes/minute sur le plan Basic
- Les données intraday ne sont pas disponibles avec ce plan
""")
