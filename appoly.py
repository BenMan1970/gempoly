import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import time
import traceback 

# NOUVEL IMPORT pour la bibliothèque Polygon
from polygon import RESTClient
# requests n'est plus directement utilisé par nous pour Polygon

st.set_page_config(page_title="Scanner Confluence Forex (Polygon Client)", page_icon="⭐", layout="wide")
st.title("🔍 Scanner Confluence Forex Premium (Données Polygon.io via Client)")
st.markdown("*Utilisation de la bibliothèque `polygon-api-client`*")

POLYGON_API_KEY = None
polygon_client = None # Instance du client Polygon

try:
    POLYGON_API_KEY = st.secrets["POLYGON_API_KEY"]
except KeyError:
    st.error("Erreur: Secret 'POLYGON_API_KEY' non défini. Configurez vos secrets.")
    st.stop()

if POLYGON_API_KEY:
    try:
        polygon_client = RESTClient(POLYGON_API_KEY)
        st.sidebar.success("Client Polygon.io initialisé.")
        # Petit test pour vérifier la connexion (optionnel, peut consommer un appel API)
        # try:
        #     test_details = polygon_client.get_ticker_details("AAPL") # Test avec un ticker connu
        #     print(f"Test de connexion Polygon réussi pour {test_details.ticker}")
        # except Exception as test_e:
        #     st.sidebar.warning(f"Test de connexion Polygon avec ticker a échoué: {test_e}. La clé est peut-être limitée.")
        #     print(f"Test de connexion Polygon avec ticker a échoué: {test_e}")

    except Exception as e:
        st.error(f"Erreur initialisation client Polygon.io: {e}")
        st.sidebar.error("Échec initialisation client Polygon.")
        polygon_client = None 
else:
    st.error("Clé API Polygon.io non disponible.")
    st.stop()

FOREX_PAIRS_POLYGON = [
    "X:EURUSD", "X:GBPUSD", "X:USDJPY", "X:USDCHF", "X:AUDUSD", 
    "X:USDCAD", "X:NZDUSD", "X:EURJPY", "X:GBPJPY", "X:EURGBP"
]

# --- Fonctions d'indicateurs techniques (INCHANGÉES, la version robuste) ---
def ema(s, p): return s.ewm(span=p, adjust=False).mean()
# ... (COLLEZ TOUTES VOS FONCTIONS D'INDICATEURS ICI : rma, hull_ma_pine, rsi_pine, adx_pine, heiken_ashi_pine, smoothed_heiken_ashi_pine, ichimoku_pine_signal) ...
# ...ELLES SONT IDENTIQUES À CELLES DU DERNIER SCRIPT COMPLET QUE JE VOUS AI DONNÉ...
# Pour la concision, je ne les répète pas ici, mais elles doivent être présentes dans votre script final.
def rma(s, p): return s.ewm(alpha=1/p, adjust=False).mean()
def hull_ma_pine(dc, p=20):
    if len(dc.dropna()) < p + int(np.sqrt(p)): return pd.Series(np.nan, index=dc.index)
    try:
        hl=max(1,int(p/2)); sl=max(1,int(np.sqrt(p)))
        if len(dc.dropna()) < hl or len(dc.dropna()) < p: return pd.Series(np.nan, index=dc.index)
        wma1=dc.rolling(window=hl).apply(lambda x:np.sum(x*np.arange(1,len(x)+1))/np.sum(np.arange(1,len(x)+1)) if len(x)==hl else np.nan, raw=True)
        wma2=dc.rolling(window=p).apply(lambda x:np.sum(x*np.arange(1,len(x)+1))/np.sum(np.arange(1,len(x)+1)) if len(x)==p else np.nan, raw=True)
        raw_hma=2*wma1-wma2
        if len(raw_hma.dropna()) < sl : return pd.Series(np.nan, index=dc.index)
        hma = raw_hma.rolling(window=sl).apply(lambda x:np.sum(x*np.arange(1,len(x)+1))/np.sum(np.arange(1,len(x)+1)) if len(x)==sl else np.nan, raw=True)
        return hma
    except Exception as e: print(f"Erreur dans hull_ma_pine: {e}"); return pd.Series(np.nan, index=dc.index)
def rsi_pine(po4,p=10):
    if len(po4.dropna()) < p + 1: return pd.Series(50.0, index=po4.index)
    try:
        d=po4.diff();g=d.where(d>0,0.0);l=-d.where(d<0,0.0);ag=rma(g,p);al=rma(l,p);al_safe = al.replace(0,1e-9) 
        rs=ag/al_safe;rsi=100-(100/(1+rs));return rsi.fillna(50)
    except Exception as e: print(f"Erreur dans rsi_pine: {e}"); return pd.Series(50.0, index=po4.index)
def adx_pine(h,l,c,p=14):
    if len(h.dropna()) < p*2 or len(l.dropna()) < p*2 or len(c.dropna()) < p*2: return pd.Series(0.0, index=h.index)
    try:
        tr1=h-l;tr2=abs(h-c.shift(1));tr3=abs(l-c.shift(1));tr=pd.concat([tr1,tr2,tr3],axis=1).max(axis=1);atr=rma(tr,p)
        um=h.diff();dm=l.shift(1)-l
        pdm=pd.Series(np.where((um>dm)&(um>0),um,0.0),index=h.index);mdm=pd.Series(np.where((dm>um)&(dm>0),dm,0.0),index=h.index)
        satr=atr.replace(0,1e-9);pdi=100*(rma(pdm,p)/satr);mdi=100*(rma(mdm,p)/satr)
        dxden=(pdi+mdi).replace(0,1e-9);dx=100*(abs(pdi-mdi)/dxden);return rma(dx,p).fillna(0)
    except Exception as e: print(f"Erreur dans adx_pine: {e}"); return pd.Series(0.0, index=h.index)
def heiken_ashi_pine(dfo):
    if len(dfo.dropna()) < 1: return pd.Series(dtype=float, index=dfo.index), pd.Series(dtype=float, index=dfo.index)
    try:
        ha=pd.DataFrame(index=dfo.index)
        ha['HA_Close']=(dfo['Open']+dfo['High']+dfo['Low']+dfo['Close'])/4;ha['HA_Open']=np.nan
        if not dfo.empty:
            ha.iloc[0,ha.columns.get_loc('HA_Open')]=(dfo['Open'].iloc[0]+dfo['Close'].iloc[0])/2
            for i in range(1,len(dfo)):ha.iloc[i,ha.columns.get_loc('HA_Open')]=(ha.iloc[i-1,ha.columns.get_loc('HA_Open')]+ha.iloc[i-1,ha.columns.get_loc('HA_Close')])/2
        return ha['HA_Open'],ha['HA_Close']
    except Exception as e: print(f"Erreur dans heiken_ashi_pine: {e}"); return pd.Series(dtype=float, index=dfo.index), pd.Series(dtype=float, index=dfo.index)
def smoothed_heiken_ashi_pine(dfo,l1=10,l2=10):
    if len(dfo.dropna()) < max(l1,l2) +1 : return pd.Series(dtype=float, index=dfo.index), pd.Series(dtype=float, index=dfo.index)
    try:
        eo=ema(dfo['Open'],l1);eh=ema(dfo['High'],l1);el=ema(dfo['Low'],l1);ec=ema(dfo['Close'],l1)
        hai=pd.DataFrame({'Open':eo,'High':eh,'Low':el,'Close':ec},index=dfo.index).dropna() 
        if hai.empty: return pd.Series(dtype=float, index=dfo.index), pd.Series(dtype=float, index=dfo.index)
        hao_i,hac_i=heiken_ashi_pine(hai);sho=ema(hao_i,l2);shc=ema(hac_i,l2);return sho,shc
    except Exception as e: print(f"Erreur dans smoothed_heiken_ashi_pine: {e}"); return pd.Series(dtype=float, index=dfo.index), pd.Series(dtype=float, index=dfo.index)
def ichimoku_pine_signal(df_high, df_low, df_close, tenkan_p=9, kijun_p=26, senkou_b_p=52):
    min_len_req=max(tenkan_p,kijun_p,senkou_b_p)
    if len(df_high.dropna())<min_len_req or len(df_low.dropna())<min_len_req or len(df_close.dropna())<min_len_req:print(f"Ichi:Data<({len(df_close)}) vs req {min_len_req}.");return 0
    try:
        ts=(df_high.rolling(window=tenkan_p).max()+df_low.rolling(window=tenkan_p).min())/2;ks=(df_high.rolling(window=kijun_p).max()+df_low.rolling(window=kijun_p).min())/2
        sa=(ts+ks)/2;sb=(df_high.rolling(window=senkou_b_p).max()+df_low.rolling(window=senkou_b_p).min())/2
        if pd.isna(df_close.iloc[-1]) or pd.isna(sa.iloc[-1]) or pd.isna(sb.iloc[-1]):print("Ichi:NaN close/spans.");return 0
        ccl=df_close.iloc[-1];cssa=sa.iloc[-1];cssb=sb.iloc[-1];ctn=max(cssa,cssb);cbn=min(cssa,cssb);sig=0
        if ccl>ctn:sig=1
        elif ccl<cbn:sig=-1
        return sig
    except Exception as e: print(f"Erreur dans ichimoku_pine_signal: {e}"); return 0


# --- Fonction get_data utilisant polygon-api-client ---
@st.cache_data(ttl=600) 
def get_data_polygon_client(symbol_pg: str, timespan_pg: str = 'hour', multiplier_pg: int = 1, days_history_pg: int = 30):
    global polygon_client # Utilise l'instance du client initialisée globalement
    if polygon_client is None: 
        st.error("FATAL: Client Polygon.io non initialisé (get_data).")
        print("FATAL: Client Polygon.io non initialisé (get_data).")
        return None
    
    date_to_dt = datetime.now(timezone.utc)
    date_from_dt = date_to_dt - timedelta(days=days_history_pg)
    
    # Format AAAA-MM-JJ pour l'API Polygon
    date_to_str = date_to_dt.strftime('%Y-%m-%d')
    date_from_str = date_from_dt.strftime('%Y-%m-%d')

    print(f"\n--- Début get_data_polygon_client: sym='{symbol_pg}', mult={multiplier_pg}, span='{timespan_pg}', from={date_from_str}, to={date_to_str} ---")
    
    try:
        # Utilisation de client.list_aggs
        aggs = polygon_client.list_aggs(
            ticker=symbol_pg,
            multiplier=multiplier_pg,
            timespan=timespan_pg,
            from_=date_from_str,
            to=date_to_str,
            adjusted=True,
            sort="asc",
            limit=5000 # Nombre max de barres
        )
        
        # aggs est un itérateur d'objets Agg(v, vw, o, c, h, l, t, n)
        # Convertir en DataFrame
        df = pd.DataFrame(aggs)
        
        print(f"Données Polygon (client) reçues pour {symbol_pg}. Nombre de barres: {len(df)}")

        if df.empty:
            st.warning(f"Polygon Client: Pas de données pour {symbol_pg}.")
            print(f"Polygon Client: Pas de données pour {symbol_pg}.")
            return None

        # Renommer les colonnes et définir l'index temporel
        # Les objets Agg ont des attributs: open, high, low, close, volume, timestamp
        df.rename(columns={
            'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 
            'volume':'Volume', 'timestamp':'timestamp_ms' # timestamp est en millisecondes
        }, inplace=True)
        
        df.index = pd.to_datetime(df['timestamp_ms'], unit='ms', utc=True)
        
        if len(df) < 60: 
            st.warning(f"Données Polygon (client) insuffisantes pour {symbol_pg} ({len(df)} barres). Requis: 60.")
            print(f"Données Polygon (client) insuffisantes pour {symbol_pg} ({len(df)} barres). Requis: 60.")
            return None
        
        print(f"Données pour {symbol_pg} OK. Retour de {len(df)}l.\n--- Fin get_data_polygon_client {symbol_pg} ---\n")
        return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna(subset=['Open','High','Low','Close'])

    # Gérer les exceptions de l'API Polygon (la bibliothèque peut lever des exceptions spécifiques)
    except Exception as e: # Exception générique pour l'instant
        # La bibliothèque polygon-api-client peut lever polygon.exceptions.NoResultsError, etc.
        st.error(f"Erreur API Polygon (client) pour {symbol_pg}: {type(e).__name__} - {e}")
        print(f"ERREUR API POLYGON (client) {symbol_pg}:\n{traceback.format_exc()}")
        return None

# --- calculate_all_signals_pine (VERSION CORRECTEMENT INDENTÉE, comme dans le message précédent) ---
# ... (COLLEZ ICI LA FONCTION calculate_all_signals_pine CORRECTEMENT INDENTÉE) ...
def calculate_all_signals_pine(data):
    if data is None or len(data) < 60: print(f"calculate_all_signals: Données non fournies ou trop courtes ({len(data) if data is not None else 'None'} lignes)."); return None
    required_cols = ['Open', 'High', 'Low', 'Close']; 
    if not all(col in data.columns for col in required_cols): print(f"calculate_all_signals: Colonnes OHLC manquantes."); return None
    close = data['Close']; high = data['High']; low = data['Low']; open_price = data['Open']
    ohlc4 = (open_price + high + low + close) / 4
    bull_confluences, bear_confluences, signal_details_pine = 0, 0, {}
    try: hma_series = hull_ma_pine(close, 20)
        if len(hma_series) >= 2 and not hma_series.iloc[-2:].isna().any():
            hma_val = hma_series.iloc[-1]; hma_prev = hma_series.iloc[-2]
            if hma_val > hma_prev: bull_confluences += 1; signal_details_pine['HMA'] = "▲"
            elif hma_val < hma_prev: bear_confluences += 1; signal_details_pine['HMA'] = "▼"
            else: signal_details_pine['HMA'] = "─"
        else: signal_details_pine['HMA'] = "N/A"
    except Exception as e: signal_details_pine['HMA'] = "ErrHMA"; print(f"Erreur HMA: {e}")
    try: rsi_series = rsi_pine(ohlc4, 10)
        if len(rsi_series) >=1 and not pd.isna(rsi_series.iloc[-1]):
            rsi_val = rsi_series.iloc[-1]; signal_details_pine['RSI_val'] = f"{rsi_val:.0f}"
            if rsi_val > 50: bull_confluences += 1; signal_details_pine['RSI'] = f"▲({rsi_val:.0f})"
            elif rsi_val < 50: bear_confluences += 1; signal_details_pine['RSI'] = f"▼({rsi_val:.0f})"
            else: signal_details_pine['RSI'] = f"─({rsi_val:.0f})"
        else: signal_details_pine['RSI'] = "N/A"
    except Exception as e: signal_details_pine['RSI'] = "ErrRSI"; signal_details_pine['RSI_val'] = "N/A"; print(f"Erreur RSI: {e}")
    try: adx_series = adx_pine(high, low, close, 14)
        if len(adx_series) >= 1 and not pd.isna(adx_series.iloc[-1]):
            adx_val = adx_series.iloc[-1]; signal_details_pine['ADX_val'] = f"{adx_val:.0f}"
            if adx_val >= 20: bull_confluences += 1; bear_confluences += 1; signal_details_pine['ADX'] = f"✔({adx_val:.0f})"
            else: signal_details_pine['ADX'] = f"✖({adx_val:.0f})"
        else: signal_details_pine['ADX'] = "N/A"
    except Exception as e: signal_details_pine['ADX'] = "ErrADX"; signal_details_pine['ADX_val'] = "N/A"; print(f"Erreur ADX: {e}")
    try: ha_open, ha_close = heiken_ashi_pine(data)
        if len(ha_open) >=1 and len(ha_close) >=1 and not pd.isna(ha_open.iloc[-1]) and not pd.isna(ha_close.iloc[-1]):
            if ha_close.iloc[-1] > ha_open.iloc[-1]: bull_confluences += 1; signal_details_pine['HA'] = "▲"
            elif ha_close.iloc[-1] < ha_open.iloc[-1]: bear_confluences += 1; signal_details_pine['HA'] = "▼"
            else: signal_details_pine['HA'] = "─"
        else: signal_details_pine['HA'] = "N/A"
    except Exception as e: signal_details_pine['HA'] = "ErrHA"; print(f"Erreur HA: {e}")
    try: sha_open, sha_close = smoothed_heiken_ashi_pine(data, 10, 10)
        if len(sha_open) >=1 and len(sha_close) >=1 and not pd.isna(sha_open.iloc[-1]) and not pd.isna(sha_close.iloc[-1]):
            if sha_close.iloc[-1] > sha_open.iloc[-1]: bull_confluences += 1; signal_details_pine['SHA'] = "▲"
            elif sha_close.iloc[-1] < sha_open.iloc[-1]: bear_confluences += 1; signal_details_pine['SHA'] = "▼"
            else: signal_details_pine['SHA'] = "─"
        else: signal_details_pine['SHA'] = "N/A"
    except Exception as e: signal_details_pine['SHA'] = "ErrSHA"; print(f"Erreur SHA: {e}")
    try: ichimoku_signal_val = ichimoku_pine_signal(high, low, close)
        if ichimoku_signal_val == 1: bull_confluences += 1; signal_details_pine['Ichi'] = "▲"
        elif ichimoku_signal_val == -1: bear_confluences += 1; signal_details_pine['Ichi'] = "▼"
        elif ichimoku_signal_val == 0 and (len(data) < max(9,26,52) or (len(data) > 0 and pd.isna(data['Close'].iloc[-1]))): signal_details_pine['Ichi'] = "N/D"
        else: signal_details_pine['Ichi'] = "─"
    except Exception as e: signal_details_pine['Ichi'] = "ErrIchi"; print(f"Erreur Ichi: {e}")
    confluence_value=max(bull_confluences,bear_confluences); direction="NEUTRE"
    if bull_confluences > bear_confluences: direction="HAUSSIER"
    elif bear_confluences > bull_confluences: direction="BAISSIER"
    elif bull_confluences == bear_confluences and bull_confluences > 0: direction="CONFLIT"
    return{'confluence_P':confluence_value,'direction_P':direction,'bull_P':bull_confluences,'bear_P':bear_confluences,'rsi_P':signal_details_pine.get('RSI_val',"N/A"),'adx_P':signal_details_pine.get('ADX_val',"N/A"),'signals_P':signal_details_pine}


# --- Fonction get_stars_pine (CORRIGÉE pour la syntaxe if/elif) ---
def get_stars_pine(confluence_value):
    # ... (COLLEZ ICI LA FONCTION get_stars_pine CORRECTEMENT INDENTÉE) ...
    if confluence_value == 6: return "⭐⭐⭐⭐⭐⭐"
    elif confluence_value == 5: return "⭐⭐⭐⭐⭐"
    elif confluence_value == 4: return "⭐⭐⭐⭐"
    elif confluence_value == 3: return "⭐⭐⭐"
    elif confluence_value == 2: return "⭐⭐"
    elif confluence_value == 1: return "⭐"
    else: return "WAIT"


# --- Interface Utilisateur (adaptée pour polygon_client) ---
col1,col2=st.columns([1,3])
with col1:
    st.subheader("⚙️ Paramètres");min_conf=st.selectbox("Confluence min (0-6)",options=[0,1,2,3,4,5,6],index=3,format_func=lambda x:f"{x} (confluence)")
    show_all=st.checkbox("Voir toutes les paires (ignorer filtre)");
    scan_dis_pg = polygon_client is None; # Utiliser polygon_client pour désactiver
    scan_tip_pg="Client Polygon non initialisé." if scan_dis_pg else "Lancer scan (Polygon.io)"
    scan_btn=st.button("🔍 Scanner (Données Polygon H1)",type="primary",use_container_width=True,disabled=scan_dis_pg,help=scan_tip_pg)

with col2:
    if scan_btn: # Implique que polygon_client n'est pas None
        st.info(f"🔄 Scan en cours (Polygon.io H1)...");pr_res=[];pb=st.progress(0);stx=st.empty()
        for i,symbol_pg_scan in enumerate(FOREX_PAIRS_POLYGON): 
            pnd=symbol_pg_scan;cp=(i+1)/len(FOREX_PAIRS_POLYGON);pb.progress(cp);stx.text(f"Analyse (Polygon H1):{pnd}({i+1}/{len(FOREX_PAIRS_POLYGON)})")
            # Appel à la nouvelle fonction get_data_polygon_client
            d_h1_pg = get_data_polygon_client(symbol_pg_scan, timespan_pg="hour", multiplier_pg=1, days_history_pg=30) 
            
            if d_h1_pg is not None:
                # ... (logique de traitement des signaux et résultats, identique à avant) ...
                sigs=calculate_all_signals_pine(d_h1_pg)
                if sigs:strs=get_stars_pine(sigs['confluence_P']);rd={'Paire':pnd,'Direction':sigs['direction_P'],'Conf. (0-6)':sigs['confluence_P'],'Étoiles':strs,'RSI':sigs['rsi_P'],'ADX':sigs['adx_P'],'Bull':sigs['bull_P'],'Bear':sigs['bear_P'],'details':sigs['signals_P']};pr_res.append(rd)
                else:pr_res.append({'Paire':pnd,'Direction':'ERREUR CALCUL','Conf. (0-6)':0,'Étoiles':'N/A','RSI':'N/A','ADX':'N/A','Bull':0,'Bear':0,'details':{'Info':'Calcul signaux (Polygon) échoué'}})
            else:pr_res.append({'Paire':pnd,'Direction':'ERREUR DONNÉES PG','Conf. (0-6)':0,'Étoiles':'N/A','RSI':'N/A','ADX':'N/A','Bull':0,'Bear':0,'details':{'Info':'Données Polygon non dispo/symb invalide(logs serveur)'}})
            print(f"Pause de 13 secondes pour limite de taux Polygon.io...");time.sleep(13) 
        
        pb.empty();stx.empty()
        if pr_res:
            # ... (logique d'affichage des résultats, identique à avant) ...
            dfa=pd.DataFrame(pr_res);dfd=dfa[dfa['Conf. (0-6)']>=min_conf].copy()if not show_all else dfa.copy()
            if not show_all:st.success(f"🎯 {len(dfd)} paire(s) avec {min_conf}+ confluence (Polygon).")
            else:st.info(f"🔍 Affichage des {len(dfd)} paires (Polygon).")
            if not dfd.empty:
                dfds=dfd.sort_values('Conf. (0-6)',ascending=False);vcs=[c for c in['Paire','Direction','Conf. (0-6)','Étoiles','RSI','ADX','Bull','Bear']if c in dfds.columns]
                st.dataframe(dfds[vcs],use_container_width=True,hide_index=True)
                with st.expander("📊 Détails des signaux (Polygon)"):
                    for _,r in dfds.iterrows():
                        sm=r.get('details',{});
                        if not isinstance(sm,dict):sm={'Info':'Détails non dispo'}
                        st.write(f"**{r.get('Paire','N/A')}** - {r.get('Étoiles','N/A')} ({r.get('Conf. (0-6)','N/A')}) - Dir: {r.get('Direction','N/A')}")
                        dc=st.columns(6);so=['HMA','RSI','ADX','HA','SHA','Ichi']
                        for idx,sk in enumerate(so):dc[idx].metric(label=sk,value=sm.get(sk,"N/P"))
                        st.divider()
            else:st.warning(f"❌ Aucune paire avec critères filtrage (Polygon). Vérifiez erreurs données/symbole.")
        else:st.error("❌ Aucune paire traitée (Polygon). Vérifiez logs serveur.")

with st.expander("ℹ️ Comment ça marche (Logique Pine Script avec Données Polygon.io)"):
    st.markdown("""**6 Signaux Confluence:** HMA(20),RSI(10),ADX(14)>=20,HA(Simple),SHA(10,10),Ichi(9,26,52).**Comptage & Étoiles:**Pine.**Source:**Polygon.io API.""")
st.caption("Scanner H1 (Polygon.io). Multi-TF non actif. Attention aux limites de taux de l'API Polygon.io.")
    
