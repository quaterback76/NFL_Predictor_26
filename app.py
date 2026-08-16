import streamlit as st
import nfl_data_py as nfl
import pandas as pd
import numpy as np
import requests

# --- KONFIGURACJA ---
st.set_page_config(page_title="NFL AI Predictor 2026 Pro", layout="wide")

# 1. BAZA STADIONÓW
STADIUMS = {
    'ARI': {'name': 'State Farm Stadium', 'lat': 33.5276, 'lon': -112.2626, 'dome': True},
    'ATL': {'name': 'Mercedes-Benz Stadium', 'lat': 33.7553, 'lon': -84.4010, 'dome': True},
    'BAL': {'name': 'M&T Bank Stadium', 'lat': 39.2780, 'lon': -76.6227, 'dome': False},
    'BUF': {'name': 'Highmark Stadium', 'lat': 42.7738, 'lon': -78.7870, 'dome': False},
    'CAR': {'name': 'Bank of America Stadium', 'lat': 35.2258, 'lon': -80.8528, 'dome': False},
    'CHI': {'name': 'Soldier Field', 'lat': 41.8623, 'lon': -87.6167, 'dome': False},
    'CIN': {'name': 'Paycor Stadium', 'lat': 39.0955, 'lon': -84.5161, 'dome': False},
    'CLE': {'name': 'Huntington Bank Field', 'lat': 41.5061, 'lon': -81.6995, 'dome': False},
    'DAL': {'name': 'AT&T Stadium', 'lat': 32.7473, 'lon': -97.0945, 'dome': True},
    'DEN': {'name': 'Empower Field at Mile High', 'lat': 39.7439, 'lon': -105.0201, 'dome': False},
    'DET': {'name': 'Ford Field', 'lat': 42.3400, 'lon': -83.0456, 'dome': True},
    'GB':  {'name': 'Lambeau Field', 'lat': 44.5013, 'lon': -88.0622, 'dome': False},
    'HOU': {'name': 'NRG Stadium', 'lat': 29.6847, 'lon': -95.4107, 'dome': True},
    'IND': {'name': 'Lucas Oil Stadium', 'lat': 39.7601, 'lon': -86.1639, 'dome': True},
    'JAX': {'name': 'EverBank Stadium', 'lat': 30.3239, 'lon': -81.6373, 'dome': False},
    'KC':  {'name': 'Arrowhead Stadium', 'lat': 39.0489, 'lon': -94.4839, 'dome': False},
    'LV':  {'name': 'Allegiant Stadium', 'lat': 36.0909, 'lon': -115.1833, 'dome': True},
    'LAC': {'name': 'SoFi Stadium', 'lat': 33.9535, 'lon': -118.3392, 'dome': True},
    'LAR': {'name': 'SoFi Stadium', 'lat': 33.9535, 'lon': -118.3392, 'dome': True},
    'MIA': {'name': 'Hard Rock Stadium', 'lat': 25.9580, 'lon': -80.2389, 'dome': False},
    'MIN': {'name': 'U.S. Bank Stadium', 'lat': 44.9735, 'lon': -93.2575, 'dome': True},
    'NE':  {'name': 'Gillette Stadium', 'lat': 42.0909, 'lon': -71.2643, 'dome': False},
    'NO':  {'name': 'Caesars Superdome', 'lat': 29.9511, 'lon': -90.0812, 'dome': True},
    'NYG': {'name': 'MetLife Stadium', 'lat': 40.8128, 'lon': -74.0742, 'dome': False},
    'NYJ': {'name': 'MetLife Stadium', 'lat': 40.8128, 'lon': -74.0742, 'dome': False},
    'PHI': {'name': 'Lincoln Financial Field', 'lat': 39.9012, 'lon': -75.1675, 'dome': False},
    'PIT': {'name': 'Acrisure Stadium', 'lat': 40.4467, 'lon': -80.0158, 'dome': False},
    'SF':  {'name': "Levi's Stadium", 'lat': 37.4033, 'lon': -121.9702, 'dome': False},
    'SEA': {'name': 'Lumen Field', 'lat': 47.5952, 'lon': -122.3316, 'dome': False},
    'TB':  {'name': 'Raymond James Stadium', 'lat': 27.9759, 'lon': -82.5033, 'dome': False},
    'TEN': {'name': 'Nissan Stadium', 'lat': 36.1665, 'lon': -86.7713, 'dome': False},
    'WAS': {'name': 'Northwest Stadium', 'lat': 38.9076, 'lon': -76.8645, 'dome': False}
}

# 2. MODUŁ TRANSFERÓW I ROZWOJU (Wartości bazowe)
ROSTER_POWER = {
    'ARI': 0.5,  'ATL': 0.0,  'BAL': 1.0,  'BUF': -0.5, 
    'CAR': 0.2,  'CHI': 1.8,  'CIN': 0.5,  'CLE': -0.2, 
    'DAL': -0.5, 'DEN': 0.3,  'DET': 2.0,  'GB':  1.5, 
    'HOU': 2.5,  'IND': 0.8,  'JAX': 0.2,  'KC':  2.2, 
    'LV':  -0.3, 'LAC': 0.7,  'LAR': 0.5,  'MIA': 0.0, 
    'MIN': 0.4,  'NE':  0.2,  'NO':  -1.0, 'NYG': 0.3, 
    'NYJ': 1.5,  'PHI': -0.5, 'PIT': 0.6,  'SF':  1.2, 
    'SEA': 1.8,  'TB':  0.0,  'TEN': 0.2,  'WAS': 0.8
}

# 3. WAGI PUNKTOWE ZA KONTUZJE POSZCZEGÓLNYCH POZYCJI
POSITION_VALS = {
    'QB': 4.0, 'WR': 1.5, 'RB': 1.0, 'TE': 1.2, 'OL': 1.0,
    'DE': 1.5, 'DT': 1.2, 'LB': 1.0, 'CB': 1.4, 'S':  0.8
}

# --- DANE ---
@st.cache_data(ttl=3600)
def load_data_2026():
    team_ppg = {}
    try:
        year = 2025
        sched = nfl.import_schedules([year])
        if sched is not None and not sched.empty:
            h_scores = sched.groupby('home_team')['home_score'].mean()
            a_scores = sched.groupby('away_team')['away_score'].mean()
            team_ppg = ((h_scores + a_scores) / 2).to_dict()
    except:
        pass
    return team_ppg

def fetch_weather(lat, lon):
    try:
        url = f"https://open-meteo.com{lat}&longitude={lon}&current=wind_speed_10m,precipitation&timezone=auto"
        r = requests.get(url, timeout=3).json()['current']
        return {"wind": r['wind_speed_10m'], "rain": r['precipitation'] > 0}
    except: 
        return {"wind": 0, "rain": False}

# --- SYMULACJA ---
def run_simulation(h_val, a_val, sims=10000):
    h_val = max(10.0, h_val) if not np.isnan(h_val) else 23.0
    a_val = max(10.0, a_val) if not np.isnan(a_val) else 21.0
    h_sims = np.random.poisson(h_val, sims)
    a_sims = np.random.poisson(a_val, sims)
    return (np.sum(h_sims > a_sims)/sims)*100, (np.sum(a_sims > h_sims)/sims)*100, np.mean(h_sims), np.mean(a_sims)

# --- START APLIKACJI ---
st.title("🏈 NFL AI Predictor 2026: Roster & Form Analysis")
team_ppg = load_data_2026()
teams = sorted(STADIUMS.keys())

c1, c2 = st.columns(2)
with c1: h_team = st.selectbox("🏠 Gospodarz:", teams, index=teams.index('HOU'))
with c2: a_team = st.selectbox("✈️ Gość:", teams, index=teams.index('NE'))

st_info = STADIUMS.get(h_team, {'name': 'Neutral Field', 'lat': 39.82, 'lon': -98.57, 'dome': False})
weather = fetch_weather(st_info['lat'], st_info['lon']) if not st_info['dome'] else {"wind":0, "rain":False}

# --- PANEL BOCZNY ---
st.sidebar.header("⚙️ Zarządzanie Składem 2026")
st.sidebar.caption("Zmień formę drużyny za pomocą suwaków:")

h_base_mod = float(ROSTER_POWER.get(h_team, 0.0))
a_base_mod = float(ROSTER_POWER.get(a_team, 0.0))

h_mod = st.sidebar.slider(f"Modyfikator {h_team}", -7.0, 7.0, h_base_mod)
a_mod = st.sidebar.slider(f"Modyfikator {a_team}", -7.0, 7.0, a_base_mod)

# --- MODEL AI KALKULACJA ---
base_h = float(team_ppg.get(h_team, 23.5) + h_mod)
base_a = float(team_ppg.get(a_team, 21.2) + a_mod)
win_h, win_a, avg_h, avg_a = run_simulation(base_h, base_a)

# --- WIDOK GŁÓWNY INTERFEJSU ---
st.divider()
st.subheader(f"📊 Symulacja Przedmeczowa: {h_team} vs {a_team}")

r1, r2, r3 = st.columns(3)
with r1:
    st.metric(f"Szansa {h_team}", f"{win_h:.1f}%")
    st.caption(f"Modyfikator: {h_mod:+.1f}")

with r2:
    st.markdown(f"<h1 style='text-align: center;'>{round(avg_h)} : {round(avg_a)}</h1>", unsafe_allow_html=True)
    st.write(f"<p style='text-align: center;'>O/U: <b>{round(avg_h + avg_a, 1)}</b></p>", unsafe_allow_html=True)

with r3:
    st.metric(f"Szansa {a_team}", f"{win_a:.1f}%")
    st.caption(f"Modyfikator: {a_mod:+.1f}")

st.info(f"💡 **Analiza:** Wyjściowa siła {h_team} oparta na algorytmach sportowych została skorygowana o {h_mod:+.1f} pkt. Pogoda na stadionie: Wiatr {weather['wind']} km/h.")
