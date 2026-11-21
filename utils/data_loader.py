from pathlib import Path
import pandas as pd
import streamlit as st
FEATURES = ['attendance_frequency','missed_session','days_since_last_visit','reschedule_frequency','remaining_package','response_to_reminder']
LABELS = ['Active','Need Attention','At Risk']
@st.cache_data
def load_data():
    path = Path(__file__).resolve().parents[1] / 'data' / 'gymnasyum_synthetic_3000.csv'
    df = pd.read_csv(path)
    required = ['member_id',*FEATURES,'risk_label']
    missing = set(required)-set(df.columns)
    if missing: raise ValueError(f'Missing columns: {sorted(missing)}')
    if not set(df.risk_label.dropna().unique()).issubset(set(LABELS)): raise ValueError('Unexpected risk labels')
    for c in FEATURES: df[c] = pd.to_numeric(df[c],errors='coerce')
    return df
