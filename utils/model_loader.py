from pathlib import Path
import joblib
import streamlit as st
from utils.data_loader import FEATURES
def load_model():
    """Read current session model, or the trusted model installed by an admin."""
    if 'uploaded_model' in st.session_state:
        return st.session_state['uploaded_model']
    path = Path(__file__).resolve().parents[1] / 'models' / 'trained_model.joblib'
    if not path.exists():
        return None
    try:
        model = joblib.load(path)
        if not callable(getattr(model, 'predict', None)):
            raise ValueError('Invalid model artifact')
        return model
    except Exception as exc:
        st.error(f'Could not load model: {exc}')
        return None

def predict(model,frame):
    values=model.predict(frame[FEATURES]); probs=None
    if hasattr(model,'predict_proba'):
        probs=model.predict_proba(frame[FEATURES])
    return values,probs


def load_uploaded_model(uploaded_file):
    """Only load a trusted notebook artifact: joblib deserialization executes code."""
    import io
    import pandas as pd
    from utils.data_loader import LABELS
    if uploaded_file.size > 100 * 1024 * 1024:
        raise ValueError('Model exceeds the 100 MB upload limit.')
    model = joblib.load(io.BytesIO(uploaded_file.getvalue()))
    if not callable(getattr(model, 'predict', None)):
        raise ValueError('This file does not contain a prediction model.')
    probe = pd.DataFrame([{feature: 0 for feature in FEATURES}])
    result = model.predict(probe[FEATURES])
    if len(result) != 1 or str(result[0]) not in LABELS:
        raise ValueError('Model prediction labels are not compatible with Gymnasyum.')
    if hasattr(model, 'predict_proba'):
        model.predict_proba(probe[FEATURES])
    return model
