
import streamlit as st

st.set_page_config(
    page_title='Gymnasyum | Risk Intelligence',
    page_icon='🏋️',
    layout='wide'
)

st.markdown(
    """<style>
    .block-container {padding-top:1.7rem}
    [data-testid='stMetric'] {
        background:#f0f5fc;
        border:1px solid #dce7f5;
        border-radius:12px;
        padding:16px
    }
    h1,h2,h3 {color:#102b50}
    </style>""",
    unsafe_allow_html=True
)

st.title('🏋️ Gymnasyum Customer Risk Intelligence')
st.caption('Customer activity intelligence • 3,000 synthetic records • Not real customer data')

st.info(
    'Choose a page from the sidebar. '
    'Predictions use the trained Machine Learning model; '
    'Gemini features require GEMINI_API_KEY.'
)

# Model status
st.sidebar.divider()
st.sidebar.subheader('⚙️ Model Status')

from utils.model_loader import load_model

if load_model() is not None:
    st.sidebar.success('Model ready for predictions')
else:
    st.sidebar.error('Model unavailable. Please contact the administrator.')
