import streamlit as st
st.set_page_config(page_title='Gymnasyum | Risk Intelligence',page_icon='🏋️',layout='wide')
st.markdown("""<style>.block-container{padding-top:1.7rem} [data-testid='stMetric']{background:#f0f5fc;border:1px solid #dce7f5;border-radius:12px;padding:16px} h1,h2,h3{color:#102b50}</style>""",unsafe_allow_html=True)
st.title('🏋️ Gymnasyum Customer Risk Intelligence')
st.caption('Customer activity intelligence • 3,000 synthetic records • Not real customer data')
st.info('Choose a page from the sidebar. Predictions require an exported trained_model.joblib; Gemini features require GEMINI_API_KEY.')


st.sidebar.divider()
st.sidebar.subheader('⚙️ Model Setup')
from utils.model_loader import load_model, load_uploaded_model
if load_model() is not None:
    st.sidebar.success('Model ready for predictions')
else:
    st.sidebar.info('No model loaded yet')
st.sidebar.caption('Local/admin use only. Upload only a trusted .joblib model. Activated models are saved locally and reused after restart. Do not enable public uploads.')
model_file = st.sidebar.file_uploader('Upload trained model (.joblib)', type=['joblib'], key='model_upload')
if model_file is not None and st.sidebar.button('Activate Model', type='primary'):
    try:
        validated = load_uploaded_model(model_file)
        from pathlib import Path
        import os
        import tempfile
        model_dir = Path(__file__).resolve().parent / 'models'
        model_dir.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix='gym_model_', suffix='.joblib', dir=model_dir)
        try:
            with os.fdopen(fd, 'wb') as f:
                f.write(model_file.getvalue())
            os.replace(temp_path, model_dir / 'trained_model.joblib')
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        st.session_state['uploaded_model'] = validated
        st.sidebar.success('Model activated for this session!')
        st.rerun()
    except Exception as exc:
        st.sidebar.error(f'Unable to activate model: {exc}')
if load_model() is not None and st.sidebar.button('Remove active model'):
    st.session_state.pop('uploaded_model', None)
    from pathlib import Path
    saved = Path(__file__).resolve().parent / 'models' / 'trained_model.joblib'
    if saved.exists():
        saved.unlink()
    st.rerun()
