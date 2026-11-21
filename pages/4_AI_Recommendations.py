import streamlit as st
import pandas as pd
from utils.data_loader import load_data
from utils.analytics import context
from utils.recommendation_engine import recommend
from utils.gemini_client import ask_gemini
st.title('AI Recommendation Center')
df=load_data()
level=st.radio('Recommendation level',['Strategic','Operational','Individual'],horizontal=True)
ctx=context(df)
if level=='Individual':
 member=st.selectbox('Member ID',df.member_id.astype(str).tolist())
 row=df[df.member_id.astype(str)==member].iloc[0]
 ctx={'member':row.to_dict(),'rules':recommend(row)}
 st.dataframe(pd.DataFrame(ctx['rules']),hide_index=True,width="stretch")
else:
 st.json(ctx)
if st.button('Generate AI recommendations',type='primary'):
 prompt=f'Create {level.lower()} retention recommendations. For every recommendation include Identified Issue, Supporting Data, Recommended Action, Priority Level, Expected Business Benefit, and Suggested KPI to Monitor. No unsupported causal or financial claims.'
 with st.spinner('Consulting Gemini...'):
  try: st.markdown(ask_gemini(prompt,ctx))
  except Exception as exc: st.error(str(exc))

st.sidebar.divider()
st.sidebar.subheader('⚙️ Model Setup')
from utils.model_loader import load_model, load_uploaded_model
if load_model() is not None:
    st.sidebar.success('Model ready for predictions')
else:
    st.sidebar.info('No model loaded yet')
st.sidebar.caption('For local/admin use only. Upload ONLY a .joblib file you created or trust. Never upload untrusted models.')
model_file = st.sidebar.file_uploader('Upload trained model (.joblib)', type=['joblib'], key='model_upload')
if model_file is not None and st.sidebar.button('Activate Model', type='primary'):
    try:
        st.session_state['uploaded_model'] = load_uploaded_model(model_file)
        st.sidebar.success('Model activated for this session!')
        st.rerun()
    except Exception as exc:
        st.sidebar.error(f'Unable to activate model: {exc}')
if 'uploaded_model' in st.session_state and st.sidebar.button('Remove uploaded model'):
    del st.session_state['uploaded_model']
    st.rerun()
