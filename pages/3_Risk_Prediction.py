import streamlit as st
import pandas as pd
from utils.data_loader import FEATURES,load_data
from utils.model_loader import load_model,predict
from utils.recommendation_engine import recommend
from utils.gemini_client import ask_gemini
st.title('Customer Risk Prediction')
model=load_model()
if model is None: st.warning('Prediction disabled: export the notebook pipeline to models/trained_model.joblib first.');st.stop()
df=load_data()
with st.form('risk_form'):
 values={}
 for c in FEATURES:
  if c=='response_to_reminder': values[c]=st.selectbox('Response to Reminder',[0,1])
  else: values[c]=st.number_input(c.replace('_',' ').title(),min_value=0,max_value=100000,value=int(df[c].median()))
 submit=st.form_submit_button('Predict Customer Risk',type='primary')
if submit:
 frame=pd.DataFrame([values],columns=FEATURES)
 try:
  labels,probs=predict(model,frame)
  st.success(f'Predicted risk: {labels[0]}')
  if probs is not None:
   st.subheader('Model class probabilities')
   st.dataframe(pd.DataFrame({'Class':model.classes_,'Probability':probs[0]}),hide_index=True)
  st.caption('Model probabilities are estimates, not guaranteed real-world inactivity rates. The rules below are illustrative, not causal explanations of the classifier.')
  rules=recommend(values)
  st.dataframe(pd.DataFrame(rules),hide_index=True,width="stretch")
  st.session_state['last_prediction']={'features':values,'predicted_label':str(labels[0]),'probabilities':dict(zip(map(str,model.classes_),map(float,probs[0]))) if probs is not None else None,'rules':rules}
 except Exception as exc: st.error(f'Prediction error: {exc}')
if 'last_prediction' in st.session_state and st.button('Generate Gemini recommendation'):
 with st.spinner('Generating recommendations...'):
  try: st.markdown(ask_gemini('Berikan ringkasan risiko, faktor yang mungkin terkait, tindakan retensi, pendekatan komunikasi, prioritas follow-up, manfaat yang diharapkan, dan KPI.',st.session_state['last_prediction']))
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
