import streamlit as st
import pandas as pd
from utils.data_loader import load_data
from utils.model_loader import load_model,predict
from utils.recommendation_engine import recommend
st.title('Customer Risk Analysis')
df=load_data().copy(); model=load_model()
if model is not None:
 try:
  labels,probs=predict(model,df);df['predicted_risk_label']=labels
  if probs is not None: df['prediction_confidence']=probs.max(axis=1).round(3)
 except Exception as exc: st.warning(f'Prediction unavailable: {exc}')
else: st.info('No trained model loaded. Showing actual labels only.')
search=st.text_input('Search Member ID')
risks=st.multiselect('Actual risk labels',sorted(df.risk_label.unique()),default=sorted(df.risk_label.unique()))
filtered=df[df.risk_label.isin(risks)]
if search: filtered=filtered[filtered.member_id.astype(str).str.contains(search,case=False,regex=False)]
order={'At Risk':0,'Need Attention':1,'Active':2}
filtered=filtered.assign(_priority=filtered.risk_label.map(order)).sort_values(['_priority','days_since_last_visit'],ascending=[True,False]).drop(columns='_priority')
st.caption('Priority ordering is a transparent business rule based on actual labels and visit recency, not a validated risk ranking.')
st.dataframe(filtered,width="stretch",hide_index=True)
st.download_button('Download filtered CSV',filtered.to_csv(index=False).encode('utf-8'),'gymnasyum_filtered.csv','text/csv')
if len(filtered):
 member=st.selectbox('Customer detail',filtered.member_id.astype(str).tolist())
 row=filtered[filtered.member_id.astype(str)==member].iloc[0]
 st.json(row.where(pd.notna(row),None).to_dict())
 st.dataframe(pd.DataFrame(recommend(row)),hide_index=True,width="stretch")

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
