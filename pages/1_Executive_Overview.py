import streamlit as st
import plotly.express as px
from utils.data_loader import load_data
from utils.analytics import summary
st.title('Executive Overview')
df=load_data()
st.caption('Synthetic demonstration dataset — labels are observed labels, not model predictions.')
selected=st.multiselect('Risk category',sorted(df.risk_label.unique()),default=sorted(df.risk_label.unique()))
df=df[df.risk_label.isin(selected)]
s=summary(df)
cols=st.columns(3)
for col,(label,value) in zip(cols,[('Total Members',s['total']),('Active Members',s['active']),('At-Risk Members',s['at_risk'])]): col.metric(label,value)
cols=st.columns(3)
for col,(label,value) in zip(cols,[('High-Priority Members (At Risk)',s['at_risk']),('Average Attendance',round(s['attendance'],2)),('Average Days Since Visit',round(s['days'],2))]): col.metric(label,value)
if df.empty: st.warning('No records match your filters.');st.stop()
a,b=st.columns(2)
with a:
 st.plotly_chart(px.pie(df,names='risk_label',hole=.6,title='Risk distribution'),width="stretch")
 st.plotly_chart(px.histogram(df,x='attendance_frequency',color='risk_label',barmode='overlay',title='Attendance frequency'),width="stretch")
with b:
 st.plotly_chart(px.box(df,x='risk_label',y='days_since_last_visit',title='Days since last visit by risk'),width="stretch")
 st.plotly_chart(px.box(df,x='risk_label',y='missed_session',title='Missed sessions by risk'),width="stretch")
st.subheader('Customer engagement overview')
st.plotly_chart(px.scatter(df,x='attendance_frequency',y='days_since_last_visit',color='risk_label',hover_data=['member_id','missed_session'],title='Attendance vs recency'),width="stretch")

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
