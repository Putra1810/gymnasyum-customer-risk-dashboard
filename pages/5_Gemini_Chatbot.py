import streamlit as st
from utils.data_loader import load_data
from utils.analytics import context
from utils.model_loader import load_model,predict
from utils.gemini_client import ask_gemini
st.title('💬 Gymnasyum AI Advisor')
df=load_data();model=load_model()
if 'chat_messages' not in st.session_state: st.session_state.chat_messages=[]
for message in st.session_state.chat_messages:
 with st.chat_message(message['role']): st.markdown(message['content'])
question=st.chat_input('Tanyakan data member, risiko, atau strategi retensi...')
if question:
 with st.chat_message('user'): st.markdown(question)
 ctx=context(df)
 # Only include matching member records. Never send all member rows to the API.
 ids=df.member_id.astype(str)
 matching=[x for x in ids if x.lower() in question.lower()]
 if matching:
  selected=df[df.member_id.astype(str).isin(matching[:5])].copy()
  ctx['requested_members']=selected.to_dict('records')
  if model is not None:
   try:
    pred,prob=predict(model,selected)
    ctx['member_predictions']={str(mid):{'label':str(label),'confidence':float(prob[i].max()) if prob is not None else None} for i,(mid,label) in enumerate(zip(selected.member_id,pred))}
   except Exception: pass
 # Exact counts are computed by Pandas and provided as structured facts.
 ctx['exact_member_counts']=df.risk_label.value_counts().to_dict()
 ctx['model_available']=model is not None
 with st.chat_message('assistant'):
  with st.spinner('Thinking...'):
   try: answer=ask_gemini(question,ctx,st.session_state.chat_messages)
   except Exception as exc: answer=str(exc)
  st.markdown(answer)
 st.session_state.chat_messages.extend([{'role':'user','content':question},{'role':'assistant','content':answer}])
if st.button('Clear chat'): st.session_state.chat_messages=[];st.rerun()

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
