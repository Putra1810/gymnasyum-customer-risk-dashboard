import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
def get_key():
    try: return os.getenv('GEMINI_API_KEY') or st.secrets.get('GEMINI_API_KEY')
    except Exception: return os.getenv('GEMINI_API_KEY')

def ask_gemini(question,context,history=None):
    key=get_key()
    if not key: raise RuntimeError('GEMINI_API_KEY is not configured. Add it to .env or Streamlit secrets.')
    from google import genai
    from google.genai import types
    client=genai.Client(api_key=key,http_options=types.HttpOptions(timeout=30000))
    instruction=('You are Gymnasyum AI Advisor. Respond in Indonesian. Dataset is synthetic. '
                 'Only use supplied aggregate statistics or member records for numerical claims. '
                 'Never invent records, accuracy, causes, financial ROI, or guaranteed outcomes. '
                 'Separate observed data from suggested actions. If data is missing, say so. '
                 'Give concise, actionable management recommendations.')
    transcript='\n'.join(f"{x['role']}: {x['content']}" for x in (history or [])[-6:])
    prompt=f'Context (JSON): {context}\nRecent conversation: {transcript}\nQuestion: {question}'
    try:
        response=client.models.generate_content(model=os.getenv('GEMINI_MODEL','gemini-2.5-flash'),contents=prompt,config=types.GenerateContentConfig(system_instruction=instruction,temperature=0.2,max_output_tokens=1200))
        return response.text or 'No response returned.'
    except Exception as exc:
        raise RuntimeError(f'Gemini request failed (check key, model, network or quota): {exc}') from exc
