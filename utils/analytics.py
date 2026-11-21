import pandas as pd
from utils.data_loader import LABELS

def summary(df):
    counts=df.risk_label.value_counts()
    return {'total':len(df),'active':int(counts.get('Active',0)),'attention':int(counts.get('Need Attention',0)),'at_risk':int(counts.get('At Risk',0)),'attendance':float(df.attendance_frequency.mean()) if len(df) else 0,'days':float(df.days_since_last_visit.mean()) if len(df) else 0}

def context(df):
    result={'total':len(df),'risk_counts':df.risk_label.value_counts().to_dict(),'by_risk':df.groupby('risk_label')[['attendance_frequency','missed_session','days_since_last_visit','reschedule_frequency','remaining_package','response_to_reminder']].mean().round(2).to_dict('index')}
    return result
