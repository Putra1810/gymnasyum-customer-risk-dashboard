def recommend(record):
    items=[]
    if record['days_since_last_visit']>=14: items.append(('Long time since last visit',f"{record['days_since_last_visit']} days",'Offer a flexible return-to-gym scheduling check-in','High','Potential re-engagement','Return visit within 14 days'))
    if record['missed_session']>=3: items.append(('Repeated missed sessions',f"{record['missed_session']} missed",'Ask about scheduling barriers and offer booking assistance','High','Potential attendance recovery','Missed sessions per month'))
    if record['response_to_reminder']==0: items.append(('No reminder response','Response indicator: 0','Test a different communication time or channel with consent','Medium','Potentially better outreach response','Reminder response rate'))
    if record['remaining_package']<=2: items.append(('Low remaining package',f"{record['remaining_package']} remaining",'Offer a package review without pressuring the member','Medium','Potential renewal engagement','Package renewal rate'))
    if not items: items.append(('No configured rule triggered','Current feature values','Maintain routine engagement and monitor attendance','Low','Continued engagement','Attendance frequency'))
    return [{'Identified Issue':a,'Supporting Data':b,'Recommended Action':c,'Priority Level':d,'Expected Business Benefit':e,'Suggested KPI to Monitor':f} for a,b,c,d,e,f in items]
