
import streamlit as st
import plotly.express as px

from utils.data_loader import load_data
from utils.analytics import summary


st.title("Executive Overview")

df = load_data()

st.caption(
    "Synthetic demonstration dataset — labels are observed labels, "
    "not model predictions."
)

# Filter risk category
selected = st.multiselect(
    "Risk category",
    sorted(df.risk_label.unique()),
    default=sorted(df.risk_label.unique())
)

df = df[df.risk_label.isin(selected)]

# Summary metrics
s = summary(df)

cols = st.columns(3)

metrics_1 = [
    ("Total Members", s["total"]),
    ("Active Members", s["active"]),
    ("At-Risk Members", s["at_risk"])
]

for col, (label, value) in zip(cols, metrics_1):
    col.metric(label, value)

cols = st.columns(3)

metrics_2 = [
    ("High-Priority Members (At Risk)", s["at_risk"]),
    ("Average Attendance", round(s["attendance"], 2)),
    ("Average Days Since Visit", round(s["days"], 2))
]

for col, (label, value) in zip(cols, metrics_2):
    col.metric(label, value)

# Handle empty filtered dataset
if df.empty:
    st.warning("No records match your filters.")
    st.stop()

# Risk distribution and customer activity charts
a, b = st.columns(2)

with a:
    st.plotly_chart(
        px.pie(
            df,
            names="risk_label",
            hole=0.6,
            title="Risk Distribution"
        ),
        width="stretch"
    )

    st.plotly_chart(
        px.histogram(
            df,
            x="attendance_frequency",
            color="risk_label",
            barmode="overlay",
            title="Attendance Frequency"
        ),
        width="stretch"
    )

with b:
    st.plotly_chart(
        px.box(
            df,
            x="risk_label",
            y="days_since_last_visit",
            title="Days Since Last Visit by Risk"
        ),
        width="stretch"
    )

    st.plotly_chart(
        px.box(
            df,
            x="risk_label",
            y="missed_session",
            title="Missed Sessions by Risk"
        ),
        width="stretch"
    )

# Customer engagement analysis
st.subheader("Customer Engagement Overview")

st.plotly_chart(
    px.scatter(
        df,
        x="attendance_frequency",
        y="days_since_last_visit",
        color="risk_label",
        hover_data=["member_id", "missed_session"],
        title="Attendance vs Recency"
    ),
    width="stretch"
)
