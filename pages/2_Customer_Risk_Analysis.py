
import streamlit as st
import pandas as pd

from utils.data_loader import load_data
from utils.model_loader import load_model, predict
from utils.recommendation_engine import recommend


st.title("Customer Risk Analysis")

# Load customer data and trained model
df = load_data().copy()
model = load_model()

# Generate model predictions
if model is not None:
    try:
        labels, probs = predict(model, df)
        df["predicted_risk_label"] = labels

        if probs is not None:
            df["prediction_confidence"] = probs.max(axis=1).round(3)

    except Exception as exc:
        st.warning(f"Prediction unavailable: {exc}")
else:
    st.info("No trained model loaded. Showing actual labels only.")

# Customer search and risk filters
search = st.text_input("Search Member ID")

risks = st.multiselect(
    "Actual risk labels",
    sorted(df.risk_label.unique()),
    default=sorted(df.risk_label.unique())
)

filtered = df[df.risk_label.isin(risks)]

if search:
    filtered = filtered[
        filtered.member_id.astype(str).str.contains(
            search,
            case=False,
            regex=False
        )
    ]

# Prioritize customers based on actual risk labels
order = {
    "At Risk": 0,
    "Need Attention": 1,
    "Active": 2
}

filtered = (
    filtered.assign(_priority=filtered.risk_label.map(order))
    .sort_values(
        ["_priority", "days_since_last_visit"],
        ascending=[True, False]
    )
    .drop(columns="_priority")
)

st.caption(
    "Priority ordering is a transparent business rule "
    "based on actual labels and visit recency, "
    "not a validated risk ranking."
)

# Display customer table
st.dataframe(
    filtered,
    width="stretch",
    hide_index=True
)

# Download filtered data
st.download_button(
    "Download filtered CSV",
    filtered.to_csv(index=False).encode("utf-8"),
    "gymnasyum_filtered.csv",
    "text/csv"
)

# Individual customer details
if len(filtered):
    member = st.selectbox(
        "Customer detail",
        filtered.member_id.astype(str).tolist()
    )

    row = filtered[
        filtered.member_id.astype(str) == member
    ].iloc[0]

    st.json(
        row.where(pd.notna(row), None).to_dict()
    )

    st.subheader("Recommended Retention Actions")

    st.dataframe(
        pd.DataFrame(recommend(row)),
        hide_index=True,
        width="stretch"
    )
