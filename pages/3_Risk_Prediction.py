
import streamlit as st
import pandas as pd

from utils.data_loader import FEATURES, load_data
from utils.model_loader import load_model, predict
from utils.recommendation_engine import recommend
from utils.gemini_client import ask_gemini


st.title("Customer Risk Prediction")

# Load trained Machine Learning model
model = load_model()

if model is None:
    st.warning(
        "Prediction unavailable. "
        "The trained model could not be loaded."
    )
    st.stop()

# Load dataset for default input values
df = load_data()

# Customer risk prediction form
with st.form("risk_form"):
    values = {}

    for c in FEATURES:
        if c == "response_to_reminder":
            values[c] = st.selectbox(
                "Response to Reminder",
                [0, 1]
            )
        else:
            values[c] = st.number_input(
                c.replace("_", " ").title(),
                min_value=0,
                max_value=100000,
                value=int(df[c].median())
            )

    submit = st.form_submit_button(
        "Predict Customer Risk",
        type="primary"
    )

# Run prediction
if submit:
    frame = pd.DataFrame([values], columns=FEATURES)

    try:
        labels, probs = predict(model, frame)

        st.success(f"Predicted risk: {labels[0]}")

        # Display prediction probabilities
        if probs is not None:
            st.subheader("Model Class Probabilities")

            probability_df = pd.DataFrame({
                "Class": model.classes_,
                "Probability": probs[0]
            })

            st.dataframe(
                probability_df,
                hide_index=True
            )

        st.caption(
            "Model probabilities are estimates, not guaranteed "
            "real-world inactivity rates. Recommendations below "
            "are illustrative rules, not causal explanations."
        )

        # Rule-based customer retention recommendations
        st.subheader("Recommended Retention Actions")

        rules = recommend(values)

        st.dataframe(
            pd.DataFrame(rules),
            hide_index=True,
            width="stretch"
        )

        # Save prediction results for Gemini
        st.session_state["last_prediction"] = {
            "features": values,
            "predicted_label": str(labels[0]),
            "probabilities": (
                dict(zip(
                    map(str, model.classes_),
                    map(float, probs[0])
                ))
                if probs is not None else None
            ),
            "rules": rules
        }

    except Exception as exc:
        st.error(f"Prediction error: {exc}")


# Optional Gemini recommendation
if (
    "last_prediction" in st.session_state
    and st.button("Generate Gemini Recommendation")
):
    with st.spinner("Generating recommendations..."):
        try:
            response = ask_gemini(
                "Berikan ringkasan risiko, faktor yang mungkin "
                "terkait, tindakan retensi, pendekatan komunikasi, "
                "prioritas follow-up, manfaat yang diharapkan, "
                "dan KPI.",
                st.session_state["last_prediction"]
            )

            st.markdown(response)

        except Exception as exc:
            st.error(f"Gemini request failed: {exc}")
