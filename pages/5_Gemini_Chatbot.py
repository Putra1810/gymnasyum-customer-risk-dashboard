
import streamlit as st

from utils.data_loader import load_data
from utils.analytics import context
from utils.model_loader import load_model, predict
from utils.gemini_client import ask_gemini


st.title("💬 Gymnasyum AI Advisor")

# Load dataset and Machine Learning model
df = load_data()
model = load_model()

# Initialize chat history
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# Display previous messages
for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User question
question = st.chat_input(
    "Tanyakan data member, risiko, atau strategi retensi..."
)

if question:
    with st.chat_message("user"):
        st.markdown(question)

    # Prepare aggregated customer data
    ctx = context(df)

    # Only include member records explicitly mentioned
    # in the question, never the entire dataset.
    ids = df.member_id.astype(str)

    matching = [
        x for x in ids
        if x.lower() in question.lower()
    ]

    if matching:
        selected = df[
            df.member_id.astype(str).isin(matching[:5])
        ].copy()

        ctx["requested_members"] = selected.to_dict("records")

        # Generate risk predictions for selected members
        if model is not None:
            try:
                pred, prob = predict(model, selected)

                ctx["member_predictions"] = {
                    str(mid): {
                        "label": str(label),
                        "confidence": (
                            float(prob[i].max())
                            if prob is not None else None
                        )
                    }
                    for i, (mid, label) in enumerate(
                        zip(selected.member_id, pred)
                    )
                }

            except Exception:
                pass

    # Provide exact counts from the dataset
    ctx["exact_member_counts"] = (
        df.risk_label.value_counts().to_dict()
    )

    ctx["model_available"] = model is not None

    # Generate Gemini response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = ask_gemini(
                    question,
                    ctx,
                    st.session_state.chat_messages
                )

            except Exception as exc:
                answer = (
                    "Maaf, layanan Gemini sedang tidak tersedia. "
                    "Silakan coba beberapa saat lagi. "
                    f"Detail: {exc}"
                )

        st.markdown(answer)

    # Save chat history
    st.session_state.chat_messages.extend([
        {
            "role": "user",
            "content": question
        },
        {
            "role": "assistant",
            "content": answer
        }
    ])

# Clear chat history
if st.button("Clear Chat"):
    st.session_state.chat_messages = []
    st.rerun()
