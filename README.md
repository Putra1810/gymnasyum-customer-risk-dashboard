# Gymnasyum Customer Risk Intelligence

Interactive Streamlit dashboard for **3,000 synthetic members**, not actual customers.

## Install and run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Export model from provided training notebook

Run the notebook's training cells through the export cell. It writes `gymnasyum_best_pipeline.joblib`, which contains preprocessing and the selected classifier. Copy that file to `models/trained_model.joblib`. Do **not** export only the estimator without preprocessing. Use the same compatible scikit-learn environment as training. Without the model, descriptive pages still work, while prediction is disabled.

Alternatively, at the end of the training notebook after `best_model.fit(...)`:

```python
import joblib
joblib.dump(best_model, 'trained_model.joblib')
```

Place the output in the `models/` folder. The app never trains a model at startup. The original notebook uses stratified 70/15/15 splits and selects between Logistic Regression and Random Forest on validation macro F1. Test metrics must be read from the notebook's exported JSON; this app does not invent them.

## Gemini setup

Copy `.env.example` to `.env` and set `GEMINI_API_KEY` to your own API key, or configure `GEMINI_API_KEY` in `.streamlit/secrets.toml`. The optional `GEMINI_MODEL` environment variable defaults to `gemini-2.5-flash`; change it to a model currently available to your API account. Do not commit API keys. API calls require internet and available quota.

## Example questions

- Berapa jumlah pelanggan yang berisiko tidak aktif?
- Apa karakteristik pelanggan dengan risiko tertinggi?
- Apa rekomendasi untuk member ORG-00001?
- Apa hubungan missed session dengan risiko pelanggan?
- Buatkan rencana tindakan customer retention berdasarkan data.

## Interpretation notes

Actual labels come from the CSV; predictions come only from the trained classifier; rules and Gemini output are management suggestions. The rule thresholds (14 days, 3 missed sessions, 2 packages) are configurable examples and **not validated cutoffs**. High-priority is defined as actual `At Risk` for dashboard counts, not a separate label. Observational associations are not causal explanations. The chatbot receives aggregate statistics and only explicitly requested member records, not the full CSV.


### Upload model in the UI
Open any dashboard page, use **Model Setup → Upload trained model (.joblib) → Activate Model** in the sidebar. The model is available in your current Streamlit session without manually renaming or copying it. Re-upload after restarting the server/session. Only upload trusted local artifacts; joblib deserialization can execute code. For multi-user/public deployments, disable or restrict this upload to authenticated administrators.
