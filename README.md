# ScamGuard AI
ScamGuard AI is a multi-class NLP-based scam message detection system for student cybersecurity awareness. It classifies messages into **Safe**, **Phishing**, **OTP Scam**, **Lottery Scam**, or **Job Scam**.

The project includes a modular ML backend (TF-IDF + Logistic Regression), a prediction engine with probability + risk score (0–100), and a Streamlit UI with explainable outputs (suspicious keywords/phrases + human-readable reasons).

## Features
- Multi-class scam detection (TF-IDF + Logistic Regression)
- Probability + risk score (0–100) with color-coded risk levels
- Explainability panel (pattern/keyword detection)
- Dynamic safety tips based on scam type
- Mini dashboard (session stats)

## Quickstart (Windows / PowerShell)
1) Install dependencies:
```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

2) (Optional) Train and save a model:
```powershell
.\venv\Scripts\python.exe scripts\train.py
```

3) Run the Streamlit app:
```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

Note: If no saved model exists, the app will train a lightweight starter model from `data/sample_messages.csv` on first run and save it to `scamguard/artifacts/model.joblib`.
