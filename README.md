# MLP-Deployment

Run model training from the repository root:

```bash
python -m src.train
```

Start the applications from the repository root:

```bash
mlflow ui
streamlit run app/streamlit_app.py
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```