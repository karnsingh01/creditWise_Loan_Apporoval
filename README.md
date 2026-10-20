# CreditWise Loan Approval — Deployment

## Files
- `app.py`: Streamlit app and training pipeline
- `requirements.txt`: Python dependencies
- `loan_approval_data.csv`: **Add your dataset file here** (not included because it was not attached)

## Run locally
```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Create a GitHub repository (for example `CreditWise-Loan-Approval`).
2. Upload `app.py`, `requirements.txt`, and `loan_approval_data.csv`.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Choose **Create app**, select your repository and branch, set the main file path to `app.py`, and deploy.
5. Share the generated public app URL.

**Privacy note:** Do not upload real applicant personal/financial information to a public repository. Use a synthetic/demo dataset. The app trains the model from the CSV when it starts and reports held-out test accuracy; it is a portfolio demo, not a production credit decision system.
