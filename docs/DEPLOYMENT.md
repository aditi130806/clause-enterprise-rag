# Clause Enterprise RAG — Deployment Guide

This guide details how to deploy **Clause Enterprise RAG** to cloud hosting environments.

---

## Primary Deployment Method: Streamlit Community Cloud (Recommended)

Deploy Clause directly from GitHub to **Streamlit Community Cloud** with zero cloud billing requirements.

### Deployment Parameters
- **GitHub Repository**: `aditi130806/clause-enterprise-rag`
- **Branch**: `main`
- **Main file path**: `app.py`

### Step-by-Step Deployment Instructions

1. **Sign in to Streamlit Community Cloud**:
   Navigate to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.

2. **Deploy New App**:
   - Click **"New app"** -> **"Use existing repo"**.
   - Select Repository: `aditi130806/clause-enterprise-rag`.
   - Select Branch: `main`.
   - Set Main file path: `app.py`.

3. **Configure App Secrets & Environment Variables**:
   Before clicking Deploy, click **"Advanced settings..."** -> **"Secrets"** and enter your credentials:
   ```toml
   GEMINI_API_KEY = "your-actual-gemini-api-key-here"
   GEMINI_MODEL = "gemini-3.8-flash"
   ```

4. **Launch Application**:
   Click **"Deploy!"**. Streamlit Community Cloud will automatically build the environment from `requirements.txt`, initialize the FAISS vectorstore and demo corpus, and launch your live application at a public URL (e.g. `https://clause-enterprise-rag.streamlit.app`).

---

## Alternative Deployment Method: Google Cloud Run (Optional Container Deployment)

For enterprise container deployments with dedicated infrastructure on Google Cloud Platform:

### Prerequisites
1. Active GCP account with billing enabled.
2. Valid `GEMINI_API_KEY` stored in Google Secret Manager.
3. Access to [Google Cloud Shell](https://shell.cloud.google.com).

### Quick Deployment via Google Cloud Shell
No local Docker or `gcloud` installation required. In Google Cloud Shell:

```bash
# 1. Clone repository
git clone https://github.com/aditi130806/clause-enterprise-rag.git
cd clause-enterprise-rag

# 2. Store Gemini key in Secret Manager
gcloud config set project YOUR_PROJECT_ID
gcloud services enable secretmanager.googleapis.com
printf '%s' 'YOUR_REAL_GEMINI_API_KEY' | gcloud secrets create clause-gemini-key --data-file=-

# 3. Deploy to Cloud Run
bash deploy_cloud_run.sh
```

---

## Resource Requirements & Runtime Notes

| Environment | Recommended Memory | Key Feature |
| :--- | :--- | :--- |
| **Streamlit Community Cloud** | Free Tier (~1-3GB) | Zero billing setup, direct GitHub integration, automatic SSL. |
| **Google Cloud Run** | `2Gi` RAM, `2` vCPU | Dedicated container scaling, enterprise SLA, custom domains. |

Both environments automatically bundle the 6-document demo policy corpus and initialize FAISS vector search in-memory.
