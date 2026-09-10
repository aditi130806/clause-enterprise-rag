# Clause Enterprise RAG — Google Cloud Run Deployment Guide

This guide details how to deploy **Clause Enterprise RAG** to **Google Cloud Run** using **Google Cloud Shell** directly in your browser.

> [!NOTE]
> **No local Docker Desktop or local `gcloud` CLI installation is required.** Deployment is executed entirely within Google Cloud Shell via `gcloud run deploy --source .`.

---

## 1. Prerequisites

1. Active **Google Cloud Platform (GCP)** account with billing enabled.
2. Valid `GEMINI_API_KEY` from Google AI Studio.
3. Access to [Google Cloud Shell](https://shell.cloud.google.com).

---

## 2. Transferring Source Code to Google Cloud Shell

Choose one of the following options to bring your code into Cloud Shell:

### Option A — GitHub Clone (Recommended)
1. Push your repository to GitHub.
2. Open [Google Cloud Shell](https://shell.cloud.google.com).
3. Clone and navigate to the project directory:
   ```bash
   git clone https://github.com/YOUR_USERNAME/clause-enterprise-rag.git
   cd clause-enterprise-rag
   ```

### Option B — Upload ZIP File
1. Zip your local project directory (ensure `.env` and `.venv` are excluded).
2. Open [Google Cloud Shell](https://shell.cloud.google.com).
3. Click **More (3 dots)** in the top right menu -> **Upload file** -> select your `.zip` file.
4. Unzip and enter the project folder:
   ```bash
   unzip clause-enterprise-rag.zip
   cd clause-enterprise-rag
   ```

---

## 3. Secret Manager Setup (Storing GEMINI_API_KEY)

Never hardcode secrets into Docker images or environment variables. Store your API key securely in Google Secret Manager:

```bash
# 1. Set your active project ID
gcloud config set project YOUR_PROJECT_ID

# 2. Enable Secret Manager API
gcloud services enable secretmanager.googleapis.com

# 3. Create the secret container
printf '%s' 'YOUR_ACTUAL_GEMINI_API_KEY' | gcloud secrets create clause-gemini-key --data-file=-

# (Optional) Add a new version if the secret already exists:
# printf '%s' 'YOUR_ACTUAL_GEMINI_API_KEY' | gcloud secrets versions add clause-gemini-key --data-file=-

# 4. Grant Secret Accessor permission to the default Cloud Run service account
PROJECT_NUMBER=$(gcloud projects describe YOUR_PROJECT_ID --format='value(projectNumber)')

gcloud secrets add-iam-policy-binding clause-gemini-key \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

---

## 4. Deploying to Google Cloud Run

### Automated Script Deployment
Edit `deploy_cloud_run.sh` to set `YOUR_PROJECT_ID` and `YOUR_REGION` (e.g., `us-central1`), then run:

```bash
bash deploy_cloud_run.sh
```

### Manual Command Execution
Alternatively, run the single deployment command directly in Cloud Shell:

```bash
gcloud run deploy clause-enterprise-rag \
    --source . \
    --region us-central1 \
    --platform managed \
    --allow-unauthenticated \
    --port 8080 \
    --memory 2Gi \
    --cpu 2 \
    --concurrency 80 \
    --timeout 300 \
    --set-env-vars GEMINI_MODEL=gemini-3.8-flash,DEFAULT_LLM_TEMPERATURE=0.0 \
    --set-secrets GEMINI_API_KEY=clause-gemini-key:latest
```

---

## 5. Recommended Resource Settings

| Setting | Recommended Value | Rationale |
| :--- | :--- | :--- |
| **Memory** | `2Gi` | FastEmbed ONNX embedding models and FAISS vector indices require ~1.2-1.5GiB RAM in-memory. |
| **CPU** | `2` | Ensures fast query embedding generation and hybrid search execution. |
| **Port** | `8080` | Streamlit container default HTTP port binding. |
| **Concurrency** | `80` | Default concurrency limit for Streamlit web sessions. |
| **Timeout** | `300s` | Accommodates initial cold-start model weights loading. |

---

## 6. Accessing the Live Application

Upon completion, Cloud Run will output the HTTPS service URL:
```text
Service [clause-enterprise-rag] revision [clause-enterprise-rag-00001-xyz] has been deployed and is serving 100% of traffic.
Service URL: https://clause-enterprise-rag-xyz-uc.a.run.app
```

Open the generated `Service URL` in your web browser to access your live Clause Enterprise RAG instance!
