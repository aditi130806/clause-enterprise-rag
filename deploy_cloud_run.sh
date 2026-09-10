#!/usr/bin/env bash
# ==============================================================================
# Clause Enterprise RAG — Google Cloud Shell Automated Deployment Script
# ==============================================================================
# Run this script directly inside Google Cloud Shell in your browser.
#
# Prerequisites in Cloud Shell:
# 1. Create or select your active GCP project.
# 2. Store your Gemini API key in Secret Manager (see Secret Manager section below).
# 3. Execute: bash deploy_cloud_run.sh
# ==============================================================================

set -euo pipefail

# Deployment Configuration Placeholders
PROJECT_ID="YOUR_PROJECT_ID"
REGION="YOUR_REGION"
SERVICE_NAME="clause-enterprise-rag"
SECRET_NAME="clause-gemini-key"

echo "------------------------------------------------------------"
echo "Starting Cloud Run Deployment for ${SERVICE_NAME}"
echo "------------------------------------------------------------"

# 1. Set Active GCP Project & Default Region
echo "[1/5] Setting active project: ${PROJECT_ID}..."
gcloud config set project "${PROJECT_ID}"
gcloud config set run/region "${REGION}"

# 2. Enable Necessary GCP Cloud APIs
echo "[2/5] Enabling required Google Cloud Service APIs..."
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    secretmanager.googleapis.com

# 3. Check / Document Secret Manager Integration
echo "[3/5] Verifying Secret Manager setup..."
echo "Note: Ensure '${SECRET_NAME}' exists before deploying."
echo "If not yet created, run:"
echo "  printf '%s' 'YOUR_REAL_GEMINI_API_KEY' | gcloud secrets create ${SECRET_NAME} --data-file=-"
echo ""

# 4. Deploy directly from source directory using Cloud Build & Cloud Run
echo "[4/5] Building container and deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
    --source . \
    --region "${REGION}" \
    --platform managed \
    --allow-unauthenticated \
    --port 8080 \
    --memory 2Gi \
    --cpu 2 \
    --concurrency 80 \
    --timeout 300 \
    --set-env-vars GEMINI_MODEL=gemini-3.8-flash,DEFAULT_LLM_TEMPERATURE=0.0 \
    --set-secrets GEMINI_API_KEY=${SECRET_NAME}:latest

# 5. Retrieve Public Service Endpoint URL
echo "[5/5] Fetching live service URL..."
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" --format 'value(status.url)')

echo "============================================================"
echo "DEPLOYMENT SUCCESSFUL!"
echo "Clause Enterprise RAG Service URL:"
echo "${SERVICE_URL}"
echo "============================================================"
