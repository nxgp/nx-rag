#!/usr/bin/env bash
# ==============================================================================
# Helper Script: Presign -> Upload -> Ingest Document into Mentera RAG Pipeline
# ==============================================================================

set -euo pipefail

FILE_PATH="${1:-$HOME/Downloads/PT-1003_ MEDSPA PATIENT TREATMENT AGREEMENT(Contract) AND INFORMED CONSENT.pdf}"
TENANT_ID="${2:-tenant123}"
PROVIDER_ID="${3:-medspa_provider}"
BASE_URL="${4:-http://localhost:8000}"

if [ ! -f "$FILE_PATH" ]; then
  echo "❌ Error: File not found at path: $FILE_PATH"
  exit 1
fi

FILENAME=$(basename "$FILE_PATH")

echo "======================================================================"
echo "🚀 Mentera RAG Direct Upload & Ingestion Pipeline Automation"
echo "======================================================================"
echo "📄 File:       $FILE_PATH"
echo "🏢 Tenant:     $TENANT_ID"
echo "🩺 Provider:   $PROVIDER_ID"
echo "======================================================================"

# ------------------------------------------------------------------------------
# Step 1: Generate Presigned Upload URL
# ------------------------------------------------------------------------------
echo "🔹 Step 1: Generating presigned upload URL..."
PRESIGN_RES=$(curl -s -X POST "$BASE_URL/upload/presign" \
  -H "Content-Type: application/json" \
  -d "{
    \"filename\": \"$FILENAME\",
    \"tenant_id\": \"$TENANT_ID\",
    \"provider_id\": \"$PROVIDER_ID\",
    \"file_type\": \"pdf\",
    \"content_type\": \"application/pdf\"
  }")

UPLOAD_URL=$(echo "$PRESIGN_RES" | python3 -c "import sys, json; print(json.load(sys.stdin).get('upload_url', ''))")
STORAGE_KEY=$(echo "$PRESIGN_RES" | python3 -c "import sys, json; print(json.load(sys.stdin).get('storage_key', ''))")

if [ -z "$UPLOAD_URL" ] || [ -z "$STORAGE_KEY" ]; then
  echo "❌ Presigned URL generation failed:"
  echo "$PRESIGN_RES"
  exit 1
fi

echo "  ✅ Presigned URL created."
echo "  🔑 Storage Key: $STORAGE_KEY"

# ------------------------------------------------------------------------------
# Step 2: Upload File Bytes Direct to Storage
# ------------------------------------------------------------------------------
echo ""
echo "🔹 Step 2: Uploading file bytes to S3..."
UPLOAD_HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --location --request PUT "$UPLOAD_URL" \
  --header "Content-Type: application/pdf" \
  --data-binary @"$FILE_PATH")

if [ "$UPLOAD_HTTP_CODE" -ne 200 ] && [ "$UPLOAD_HTTP_CODE" -ne 201 ]; then
  echo "❌ Upload failed with HTTP status code: $UPLOAD_HTTP_CODE"
  echo "   (Make sure the uvicorn server has loaded your exported AWS credentials)"
  exit 1
fi

echo "  ✅ Binary payload successfully uploaded (HTTP $UPLOAD_HTTP_CODE)."

# ------------------------------------------------------------------------------
# Step 3: Trigger Ingestion & Vector Indexing
# ------------------------------------------------------------------------------
echo ""
echo "🔹 Step 3: Triggering ingestion pipeline..."
INGEST_RES=$(curl -s -X POST "$BASE_URL/ingest" \
  -H "Content-Type: application/json" \
  -d "{
    \"storage_key\": \"$STORAGE_KEY\",
    \"tenant_id\": \"$TENANT_ID\",
    \"provider_id\": \"$PROVIDER_ID\",
    \"document_type\": \"pdf\",
    \"chunk_strategy\": \"recursive\"
  }")

echo "======================================================================"
echo "$INGEST_RES" | python3 -m json.tool
