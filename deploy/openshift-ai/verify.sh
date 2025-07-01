#!/bin/bash
# OpenShift AI DSP Verification Script
# Author: Carlos Estay (pkstaz)

echo "🔍 Verifying OpenShift AI DSP deployment..."

# Check namespace
echo "📦 Checking namespace..."
oc get namespace rag-openshift-ai

# Check DSP Application
echo "🔬 Checking Data Science Pipeline Application..."
oc get datasciencepipelinesapplications -n rag-openshift-ai

# Check pods
echo "🚀 Checking pods..."
oc get pods -n rag-openshift-ai

# Check secrets
echo "🔑 Checking secrets..."
oc get secrets -n rag-openshift-ai

# Check MinIO connectivity
echo "🗄️ Testing MinIO connectivity..."
MINIO_POD=$(oc get pods -n rag-openshift-ai -l app=minio -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
if [ ! -z "$MINIO_POD" ]; then
    echo "✅ MinIO pod found: $MINIO_POD"
else
    echo "⚠️ MinIO pod not found in DSP namespace (may be external)"
fi

# Show pipeline UI URL
echo "🌐 Pipeline UI should be available through OpenShift AI Dashboard"
echo "   Navigate to: Data Science Projects > rag-openshift-ai > Pipelines"

echo "✅ Verification completed!"
