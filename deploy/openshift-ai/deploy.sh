#!/bin/bash
# OpenShift AI DSP Deployment Script
# Author: Carlos Estay (pkstaz)

set -e

echo "🚀 Deploying RAG Pipeline to OpenShift AI..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if oc is available
if ! command -v oc &> /dev/null; then
    echo -e "${RED}❌ oc command not found. Please install OpenShift CLI${NC}"
    exit 1
fi

# Check if logged in to OpenShift
if ! oc whoami &> /dev/null; then
    echo -e "${RED}❌ Not logged in to OpenShift. Please login first${NC}"
    exit 1
fi

echo -e "${GREEN}✅ OpenShift CLI available and logged in${NC}"

# Deploy namespace
echo -e "${YELLOW}📦 Creating namespace...${NC}"
oc apply -f deploy/openshift-ai/namespace.yaml

# Deploy RBAC
echo -e "${YELLOW}🔐 Creating RBAC...${NC}"
oc apply -f deploy/openshift-ai/service-account.yaml
oc apply -f deploy/openshift-ai/cluster-role.yaml
oc apply -f deploy/openshift-ai/cluster-role-binding.yaml

# Deploy MinIO secret
echo -e "${YELLOW}🔑 Creating MinIO secret...${NC}"
oc apply -f deploy/openshift-ai/minio-secret.yaml

# Deploy Data Science Pipeline Application
echo -e "${YELLOW}🔬 Creating Data Science Pipeline Application...${NC}"
oc apply -f deploy/openshift-ai/dsp-config.yaml

# Wait for DSP to be ready
echo -e "${YELLOW}⏳ Waiting for DSP to be ready...${NC}"
oc wait --for=condition=Ready datasciencepipelinesapplication/rag-pipeline-dsp -n rag-openshift-ai --timeout=300s

# Check status
echo -e "${YELLOW}📊 Checking DSP status...${NC}"
oc get datasciencepipelinesapplications -n rag-openshift-ai
oc get pods -n rag-openshift-ai

echo -e "${GREEN}✅ OpenShift AI DSP deployed successfully!${NC}"
echo -e "${GREEN}🌐 Access the pipeline UI through OpenShift AI Dashboard${NC}"

# Show next steps
echo -e "${YELLOW}📋 Next Steps:${NC}"
echo "1. Access OpenShift AI Dashboard"
echo "2. Navigate to Data Science Pipelines"
echo "3. Upload the compiled pipeline: rag_simple_pipeline_v1.yaml"
echo "4. Create an experiment and run the pipeline"
echo "5. Monitor execution in the pipeline UI"

echo -e "${GREEN}🎉 Deployment completed!${NC}"
