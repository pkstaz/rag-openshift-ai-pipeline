# RAG OpenShift AI Pipeline

This project contains a RAG (Retrieval-Augmented Generation) processing pipeline for OpenShift AI, deployable as a Helm chart and orchestrated with Tekton Pipelines.

## 🚀 What does this pipeline do?
- Extracts text from documents stored in MinIO (PDF, DOCX, TXT)
- Fragments text into chunks
- Generates embeddings for each chunk using Sentence Transformers
- Saves embeddings to MinIO
- Indexes embeddings in an Elasticsearch cluster for semantic searches

## 📦 Deployment with Helm

### 1. Requirements
- OpenShift/Kubernetes with Tekton Pipelines installed
- Access to a MinIO cluster accessible from the pipeline
- Access to an Elasticsearch cluster accessible from the pipeline
- Helm 3.x

### 2. Main parameters (`values.yaml`)
```yaml
pipelineName: rag-tekton-pipeline
namespace: rag-openshift-ai
minioEndpoint: "<your-minio-endpoint>"
minioAccessKey: "<your-minio-access-key>"
minioSecretKey: "<your-minio-secret-key>"
bucketName: raw-documents
objectKey: test-document-openshift-ai.txt
chunkSize: "512"
chunkOverlap: "50"
elasticsearchEndpoint: "https://<your-elasticsearch-endpoint>"
elasticsearchUsername: "<your-elasticsearch-username>"
elasticsearchPassword: "<your-elasticsearch-password>"
```

### 3. Chart installation

```sh
helm install rag-pipeline . -n rag-openshift-ai --create-namespace \
  -f values.yaml
```

To update:
```sh
helm upgrade rag-pipeline . -n rag-openshift-ai -f values.yaml
```

### 4. Configurable parameters
- **MinIO**: endpoint, accessKey, input bucket and object to process
- **Chunking**: size and overlap of fragments
- **Elasticsearch**: endpoint, username and password

### 5. Chart structure
- `templates/tekton_rag_pipeline.yaml`: Main Tekton pipeline template
- `values.yaml`: Default and configurable values
- `Chart.yaml`: Chart metadata

---

# RAG OpenShift AI Pipeline

Este proyecto contiene un pipeline de procesamiento RAG (Retrieval-Augmented Generation) para OpenShift AI, desplegable como un Helm chart y orquestado con Tekton Pipelines.

## 🚀 ¿Qué hace este pipeline?
- Extrae texto de documentos almacenados en MinIO (PDF, DOCX, TXT)
- Fragmenta el texto en chunks
- Genera embeddings para cada chunk usando Sentence Transformers
- Guarda los embeddings en MinIO
- Indexa los embeddings en un clúster de Elasticsearch para búsquedas semánticas

## 📦 Despliegue con Helm

### 1. Requisitos
- OpenShift/Kubernetes con Tekton Pipelines instalado
- Acceso a un clúster MinIO accesible desde el pipeline
- Acceso a un clúster Elasticsearch accesible desde el pipeline
- Helm 3.x

### 2. Parámetros principales (`values.yaml`)
```yaml
pipelineName: rag-tekton-pipeline
namespace: rag-openshift-ai
minioEndpoint: "<your-minio-endpoint>"
minioAccessKey: "<your-minio-access-key>"
minioSecretKey: "<your-minio-secret-key>"
bucketName: raw-documents
objectKey: test-document-openshift-ai.txt
chunkSize: "512"
chunkOverlap: "50"
elasticsearchEndpoint: "https://<your-elasticsearch-endpoint>"
elasticsearchUsername: "<your-elasticsearch-username>"
elasticsearchPassword: "<your-elasticsearch-password>"
```

### 3. Instalación del chart

```sh
helm install rag-pipeline . -n rag-openshift-ai --create-namespace \
  -f values.yaml
```

Para actualizar:
```sh
helm upgrade rag-pipeline . -n rag-openshift-ai -f values.yaml
```

### 4. Parámetros configurables
- **MinIO**: endpoint, accessKey, bucket de entrada y objeto a procesar
- **Chunking**: tamaño y overlap de los fragmentos
- **Elasticsearch**: endpoint, usuario y password

### 5. Estructura del chart
- `templates/tekton_rag_pipeline.yaml`: Template principal del pipeline Tekton
- `values.yaml`: Valores por defecto y configurables
- `Chart.yaml`: Metadata del chart

---

> For more details about pipeline structure or troubleshooting, check the Tekton task logs and ensure connectivity between pods and MinIO and Elasticsearch services.

> Para más detalles sobre la estructura de los pipelines o troubleshooting, revisa los logs de las tasks en Tekton y asegúrate de la conectividad entre los pods y los servicios de MinIO y Elasticsearch.
