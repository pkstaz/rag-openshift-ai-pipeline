# RAG OpenShift AI Pipeline

Este proyecto contiene un pipeline de procesamiento RAG (Retrieval-Augmented Generation) para OpenShift AI, desplegable como un Helm chart y orquestado con Tekton Pipelines.

## 🚀 ¿Qué hace este pipeline?
- Extrae texto de documentos almacenados en MinIO
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
minioEndpoint: minio-api-poc-rag.apps.cluster-2gbhp.2gbhp.sandbox1120.opentlc.com
minioAccessKey: minio
minioSecretKey: minio123
bucketName: raw-documents
objectKey: test-document-openshift-ai.txt
chunkSize: "512"
chunkOverlap: "50"
elasticsearchEndpoint: "http://elasticsearch-es-default:9200"
elasticsearchUsername: "elastic"
elasticsearchPassword: "<tu-password>"
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
- **MinIO**: endpoint, accessKey, secretKey, bucket de entrada y objeto a procesar
- **Chunking**: tamaño y overlap de los fragmentos
- **Elasticsearch**: endpoint, usuario y password

### 5. Estructura del chart
- `templates/tekton_rag_pipeline.yaml`: Template principal del pipeline Tekton
- `values.yaml`: Valores por defecto y configurables
- `Chart.yaml`: Metadata del chart

---

> Para más detalles sobre la estructura de los pipelines o troubleshooting, revisa los logs de las tasks en Tekton y asegúrate de la conectividad entre los pods y los servicios de MinIO y Elasticsearch.
