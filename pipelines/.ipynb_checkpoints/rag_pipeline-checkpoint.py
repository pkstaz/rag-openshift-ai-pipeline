"""
RAG Pipeline Principal - OpenShift AI Data Science Pipeline
Orquesta todos los components para procesamiento completo de documentos
"""

from kfp import dsl
from kfp.dsl import pipeline

# Importar components
from components.text_processing import extract_text_component, chunk_text_component
from components.vector_processing import generate_embeddings_component, index_elasticsearch_component

@pipeline(
    name="rag-document-processing-v1",
    description="Pipeline completo de procesamiento de documentos RAG para OpenShift AI"
)
def rag_document_pipeline(
    bucket_name: str = "raw-documents",
    object_key: str = "",
    minio_endpoint: str = "rag-documents:9000",
    minio_access_key: str = "minio",
    minio_secret_key: str = "minio123",
    es_endpoint: str = "elasticsearch:9200",
    es_index: str = "rag-documents",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
):
    """
    Pipeline completo de procesamiento de documentos RAG.

    Args:
        bucket_name: Nombre del bucket en MinIO
        object_key: Path del archivo a procesar
        minio_endpoint: Endpoint de MinIO
        minio_access_key: Access key de MinIO
        minio_secret_key: Secret key de MinIO
        es_endpoint: Endpoint de ElasticSearch
        es_index: Nombre del índice en ElasticSearch
        chunk_size: Tamaño de chunks en tokens
        chunk_overlap: Overlap entre chunks en tokens
        embedding_model: Modelo para generar embeddings

    Returns:
        Status de indexación final
    """

    # Step 1: Extract text from document
    extract_task = extract_text_component(
        bucket_name=bucket_name,
        object_key=object_key,
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key
    )
    extract_task.set_display_name("📄 Extract Text")
    extract_task.set_cpu_limit("500m")
    extract_task.set_memory_limit("1Gi")
    extract_task.set_retry(3)

    # Step 2: Chunk the extracted text
    chunk_task = chunk_text_component(
        extracted_text=extract_task.outputs['extracted_text'],
        metadata=extract_task.outputs['metadata'],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    ).after(extract_task)
    chunk_task.set_display_name("🧩 Chunk Text")
    chunk_task.set_cpu_limit("500m")
    chunk_task.set_memory_limit("1Gi")
    chunk_task.set_retry(3)

    # Step 3: Generate embeddings
    embedding_task = generate_embeddings_component(
        chunks=chunk_task.outputs['chunks'],
        model_name=embedding_model
    ).after(chunk_task)
    embedding_task.set_display_name("🎯 Generate Embeddings")
    embedding_task.set_cpu_limit("1000m")
    embedding_task.set_memory_limit("4Gi")
    embedding_task.set_retry(2)
    # embedding_task.set_gpu_limit("1")  # Uncomment si hay GPUs disponibles

    # Step 4: Index in ElasticSearch
    index_task = index_elasticsearch_component(
        enriched_chunks=embedding_task.outputs['embeddings'],
        es_endpoint=es_endpoint,
        es_index=es_index
    ).after(embedding_task)
    index_task.set_display_name("🔍 Index ElasticSearch")
    index_task.set_cpu_limit("500m")
    index_task.set_memory_limit("2Gi")
    index_task.set_retry(3)

    # Return final status
    return index_task.outputs['index_status']


# Pipeline alternativo para batch processing
@pipeline(
    name="rag-batch-processing-v1", 
    description="Pipeline para procesamiento batch de múltiples documentos"
)
def rag_batch_pipeline(
    bucket_name: str = "raw-documents",
    file_pattern: str = "*.pdf",
    es_endpoint: str = "elasticsearch:9200",
    es_index: str = "rag-documents",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 10
):
    """
    Pipeline para procesamiento batch de múltiples documentos.
    Útil para procesamiento inicial de grandes volúmenes.
    """
    # TODO: Implementar en futuras versiones
    # - Listar archivos en bucket por patrón
    # - Procesar en batches paralelos
    # - Consolidar resultados
    pass


if __name__ == "__main__":
    # Para testing local del pipeline definition
    print("✅ RAG Pipeline definido correctamente")
    print("📋 Pipeline functions disponibles:")
    print("  - rag_document_pipeline: Procesamiento individual")
    print("  - rag_batch_pipeline: Procesamiento batch (TODO)")
