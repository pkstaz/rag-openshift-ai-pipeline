"""
Vector Processing Components para RAG Pipeline
Incluye: generate_embeddings_component y index_elasticsearch_component
"""

from kfp.dsl import component, Input, Output, Dataset

@component(
    base_image="pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel",
    packages_to_install=[
        "sentence-transformers==2.2.2",
        "numpy==1.24.3"
    ]
)
def generate_embeddings_component(
    chunks: Input[Dataset],
    model_name: str,
    embeddings: Output[Dataset]
):
    """
    Genera embeddings vectoriales para los chunks de texto.

    Args:
        chunks: Input dataset con chunks de texto
        model_name: Nombre del modelo de embeddings
        embeddings: Output dataset con embeddings generados
    """
    import json
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import torch

    # Verificar si hay GPU disponible
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Usando device: {device}")

    # Cargar modelo de embeddings
    print(f"📥 Cargando modelo: {model_name}")
    model = SentenceTransformer(model_name, device=device)

    # Leer chunks
    with open(chunks.path, 'r', encoding='utf-8') as f:
        chunk_data = json.load(f)

    print(f"📝 Procesando {len(chunk_data)} chunks")

    # Extraer textos para embedding
    texts = [chunk['text'] for chunk in chunk_data]

    # Generar embeddings en batches para eficiencia
    batch_size = 32
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = model.encode(
            batch_texts,
            convert_to_numpy=True,
            show_progress_bar=True if i == 0 else False,
            normalize_embeddings=True
        )
        all_embeddings.extend(batch_embeddings)

        if i % (batch_size * 5) == 0:
            print(f"  Procesado: {min(i + batch_size, len(texts))}/{len(texts)} chunks")

    print(f"✅ Embeddings generados: {len(all_embeddings)} vectores de {len(all_embeddings[0])} dimensiones")

    # Combinar chunks con sus embeddings
    enriched_chunks = []
    for chunk, embedding in zip(chunk_data, all_embeddings):
        enriched_chunk = chunk.copy()
        enriched_chunk['embedding'] = embedding.tolist()
        enriched_chunk['embedding_dim'] = len(embedding)
        enriched_chunk['embedding_model'] = model_name
        enriched_chunks.append(enriched_chunk)

    # Guardar chunks enriquecidos con embeddings
    with open(embeddings.path, 'w', encoding='utf-8') as f:
        json.dump(enriched_chunks, f, indent=2, ensure_ascii=False)

    print(f"✅ Chunks enriquecidos guardados")


@component(
    base_image="pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel",
    packages_to_install=[
        "elasticsearch==8.11.0"
    ]
)
def index_elasticsearch_component(
    enriched_chunks: Input[Dataset],
    es_endpoint: str,
    es_index: str,
    index_status: Output[Dataset]
):
    """
    Indexa chunks enriquecidos en ElasticSearch.

    Args:
        enriched_chunks: Input dataset con chunks y embeddings
        es_endpoint: Endpoint de ElasticSearch
        es_index: Nombre del índice
        index_status: Output dataset con status de indexación
    """
    import json
    from datetime import datetime
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk

    # Conectar a ElasticSearch
    try:
        es = Elasticsearch([es_endpoint], verify_certs=False)

        if not es.ping():
            raise Exception("No se puede conectar a ElasticSearch")

        print(f"✅ Conectado a ElasticSearch: {es_endpoint}")
    except Exception as e:
        raise Exception(f"Error conectando a ElasticSearch: {str(e)}")

    # Leer chunks enriquecidos
    with open(enriched_chunks.path, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)

    print(f"📝 Indexando {len(chunks_data)} chunks en índice: {es_index}")

    # Definir mapping del índice
    index_mapping = {
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "text": {
                    "type": "text",
                    "analyzer": "standard"
                },
                "embedding": {
                    "type": "dense_vector",
                    "dims": chunks_data[0]['embedding_dim'] if chunks_data else 384,
                    "index": True,
                    "similarity": "cosine"
                },
                "source_document": {"type": "keyword"},
                "file_type": {"type": "keyword"},
                "chunk_index": {"type": "integer"},
                "total_chunks": {"type": "integer"},
                "token_count": {"type": "integer"},
                "char_count": {"type": "integer"},
                "word_count": {"type": "integer"},
                "processed_at": {"type": "date"},
                "indexed_at": {"type": "date"}
            }
        },
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        }
    }

    # Crear índice si no existe
    if not es.indices.exists(index=es_index):
        es.indices.create(index=es_index, body=index_mapping)
        print(f"✅ Índice creado: {es_index}")
    else:
        print(f"ℹ️ Índice ya existe: {es_index}")

    # Preparar documentos para bulk indexing
    documents = []
    for chunk in chunks_data:
        doc = {
            "_index": es_index,
            "_id": chunk['chunk_id'],
            "_source": {
                **chunk,
                "indexed_at": datetime.now().isoformat()
            }
        }
        documents.append(doc)

    # Indexar en batches
    try:
        success_count, failed_items = bulk(
            es,
            documents,
            chunk_size=100,
            request_timeout=300
        )

        print(f"✅ Indexación completada:")
        print(f"  Documentos exitosos: {success_count}")
        print(f"  Documentos fallidos: {len(failed_items) if failed_items else 0}")

    except Exception as e:
        raise Exception(f"Error en bulk indexing: {str(e)}")

    # Refresh del índice
    es.indices.refresh(index=es_index)

    # Verificar indexación
    doc_count = es.count(index=es_index)['count']
    print(f"✅ Total documentos en índice: {doc_count}")

    # Preparar status de indexación
    indexing_status = {
        "index_name": es_index,
        "total_chunks": len(chunks_data),
        "indexed_chunks": success_count,
        "failed_chunks": len(failed_items) if failed_items else 0,
        "total_documents_in_index": doc_count,
        "indexed_at": datetime.now().isoformat(),
        "success": len(failed_items) == 0 if failed_items else True
    }

    # Guardar status
    with open(index_status.path, 'w', encoding='utf-8') as f:
        json.dump(indexing_status, f, indent=2)

    print(f"✅ Status de indexación guardado")
