"""
RAG Pipeline Standalone - Versión que compila correctamente
Todos los components están definidos en el mismo archivo
"""

from kfp import dsl
from kfp.dsl import component, pipeline, Input, Output, Dataset

@component(
    base_image="pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel",
    packages_to_install=["PyPDF2==3.0.1", "python-docx==0.8.11", "minio==7.1.17", "chardet==5.2.0"]
)
def extract_text_component(
    bucket_name: str,
    object_key: str,
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    extracted_text: Output[Dataset],
    metadata: Output[Dataset]
):
    import os, json, tempfile
    from pathlib import Path
    from datetime import datetime
    from minio import Minio
    import PyPDF2
    from docx import Document
    import chardet

    minio_client = Minio(minio_endpoint, access_key=minio_access_key, secret_key=minio_secret_key, secure=False)

    with tempfile.TemporaryDirectory() as temp_dir:
        local_file_path = os.path.join(temp_dir, object_key.split('/')[-1])
        minio_client.fget_object(bucket_name, object_key, local_file_path)

        file_extension = Path(local_file_path).suffix.lower()
        file_size = os.path.getsize(local_file_path)
        extracted_content = ""

        if file_extension == '.pdf':
            with open(local_file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    extracted_content += page.extract_text() + "\n"
        elif file_extension == '.docx':
            doc = Document(local_file_path)
            for paragraph in doc.paragraphs:
                extracted_content += paragraph.text + "\n"
        elif file_extension in ['.txt', '.md']:
            with open(local_file_path, 'rb') as file:
                raw_data = file.read()
                encoding = chardet.detect(raw_data)['encoding']
            with open(local_file_path, 'r', encoding=encoding) as file:
                extracted_content = file.read()

        document_metadata = {
            "source_file": object_key,
            "file_type": file_extension,
            "file_size": file_size,
            "processed_at": datetime.now().isoformat(),
            "char_count": len(extracted_content),
            "word_count": len(extracted_content.split()),
            "bucket_name": bucket_name
        }

        with open(extracted_text.path, 'w', encoding='utf-8') as f:
            f.write(extracted_content)
        with open(metadata.path, 'w', encoding='utf-8') as f:
            json.dump(document_metadata, f, indent=2)

@component(
    base_image="pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel",
    packages_to_install=["tiktoken==0.5.1", "langchain==0.0.350"]
)
def chunk_text_component(
    extracted_text: Input[Dataset],
    metadata: Input[Dataset],
    chunk_size: int,
    chunk_overlap: int,
    chunks: Output[Dataset]
):
    import json, tiktoken
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    with open(extracted_text.path, 'r', encoding='utf-8') as f:
        text_content = f.read()
    with open(metadata.path, 'r', encoding='utf-8') as f:
        doc_metadata = json.load(f)

    encoding = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(encoding.encode(text))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size * 4,
        chunk_overlap=chunk_overlap * 4,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    text_chunks = text_splitter.split_text(text_content)
    processed_chunks = []

    for i, chunk_text in enumerate(text_chunks):
        token_count = count_tokens(chunk_text)
        chunk_metadata = {
            "chunk_id": f"{doc_metadata['source_file']}_chunk_{i:04d}",
            "chunk_index": i,
            "total_chunks": len(text_chunks),
            "text": chunk_text.strip(),
            "token_count": token_count,
            "char_count": len(chunk_text),
            "word_count": len(chunk_text.split()),
            "source_document": doc_metadata['source_file'],
            "file_type": doc_metadata['file_type'],
            "processed_at": doc_metadata['processed_at']
        }
        processed_chunks.append(chunk_metadata)

    processed_chunks = [chunk for chunk in processed_chunks if chunk['token_count'] >= 10]

    with open(chunks.path, 'w', encoding='utf-8') as f:
        json.dump(processed_chunks, f, indent=2, ensure_ascii=False)

@component(
    base_image="pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel",
    packages_to_install=["sentence-transformers==2.2.2", "numpy==1.24.3"]
)
def generate_embeddings_component(
    chunks: Input[Dataset],
    model_name: str,
    embeddings: Output[Dataset]
):
    import json, numpy as np
    from sentence_transformers import SentenceTransformer
    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(model_name, device=device)

    with open(chunks.path, 'r', encoding='utf-8') as f:
        chunk_data = json.load(f)

    texts = [chunk['text'] for chunk in chunk_data]
    all_embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    enriched_chunks = []
    for chunk, embedding in zip(chunk_data, all_embeddings):
        enriched_chunk = chunk.copy()
        enriched_chunk['embedding'] = embedding.tolist()
        enriched_chunk['embedding_dim'] = len(embedding)
        enriched_chunk['embedding_model'] = model_name
        enriched_chunks.append(enriched_chunk)

    with open(embeddings.path, 'w', encoding='utf-8') as f:
        json.dump(enriched_chunks, f, indent=2, ensure_ascii=False)

@component(
    base_image="pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel",
    packages_to_install=["elasticsearch==8.11.0"]
)
def index_elasticsearch_component(
    enriched_chunks: Input[Dataset],
    es_endpoint: str,
    es_index: str,
    index_status: Output[Dataset]
):
    import json
    from datetime import datetime
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk

    es = Elasticsearch([es_endpoint], verify_certs=False)

    with open(enriched_chunks.path, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)

    index_mapping = {
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "text": {"type": "text", "analyzer": "standard"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": chunks_data[0]['embedding_dim'] if chunks_data else 384,
                    "index": True,
                    "similarity": "cosine"
                },
                "source_document": {"type": "keyword"},
                "file_type": {"type": "keyword"},
                "processed_at": {"type": "date"},
                "indexed_at": {"type": "date"}
            }
        }
    }

    if not es.indices.exists(index=es_index):
        es.indices.create(index=es_index, body=index_mapping)

    documents = []
    for chunk in chunks_data:
        doc = {
            "_index": es_index,
            "_id": chunk['chunk_id'],
            "_source": {**chunk, "indexed_at": datetime.now().isoformat()}
        }
        documents.append(doc)

    success_count, failed_items = bulk(es, documents, chunk_size=100)

    indexing_status = {
        "index_name": es_index,
        "total_chunks": len(chunks_data),
        "indexed_chunks": success_count,
        "failed_chunks": len(failed_items) if failed_items else 0,
        "indexed_at": datetime.now().isoformat(),
        "success": len(failed_items) == 0 if failed_items else True
    }

    with open(index_status.path, 'w', encoding='utf-8') as f:
        json.dump(indexing_status, f, indent=2)

@pipeline(
    name="rag-document-processing-v1",
    description="RAG Document Processing Pipeline - Carlos Estay (pkstaz)"
)
def rag_document_pipeline(
    bucket_name: str = "raw-documents",
    object_key: str = "",
    minio_endpoint: str = "minio:9000",
    minio_access_key: str = "minio",
    minio_secret_key: str = "minio123",
    es_endpoint: str = "elasticsearch:9200",
    es_index: str = "rag-documents",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
):
    extract_task = extract_text_component(
        bucket_name=bucket_name,
        object_key=object_key,
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key
    )
    extract_task.set_display_name("📄 Extract Text").set_cpu_limit("500m").set_memory_limit("1Gi")

    chunk_task = chunk_text_component(
        extracted_text=extract_task.outputs['extracted_text'],
        metadata=extract_task.outputs['metadata'],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    ).after(extract_task)
    chunk_task.set_display_name("🧩 Chunk Text").set_cpu_limit("500m").set_memory_limit("1Gi")

    embedding_task = generate_embeddings_component(
        chunks=chunk_task.outputs['chunks'],
        model_name=embedding_model
    ).after(chunk_task)
    embedding_task.set_display_name("🎯 Generate Embeddings").set_cpu_limit("1000m").set_memory_limit("4Gi")

    index_task = index_elasticsearch_component(
        enriched_chunks=embedding_task.outputs['embeddings'],
        es_endpoint=es_endpoint,
        es_index=es_index
    ).after(embedding_task)
    index_task.set_display_name("🔍 Index ElasticSearch").set_cpu_limit("500m").set_memory_limit("2Gi")

    return index_task.outputs['index_status']
