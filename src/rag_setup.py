# src/rag_setup.py
import os
from src.pdf_reader import extract_pdf_text
from src.chunker import create_chunks
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.rag import RAG
from src.storage import save_chunks, load_chunks
from src.ollama_client import OllamaClient


def get_rag(pdf_path: str):

    vectorstore_dir = "vectorstore"
    
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    index_path = f"{vectorstore_dir}/{pdf_name}.index"

    chunks_path = f"{vectorstore_dir}/{pdf_name}.json"

    os.makedirs(
        vectorstore_dir,
        exist_ok=True
    )

    embedder = Embedder()

    if os.path.exists(index_path) and os.path.exists(chunks_path):

        print("Loading existing index...")

        chunks = load_chunks(
            chunks_path
        )

        vector_store = VectorStore.load(
            index_path
        )

    else:

        print("Building index...")

        pages = extract_pdf_text(
            pdf_path
        )

        chunks = create_chunks(
            pages
        )

        embeddings = embedder.create_embeddings(
            chunks
        )

        vector_store = VectorStore(
            embeddings.shape[1]
        )

        vector_store.add_embeddings(
            embeddings
        )

        save_chunks(
            chunks,
            chunks_path
        )

        vector_store.save(
            index_path
        )

    retriever = Retriever(
        embedder,
        vector_store,
        chunks
    )

    llm = OllamaClient()

    rag = RAG(
        retriever,
        llm
    )

    return rag