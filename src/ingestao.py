"""
Constrói o índice vetorial (FAISS) a partir dos documentos em documentos/.

Uso:
    python src/ingestao.py
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

RAIZ = Path(__file__).resolve().parent.parent
PASTA_DOCUMENTOS = RAIZ / "documentos"
PASTA_VECTORSTORE = RAIZ / "vectorstore"

MODELO_EMBEDDING = "models/text-embedding-004"


def carregar_documentos() -> list:
    """Lê todos os PDFs de documentos/ e retorna a lista de páginas carregadas."""
    documentos = []
    for caminho_pdf in sorted(PASTA_DOCUMENTOS.glob("*.pdf")):
        documentos.extend(PyPDFLoader(str(caminho_pdf)).load())
    return documentos


def dividir_em_chunks(documentos: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    return splitter.split_documents(documentos)


def construir_indice(chunks: list) -> FAISS:
    embeddings = GoogleGenerativeAIEmbeddings(model=MODELO_EMBEDDING)
    return FAISS.from_documents(chunks, embeddings)


def main():
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY não encontrada. Copie .env.example para .env e preencha sua chave."
        )

    documentos = carregar_documentos()
    if not documentos:
        raise RuntimeError(f"Nenhum PDF encontrado em {PASTA_DOCUMENTOS}")
    print(f"{len(documentos)} página(s) carregada(s) de {PASTA_DOCUMENTOS}")

    chunks = dividir_em_chunks(documentos)
    print(f"{len(chunks)} chunk(s) gerado(s)")

    indice = construir_indice(chunks)
    indice.save_local(str(PASTA_VECTORSTORE))
    print(f"Índice salvo em {PASTA_VECTORSTORE}")


if __name__ == "__main__":
    main()
