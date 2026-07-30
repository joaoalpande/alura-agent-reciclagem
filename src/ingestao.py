"""
Constrói o índice vetorial (FAISS) a partir dos documentos em documentos/.

Uso:
    python src/ingestao.py
"""
import io
import os
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

try:
    from src.embeddings import GoogleGenerativeAIEmbeddingsComRetry
except ImportError:
    # Executado como script direto (`python src/ingestao.py`): a raiz do projeto não
    # está no sys.path, só a pasta src/ — importa sem o prefixo "src.".
    from embeddings import GoogleGenerativeAIEmbeddingsComRetry

load_dotenv()

RAIZ = Path(__file__).resolve().parent.parent
PASTA_DOCUMENTOS = RAIZ / "documentos"
PASTA_VECTORSTORE = RAIZ / "vectorstore"

MODELO_EMBEDDING = "models/gemini-embedding-001"


def listar_pdfs(pasta: Path) -> list:
    """Lista os PDFs de uma pasta comparando a extensão sem diferenciar maiúsculas de
    minúsculas — glob('*.pdf') é case-sensitive no Linux (onde a OCI roda) e ignoraria
    silenciosamente arquivos como 'Relatorio.PDF'."""
    if not pasta.exists():
        return []
    return sorted(p for p in pasta.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")


def carregar_documentos() -> list:
    """Lê todos os PDFs de documentos/ e retorna a lista de páginas carregadas."""
    documentos = []
    for caminho_pdf in listar_pdfs(PASTA_DOCUMENTOS):
        documentos.extend(PyPDFLoader(str(caminho_pdf)).load())
    return documentos


def extrair_amostra_texto(conteudo_pdf: bytes, max_paginas: int = 3) -> str:
    """Extrai o texto das primeiras páginas de um PDF em memória (sem salvar em disco) —
    usado para validar o assunto do documento antes de aceitar o upload."""
    leitor = PdfReader(io.BytesIO(conteudo_pdf))
    return "\n".join(pagina.extract_text() or "" for pagina in leitor.pages[:max_paginas])


def texto_parece_corrompido(texto: str, limite_proporcao: float = 0.05) -> bool:
    """Detecta texto de PDF extraído com caracteres de controle em excesso — sintoma de
    fonte embutida com codificação customizada que o pypdf não consegue mapear de volta
    para texto legível (o conteúdo vem 'embaralhado', não é problema de OCR nem de
    conteúdo ausente). Indexar esse texto geraria embeddings inúteis, que nunca vão
    casar com nenhuma pergunta."""
    if not texto.strip():
        return False
    suspeitos = sum(1 for c in texto if ord(c) < 32 and c not in ("\n", "\r", "\t"))
    return (suspeitos / len(texto)) > limite_proporcao


def dividir_em_chunks(documentos: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    return splitter.split_documents(documentos)


def construir_indice(chunks: list) -> FAISS:
    embeddings = GoogleGenerativeAIEmbeddingsComRetry(model=MODELO_EMBEDDING)
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
