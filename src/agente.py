"""
Agente de perguntas e respostas (RAG) sobre os documentos indexados.

Expõe responder(pergunta) -> {"resposta": str, "fontes": list[Document]}.
"""
from pathlib import Path

from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

RAIZ = Path(__file__).resolve().parent.parent
PASTA_VECTORSTORE = RAIZ / "vectorstore"

MODELO_EMBEDDING = "models/text-embedding-004"
MODELO_CHAT = "gemini-2.0-flash"

PROMPT_SISTEMA = (
    "Você é um assistente que responde perguntas sobre documentos internos da empresa. "
    "Use apenas as informações do contexto abaixo para responder. "
    "Se a resposta não estiver no contexto, diga claramente que não encontrou "
    "essa informação nos documentos disponíveis. Responda em português, de forma clara e objetiva.\n\n"
    "Contexto:\n{context}"
)


def _carregar_cadeia():
    if not PASTA_VECTORSTORE.exists():
        raise RuntimeError(
            "Índice não encontrado. Rode antes: python src/ingestao.py"
        )

    embeddings = GoogleGenerativeAIEmbeddings(model=MODELO_EMBEDDING)
    vectorstore = FAISS.load_local(
        str(PASTA_VECTORSTORE),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatGoogleGenerativeAI(model=MODELO_CHAT, temperature=0.2)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PROMPT_SISTEMA),
            ("human", "{input}"),
        ]
    )

    cadeia_documentos = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, cadeia_documentos)


_cadeia = None


def _obter_cadeia():
    global _cadeia
    if _cadeia is None:
        _cadeia = _carregar_cadeia()
    return _cadeia


def responder(pergunta: str) -> dict:
    cadeia = _obter_cadeia()
    resultado = cadeia.invoke({"input": pergunta})
    return {
        "resposta": resultado["answer"],
        "fontes": resultado["context"],
    }
