"""
Agente de perguntas e respostas sobre os documentos internos da empresa.

Combina duas ferramentas:
- buscar_no_manual: busca trechos relevantes no manual de reciclagem (PDF), via FAISS.
- consultar_dados_reciclagem: calcula média/total/máximo/mínimo sobre o relatório mensal de
  reciclagem (CSV), usando pandas com parâmetros fixos (o modelo nunca executa código livre).

Expõe responder(pergunta) -> {"resposta": str, "fontes": list[dict]}.
"""
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, Field

load_dotenv()

RAIZ = Path(__file__).resolve().parent.parent
PASTA_VECTORSTORE = RAIZ / "vectorstore"
CAMINHO_CSV = RAIZ / "documentos" / "relatorio_reciclagem_mensal.csv"

MODELO_EMBEDDING = "models/gemini-embedding-001"

# Cadeia de fallback: cada modelo tem cota gratuita separada. Se o atual esgotar
# a cota, tenta o próximo, do mais capaz para o mais "de reserva".
MODELOS_CHAT_FALLBACK = (
    "gemini-flash-latest",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemma-4-26b-a4b-it",
)

METRICAS_PERMITIDAS = ("percentual_reciclado", "quantidade_kg")
OPERACOES_PERMITIDAS = ("media", "total", "maximo", "minimo")

PROMPT_AGENTE = (
    "Você é o Alura Agent, um assistente que responde perguntas sobre documentos internos da "
    "empresa. Você tem duas ferramentas:\n"
    "- 'buscar_no_manual': retorna trechos do manual de reciclagem (PDF) relacionados à pergunta. "
    "Use para políticas, procedimentos e regras.\n"
    "- 'consultar_dados_reciclagem': calcula média, total, máximo ou mínimo de percentual "
    "reciclado ou quantidade (kg), a partir do relatório mensal (CSV). Use para perguntas "
    "numéricas ou percentuais.\n"
    "Escolha a ferramenta certa para cada pergunta e baseie sua resposta apenas no que ela "
    "retornar. Se nenhuma ferramenta trouxer a informação, diga claramente que não encontrou a "
    "informação nos documentos disponíveis. Responda sempre em português, de forma clara e "
    "objetiva."
)


def _montar_retriever():
    if not PASTA_VECTORSTORE.exists():
        raise RuntimeError("Índice não encontrado. Rode antes: python src/ingestao.py")

    embeddings = GoogleGenerativeAIEmbeddings(model=MODELO_EMBEDDING)
    vectorstore = FAISS.load_local(
        str(PASTA_VECTORSTORE), embeddings, allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever(search_kwargs={"k": 4})


def _carregar_dataframe_reciclagem() -> pd.DataFrame:
    if not CAMINHO_CSV.exists():
        raise RuntimeError(f"CSV não encontrado em {CAMINHO_CSV}")
    return pd.read_csv(CAMINHO_CSV)


class ConsultaReciclagem(BaseModel):
    mes: Optional[str] = Field(
        default=None,
        description="Mês no formato AAAA-MM, ex: '2026-03'. Deixe vazio para considerar todos os meses.",
    )
    material: Optional[str] = Field(
        default=None,
        description="Material: Papel, Plastico, Metal, Vidro ou Eletronicos. Deixe vazio para considerar todos.",
    )
    metrica: Literal["percentual_reciclado", "quantidade_kg"] = Field(
        default="percentual_reciclado", description="Métrica a consultar."
    )
    operacao: Literal["media", "total", "maximo", "minimo"] = Field(
        default="media", description="Operação de agregação a aplicar."
    )


def calcular_dados_reciclagem(
    df: pd.DataFrame,
    mes: Optional[str] = None,
    material: Optional[str] = None,
    metrica: str = "percentual_reciclado",
    operacao: str = "media",
) -> str:
    """Filtra o DataFrame de reciclagem por mês/material e agrega a métrica pedida.

    Função pura (sem dependência de FAISS/Gemini) para poder ser testada isoladamente.
    """
    if metrica not in METRICAS_PERMITIDAS:
        return f"Métrica inválida. Use uma de: {METRICAS_PERMITIDAS}"
    if operacao not in OPERACOES_PERMITIDAS:
        return f"Operação inválida. Use uma de: {OPERACOES_PERMITIDAS}"

    df_filtrado = df
    if mes:
        df_filtrado = df_filtrado[df_filtrado["mes"] == mes]
    if material:
        df_filtrado = df_filtrado[df_filtrado["material"].str.lower() == material.lower()]

    if df_filtrado.empty:
        return "Nenhum dado encontrado para os filtros informados."

    serie = df_filtrado[metrica]
    valor = {
        "media": serie.mean(),
        "total": serie.sum(),
        "maximo": serie.max(),
        "minimo": serie.min(),
    }[operacao]

    return (
        f"{operacao} de {metrica} (mês={mes or 'todos'}, material={material or 'todos'}) "
        f"= {valor:.2f}"
    )


def _construir_ferramentas():
    retriever = _montar_retriever()
    df_reciclagem = _carregar_dataframe_reciclagem()

    @tool
    def buscar_no_manual(pergunta: str) -> str:
        """Busca trechos do manual de reciclagem (PDF) relevantes para a pergunta."""
        documentos = retriever.invoke(pergunta)
        if not documentos:
            return "Nenhum trecho relevante encontrado no manual."
        return "\n\n".join(
            f"[página {doc.metadata.get('page', '?')}] {doc.page_content}"
            for doc in documentos
        )

    @tool(args_schema=ConsultaReciclagem)
    def consultar_dados_reciclagem(
        mes: Optional[str] = None,
        material: Optional[str] = None,
        metrica: str = "percentual_reciclado",
        operacao: str = "media",
    ) -> str:
        """Calcula média, total, máximo ou mínimo de percentual reciclado ou quantidade (kg) a
        partir do relatório mensal de reciclagem (CSV), filtrando por mês e/ou material."""
        return calcular_dados_reciclagem(df_reciclagem, mes, material, metrica, operacao)

    return [buscar_no_manual, consultar_dados_reciclagem]


_llms_por_modelo = {}
_ferramentas_lista = None
_ferramentas_por_nome = None

LIMITE_CHAMADAS_FERRAMENTA = 5

PISTAS_ERRO_RECUPERAVEL = ("429", "RESOURCE_EXHAUSTED", "quota", "Quota", "NOT_FOUND", "404")


def _eh_erro_recuperavel(erro: Exception) -> bool:
    """Erros de cota esgotada ou modelo indisponível: vale tentar o próximo da cadeia."""
    mensagem = str(erro)
    return any(pista in mensagem for pista in PISTAS_ERRO_RECUPERAVEL)


def _obter_executor():
    """Garante que ferramentas e todos os modelos da cadeia de fallback estão prontos
    (não faz chamada de API, só monta os objetos — usado para pré-carregar no Streamlit)."""
    for nome_modelo in MODELOS_CHAT_FALLBACK:
        _obter_llm_para_modelo(nome_modelo)


def _obter_ferramentas():
    global _ferramentas_lista, _ferramentas_por_nome
    if _ferramentas_lista is None:
        _ferramentas_lista = _construir_ferramentas()
        _ferramentas_por_nome = {f.name: f for f in _ferramentas_lista}
    return _ferramentas_lista, _ferramentas_por_nome


def _obter_llm_para_modelo(nome_modelo: str):
    if nome_modelo not in _llms_por_modelo:
        ferramentas, _ = _obter_ferramentas()
        # max_retries=1: sem retry interno da biblioteca (5-6 tentativas com espera
        # crescente, ~60s) — a cadeia de fallback já cuida da resiliência entre modelos.
        llm = ChatGoogleGenerativeAI(model=nome_modelo, temperature=0.2, max_retries=1)
        _llms_por_modelo[nome_modelo] = llm.bind_tools(ferramentas)
    return _llms_por_modelo[nome_modelo]


def _invocar_com_fallback(mensagens):
    ultimo_erro = None
    for nome_modelo in MODELOS_CHAT_FALLBACK:
        llm_com_ferramentas = _obter_llm_para_modelo(nome_modelo)
        try:
            return llm_com_ferramentas.invoke(mensagens)
        except Exception as erro:
            if not _eh_erro_recuperavel(erro):
                raise
            ultimo_erro = erro
    raise ultimo_erro


def _extrair_texto(conteudo) -> str:
    """O conteúdo da resposta pode vir como string simples ou como lista de blocos
    (ex.: {"type": "text", "text": "..."}), dependendo do modelo. Normaliza para string."""
    if isinstance(conteudo, str):
        return conteudo
    if isinstance(conteudo, list):
        return "".join(
            bloco.get("text", "")
            for bloco in conteudo
            if isinstance(bloco, dict) and bloco.get("type") == "text"
        )
    return str(conteudo)


def responder(pergunta: str) -> dict:
    _, ferramentas_por_nome = _obter_ferramentas()
    mensagens = [SystemMessage(content=PROMPT_AGENTE), HumanMessage(content=pergunta)]
    fontes = []

    for _ in range(LIMITE_CHAMADAS_FERRAMENTA):
        resposta = _invocar_com_fallback(mensagens)
        mensagens.append(resposta)

        if not resposta.tool_calls:
            return {"resposta": _extrair_texto(resposta.content), "fontes": fontes}

        for chamada in resposta.tool_calls:
            ferramenta = ferramentas_por_nome[chamada["name"]]
            resultado = ferramenta.invoke(chamada["args"])
            fontes.append(
                {
                    "ferramenta": chamada["name"],
                    "entrada": chamada["args"],
                    "saida": resultado,
                }
            )
            mensagens.append(
                ToolMessage(content=str(resultado), tool_call_id=chamada["id"])
            )

    return {
        "resposta": "Não consegui concluir a resposta após várias tentativas.",
        "fontes": fontes,
    }
