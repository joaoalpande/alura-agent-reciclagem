"""
Wrapper de GoogleGenerativeAIEmbeddings com retry automático.

A cota gratuita de embeddings (gemini-embedding-001) tem um limite por minuto. Ao
estourar, a API retorna 429 RESOURCE_EXHAUSTED com um 'retryDelay' sugerido (geralmente
poucos segundos) — essa classe espera esse tempo e tenta de novo, em vez de propagar o
erro na primeira falha. Usada tanto na ingestão (construir_indice) quanto na busca
(retriever, via embed_query), que são os dois pontos que chamam a API de embeddings.
"""
import re
import time

from langchain_google_genai import GoogleGenerativeAIEmbeddings

PISTAS_ERRO_COTA = ("429", "RESOURCE_EXHAUSTED", "quota", "Quota")
MAX_TENTATIVAS = 4
ESPERA_PADRAO_SEGUNDOS = 15.0


def eh_erro_de_cota(erro) -> bool:
    """Verifica se um erro (ou sua mensagem) indica cota/rate-limit do Gemini esgotado.
    Compartilhado entre o retry de embeddings, a cadeia de fallback de chat (agente.py) e
    as mensagens de erro da interface (app.py), para não duplicar essa lista em cada
    lugar que precisa distinguir "cota esgotada" de outros tipos de erro."""
    return any(pista in str(erro) for pista in PISTAS_ERRO_COTA)


def _extrair_retry_delay(erro: Exception) -> float:
    """Lê o 'retryDelay' sugerido pela própria API (ex.: retryDelay: '3s') na mensagem
    de erro; usa um valor padrão se não encontrar."""
    match = re.search(r"retryDelay['\"]?:\s*['\"]?(\d+(?:\.\d+)?)s", str(erro))
    return float(match.group(1)) + 1 if match else ESPERA_PADRAO_SEGUNDOS


class GoogleGenerativeAIEmbeddingsComRetry(GoogleGenerativeAIEmbeddings):
    def embed_documents(self, texts, **kwargs):
        for tentativa in range(MAX_TENTATIVAS):
            try:
                return super().embed_documents(texts, **kwargs)
            except Exception as erro:
                if tentativa == MAX_TENTATIVAS - 1 or not eh_erro_de_cota(erro):
                    raise
                time.sleep(_extrair_retry_delay(erro))

    def embed_query(self, text, **kwargs):
        for tentativa in range(MAX_TENTATIVAS):
            try:
                return super().embed_query(text, **kwargs)
            except Exception as erro:
                if tentativa == MAX_TENTATIVAS - 1 or not eh_erro_de_cota(erro):
                    raise
                time.sleep(_extrair_retry_delay(erro))
