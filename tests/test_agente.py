"""
Testes unitários da lógica pura do agente (sem chamar a API do Gemini nem carregar o FAISS).
Rodar com: pytest
"""
import pandas as pd
import pytest

from src.agente import _extrair_texto, calcular_dados_reciclagem


@pytest.fixture
def df_reciclagem():
    return pd.DataFrame(
        [
            {"mes": "2026-01", "material": "Papel", "percentual_reciclado": 60.0, "quantidade_kg": 100},
            {"mes": "2026-02", "material": "Papel", "percentual_reciclado": 70.0, "quantidade_kg": 200},
            {"mes": "2026-01", "material": "Metal", "percentual_reciclado": 80.0, "quantidade_kg": 50},
        ]
    )


def test_media_filtrando_por_material(df_reciclagem):
    resultado = calcular_dados_reciclagem(df_reciclagem, material="Papel", operacao="media")
    assert "= 65.00" in resultado


def test_valor_filtrando_por_mes_e_material(df_reciclagem):
    resultado = calcular_dados_reciclagem(
        df_reciclagem, mes="2026-02", material="papel", operacao="media"
    )
    assert "= 70.00" in resultado


def test_total_quantidade_kg(df_reciclagem):
    resultado = calcular_dados_reciclagem(
        df_reciclagem, material="Papel", metrica="quantidade_kg", operacao="total"
    )
    assert "= 300.00" in resultado


def test_maximo_e_minimo(df_reciclagem):
    maximo = calcular_dados_reciclagem(df_reciclagem, operacao="maximo")
    minimo = calcular_dados_reciclagem(df_reciclagem, operacao="minimo")
    assert "= 80.00" in maximo
    assert "= 60.00" in minimo


def test_filtro_sem_correspondencia(df_reciclagem):
    resultado = calcular_dados_reciclagem(df_reciclagem, mes="2099-12")
    assert "Nenhum dado encontrado" in resultado


def test_metrica_invalida(df_reciclagem):
    resultado = calcular_dados_reciclagem(df_reciclagem, metrica="cor_favorita")
    assert "Métrica inválida" in resultado


def test_operacao_invalida(df_reciclagem):
    resultado = calcular_dados_reciclagem(df_reciclagem, operacao="mediana")
    assert "Operação inválida" in resultado


def test_extrair_texto_com_string_simples():
    assert _extrair_texto("olá mundo") == "olá mundo"


def test_extrair_texto_com_blocos_estruturados():
    conteudo = [{"type": "text", "text": "parte 1. "}, {"type": "text", "text": "parte 2."}]
    assert _extrair_texto(conteudo) == "parte 1. parte 2."
