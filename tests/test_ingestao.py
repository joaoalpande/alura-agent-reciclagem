"""
Testes unitários das funções puras de ingestão (sem chamar a API do Gemini).
Rodar com: pytest
"""
from src.ingestao import listar_pdfs, texto_parece_corrompido


def test_listar_pdfs_ignora_maiuscula_minuscula(tmp_path):
    (tmp_path / "manual.pdf").write_bytes(b"conteudo")
    (tmp_path / "RELATORIO.PDF").write_bytes(b"conteudo")
    (tmp_path / "notas.txt").write_bytes(b"conteudo")

    nomes = sorted(p.name for p in listar_pdfs(tmp_path))

    assert nomes == ["RELATORIO.PDF", "manual.pdf"]


def test_listar_pdfs_pasta_inexistente(tmp_path):
    assert listar_pdfs(tmp_path / "nao_existe") == []


def test_listar_pdfs_pasta_vazia(tmp_path):
    assert listar_pdfs(tmp_path) == []


def test_texto_parece_corrompido_com_texto_normal():
    texto = "Alpande Tech\nManual de Política de Gestão de Materiais Reciclados"
    assert texto_parece_corrompido(texto) is False


def test_texto_parece_corrompido_com_caracteres_de_controle():
    # Simula o sintoma real de fonte com codificação customizada não mapeada pelo pypdf.
    texto = "\x14\x1c\x12\x14\x14\x12\x15\x13\x14\x18 /LPLWHV\x03GH\x033HVRV\x03H\x03'LPHQV}HV"
    assert texto_parece_corrompido(texto) is True


def test_texto_parece_corrompido_com_texto_vazio():
    assert texto_parece_corrompido("") is False
    assert texto_parece_corrompido("   ") is False
