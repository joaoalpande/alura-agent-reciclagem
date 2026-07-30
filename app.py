"""
Interface de chat (Streamlit) para o Alura Agent.

Uso:
    streamlit run app.py
"""
from pathlib import Path

import streamlit as st

from src.agente import responder

RAIZ = Path(__file__).resolve().parent
PASTA_DOCUMENTOS = RAIZ / "documentos"

st.set_page_config(page_title="Alura Agent", page_icon="♻️")

# O texto do file_uploader ("Drag and drop...", "Browse files") vem embutido no bundle
# JS do Streamlit e não tem parâmetro de idioma — só dá pra traduzir sobrescrevendo via
# CSS. Depende dos data-testid internos da versão 1.51.0; pode voltar a aparecer em
# inglês se o Streamlit mudar essa marcação numa atualização futura.
st.markdown(
    """
    <style>
    [data-testid="stFileUploaderDropzoneInstructions"] > div > span {
        font-size: 0;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] > div > span:first-of-type::after {
        content: "Arraste e solte os arquivos aqui";
        font-size: 1rem;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] > div > span:last-of-type::after {
        content: "Limite de 200MB por arquivo • PDF";
        font-size: 0.8rem;
    }
    [data-testid="stFileUploaderDropzone"] button {
        font-size: 0;
    }
    [data-testid="stFileUploaderDropzone"] button::after {
        content: "Procurar arquivos";
        font-size: 14px;
    }
    [data-testid="stFileUploaderFileErrorMessage"] {
        font-size: 0;
    }
    [data-testid="stFileUploaderFileErrorMessage"]::after {
        /* Cobre tanto "tipo de arquivo inválido" quanto "arquivo maior que 200MB" —
        são duas mensagens nativas diferentes no mesmo elemento, e CSS não distingue
        qual delas é; por isso o texto é genérico o bastante pra valer nos dois casos. */
        content: "Arquivo não aceito — confira o tipo (PDF) e o tamanho (até 200MB).";
        font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.image(str(RAIZ / "assets" / "banner-challenge.png"))
st.title("♻️ Alura Agent — Documentos Internos da Alpande Tech")
st.caption(
    "Pergunte sobre os documentos internos da empresa: o manual de reciclagem (PDF, "
    "expansível por upload) e o relatório mensal de reciclagem (CSV)."
)


@st.cache_resource
def preparar_agente():
    from src.agente import _obter_executor
    from src.ingestao import carregar_documentos, construir_indice, dividir_em_chunks

    # vectorstore/ não é versionado no Git (é gerado, não código-fonte). Num clone novo do
    # repositório (ex.: primeiro deploy na OCI), ele ainda não existe — constrói sozinho a
    # partir dos PDFs já presentes em documentos/ (o manual padrão), em vez de exigir rodar
    # `python src/ingestao.py` manualmente antes do primeiro `streamlit run`.
    if not (RAIZ / "vectorstore").exists():
        documentos = carregar_documentos()
        if documentos:
            chunks = dividir_em_chunks(documentos)
            indice = construir_indice(chunks)
            indice.save_local(str(RAIZ / "vectorstore"))

    # Força o carregamento do índice FAISS, do CSV e do modelo uma única vez por sessão do servidor.
    _obter_executor()
    return True


if "indice_desatualizado" not in st.session_state:
    st.session_state.indice_desatualizado = False
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "upload_mensagem" not in st.session_state:
    st.session_state.upload_mensagem = None

with st.sidebar:
    st.subheader("Gestão de documentos")
    arquivos = st.file_uploader(
        "Adicione PDFs ao manual de reciclagem",
        type="pdf",
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )
    if arquivos:
        from src.agente import eh_documento_sobre_reciclagem
        from src.ingestao import extrair_amostra_texto, texto_parece_corrompido

        PASTA_DOCUMENTOS.mkdir(exist_ok=True)
        salvos, recusados, duplicados, invalidos, corrompidos = [], [], [], [], []
        with st.spinner("Verificando se os arquivos são sobre reciclagem..."):
            for arquivo in arquivos:
                if (PASTA_DOCUMENTOS / arquivo.name).exists():
                    duplicados.append(arquivo.name)
                    continue
                conteudo = arquivo.getvalue()
                try:
                    amostra = extrair_amostra_texto(conteudo)
                except Exception:
                    invalidos.append(arquivo.name)
                    continue
                if texto_parece_corrompido(amostra):
                    corrompidos.append(arquivo.name)
                elif eh_documento_sobre_reciclagem(amostra):
                    (PASTA_DOCUMENTOS / arquivo.name).write_bytes(conteudo)
                    salvos.append(arquivo.name)
                else:
                    recusados.append(arquivo.name)

        if salvos:
            st.session_state.indice_desatualizado = True
        st.session_state.upload_mensagem = {
            "salvos": salvos, "recusados": recusados, "duplicados": duplicados,
            "invalidos": invalidos, "corrompidos": corrompidos,
        }
        # Muda a key do uploader para forçar o widget a esvaziar (senão o Streamlit mantém
        # os mesmos arquivos "selecionados" e reprocessa tudo de novo a cada rerun futuro,
        # inclusive ao clicar em outros botões da página).
        st.session_state.uploader_key += 1
        st.rerun()

    if st.session_state.upload_mensagem:
        msg = st.session_state.upload_mensagem
        if msg["salvos"]:
            if len(msg["salvos"]) == 1:
                aviso = "1 arquivo salvo, mas o agente ainda **não pesquisa nele**."
            else:
                aviso = (
                    f"{len(msg['salvos'])} arquivos salvos, mas o agente ainda "
                    "**não pesquisa neles**."
                )
            st.warning(f"{aviso} Clique em 'Reconstruir índice FAISS' abaixo para isso.")

        if msg["duplicados"]:
            if len(msg["duplicados"]) == 1:
                aviso = (
                    f"Já existe um documento com esse nome, upload ignorado: "
                    f"{msg['duplicados'][0]}."
                )
            else:
                aviso = (
                    "Já existem documentos com esses nomes, upload ignorado: "
                    f"{', '.join(msg['duplicados'])}."
                )
            st.info(f"{aviso} Exclua o atual primeiro se quiser substituí-lo.")

        if msg["recusados"]:
            if len(msg["recusados"]) == 1:
                aviso = (
                    "Recusado por não parecer relacionado a reciclagem/gestão de "
                    f"resíduos: {msg['recusados'][0]}"
                )
            else:
                aviso = (
                    "Recusados por não parecerem relacionados a reciclagem/gestão de "
                    f"resíduos: {', '.join(msg['recusados'])}"
                )
            st.error(aviso)

        if msg["corrompidos"]:
            if len(msg["corrompidos"]) == 1:
                aviso = (
                    "O texto extraído deste PDF veio corrompido (fonte com codificação "
                    f"não padrão) e não seria útil para busca: {msg['corrompidos'][0]}. "
                    "Tente reexportar o PDF de outra fonte."
                )
            else:
                aviso = (
                    "O texto extraído destes PDFs veio corrompido (fonte com codificação "
                    f"não padrão) e não seria útil para busca: {', '.join(msg['corrompidos'])}. "
                    "Tente reexportar os PDFs de outra fonte."
                )
            st.error(aviso)

        if msg["invalidos"]:
            if len(msg["invalidos"]) == 1:
                aviso = f"Não foi possível ler como PDF (arquivo corrompido ou inválido): {msg['invalidos'][0]}"
            else:
                aviso = (
                    "Não foi possível ler como PDF (arquivos corrompidos ou inválidos): "
                    f"{', '.join(msg['invalidos'])}"
                )
            st.error(aviso)

        st.session_state.upload_mensagem = None

    from src.ingestao import listar_pdfs

    st.markdown("**Documentos atuais**")
    pdfs_atuais = listar_pdfs(PASTA_DOCUMENTOS)
    if not pdfs_atuais:
        st.caption("Nenhum PDF em documentos/ no momento.")
    for pdf in pdfs_atuais:
        col_nome, col_excluir = st.columns([4, 1])
        col_nome.write(pdf.name)
        if col_excluir.button("🗑️", key=f"excluir_{pdf.name}", help=f"Remover {pdf.name}"):
            pdf.unlink()
            st.session_state.indice_desatualizado = True
            st.rerun()

    if st.session_state.indice_desatualizado:
        st.warning("⚠️ Índice desatualizado — reconstrua antes de perguntar sobre os documentos novos.")
    else:
        st.caption("✅ Índice em dia com os documentos atuais.")

    if st.button("🔄 Reconstruir índice FAISS"):
        import shutil

        from src.agente import resetar_cache
        from src.ingestao import carregar_documentos, construir_indice, dividir_em_chunks

        with st.spinner("Reconstruindo índice a partir dos documentos..."):
            try:
                documentos = carregar_documentos()
                if not documentos:
                    # Sem nenhum PDF na pasta, um índice antigo deixado no disco faria o
                    # agente continuar respondendo com base em documentos já excluídos.
                    if (RAIZ / "vectorstore").exists():
                        shutil.rmtree(RAIZ / "vectorstore")
                    resetar_cache()
                    preparar_agente.clear()
                    st.session_state.indice_desatualizado = False
                    st.warning(
                        f"Nenhum PDF encontrado em {PASTA_DOCUMENTOS} — índice removido. "
                        "A busca em documentos ficará indisponível até adicionar um PDF e "
                        "reconstruir novamente."
                    )
                else:
                    chunks = dividir_em_chunks(documentos)
                    indice = construir_indice(chunks)
                    indice.save_local(str(RAIZ / "vectorstore"))
                    resetar_cache()
                    preparar_agente.clear()
                    st.session_state.indice_desatualizado = False
                    st.success("Índice reconstruído com sucesso!")
            except Exception as erro:
                mensagem_erro = str(erro)
                if any(
                    pista in mensagem_erro
                    for pista in ("429", "RESOURCE_EXHAUSTED", "quota", "Quota")
                ):
                    st.error(
                        "⚠️ A cota gratuita do modelo de embeddings foi atingida (limite por "
                        "minuto/dia do Gemini). Aguarde um pouco e clique em 'Reconstruir "
                        "índice FAISS' novamente."
                    )
                else:
                    st.error(f"Falha ao reconstruir o índice: {erro}")

    st.divider()
    st.subheader("Dados numéricos (fixo)")
    st.caption(
        "📊 relatorio_reciclagem_mensal.csv — não é gerenciado por upload aqui, pois a "
        "ferramenta de cálculo espera colunas fixas (mês, material, % reciclado, kg). "
        "Pergunte por médias, totais, máximos ou mínimos."
    )

try:
    preparar_agente()
except RuntimeError as erro:
    st.error(str(erro))
    st.stop()

if "historico" not in st.session_state:
    st.session_state.historico = []

if not st.session_state.historico:
    with st.expander("💡 Exemplos de perguntas"):
        st.markdown(
            "- Quais materiais podem ser reciclados segundo o manual?\n"
            "- Como devo separar o lixo orgânico?\n"
            "- Qual foi o percentual médio de reciclagem no último mês? *(usa o CSV)*\n"
            "- Qual material teve o maior total reciclado em kg? *(usa o CSV)*"
        )

for mensagem in st.session_state.historico:
    with st.chat_message(mensagem["papel"]):
        st.markdown(mensagem["conteudo"])

pergunta = st.chat_input("Digite sua pergunta sobre o documento...")

if pergunta:
    st.session_state.historico.append({"papel": "user", "conteudo": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    with st.chat_message("assistant"):
        fontes = []
        modelo_usado = None
        with st.spinner("Consultando o documento..."):
            try:
                resultado = responder(pergunta)
                resposta_texto = resultado["resposta"]
                fontes = resultado["fontes"]
                modelo_usado = resultado["modelo"]
            except Exception as erro:
                mensagem_erro = str(erro)
                if any(
                    pista in mensagem_erro
                    for pista in ("429", "RESOURCE_EXHAUSTED", "quota", "Quota")
                ):
                    resposta_texto = (
                        "⚠️ A cota gratuita diária do Gemini foi atingida. Tente novamente "
                        "amanhã, quando a cota é renovada (o limite gratuito reseta a cada 24h)."
                    )
                elif "Índice não encontrado" in mensagem_erro:
                    resposta_texto = (
                        "⚠️ Nenhum PDF está indexado no momento (todos foram removidos). "
                        "Adicione um PDF na barra lateral e clique em 'Reconstruir índice "
                        "FAISS' para eu voltar a responder sobre documentos."
                    )
                else:
                    resposta_texto = (
                        "⚠️ Ocorreu um erro ao consultar o agente. Tente novamente em instantes."
                    )

        st.markdown(resposta_texto)

        NOMES_FONTE = {
            "buscar_no_manual": "Documentos em PDF (RAG)",
            "consultar_dados_reciclagem": "Relatório mensal (CSV)",
        }
        if modelo_usado:
            nomes_usados = {NOMES_FONTE.get(f["ferramenta"], f["ferramenta"]) for f in fontes}
            fonte_dados = ", ".join(sorted(nomes_usados)) or "resposta direta do modelo"
            st.caption(f"🤖 Fonte usada: {fonte_dados} · Modelo: {modelo_usado}")

        if fontes:
            with st.expander("Como o agente chegou nessa resposta"):
                for i, fonte in enumerate(fontes, start=1):
                    st.markdown(f"**Passo {i} — ferramenta `{fonte['ferramenta']}`**")
                    st.markdown(f"Entrada: `{fonte['entrada']}`")
                    st.text(fonte["saida"])

    st.session_state.historico.append({"papel": "assistant", "conteudo": resposta_texto})
