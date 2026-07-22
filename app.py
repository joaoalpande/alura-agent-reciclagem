"""
Interface de chat (Streamlit) para o Alura Agent.

Uso:
    streamlit run app.py
"""
import streamlit as st

from src.agente import responder

st.set_page_config(page_title="Alura Agent", page_icon="♻️")
st.title("♻️ Alura Agent — Manual e Relatório de Reciclagem")
st.caption(
    "Pergunte sobre o manual de reciclagem (PDF) ou sobre os percentuais do relatório "
    "mensal de reciclagem (CSV)."
)


@st.cache_resource
def preparar_agente():
    # Força o carregamento do índice FAISS, do CSV e do modelo uma única vez por sessão do servidor.
    from src.agente import _obter_executor
    _obter_executor()
    return True


try:
    preparar_agente()
except RuntimeError as erro:
    st.error(str(erro))
    st.stop()

if "historico" not in st.session_state:
    st.session_state.historico = []

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
        with st.spinner("Consultando o documento..."):
            try:
                resultado = responder(pergunta)
                resposta_texto = resultado["resposta"]
                fontes = resultado["fontes"]
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
                else:
                    resposta_texto = (
                        "⚠️ Ocorreu um erro ao consultar o agente. Tente novamente em instantes."
                    )

        st.markdown(resposta_texto)

        if fontes:
            with st.expander("Como o agente chegou nessa resposta"):
                for i, fonte in enumerate(fontes, start=1):
                    st.markdown(f"**Passo {i} — ferramenta `{fonte['ferramenta']}`**")
                    st.markdown(f"Entrada: `{fonte['entrada']}`")
                    st.text(fonte["saida"])

    st.session_state.historico.append({"papel": "assistant", "conteudo": resposta_texto})
