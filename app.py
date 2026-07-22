"""
Interface de chat (Streamlit) para o Alura Agent.

Uso:
    streamlit run app.py
"""
import streamlit as st

from src.agente import responder

st.set_page_config(page_title="Alura Agent", page_icon="♻️")
st.title("♻️ Alura Agent — Manual de Reciclagem")
st.caption("Pergunte algo sobre o manual de reciclagem indexado neste projeto.")


@st.cache_resource
def preparar_agente():
    # Força o carregamento do índice FAISS e do modelo uma única vez por sessão do servidor.
    from src.agente import _obter_cadeia
    _obter_cadeia()
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
        with st.spinner("Consultando o documento..."):
            resultado = responder(pergunta)
        st.markdown(resultado["resposta"])

        with st.expander("Trechos usados como fonte"):
            for i, doc in enumerate(resultado["fontes"], start=1):
                pagina = doc.metadata.get("page", "?")
                st.markdown(f"**Fonte {i} — página {pagina}**")
                st.text(doc.page_content)

    st.session_state.historico.append(
        {"papel": "assistant", "conteudo": resultado["resposta"]}
    )
