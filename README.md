# Alura Agent — Agente de IA para Documentos Internos

Projeto do desafio final **Alura Agent** (Alura + Oracle). Um agente de inteligência artificial
que responde perguntas sobre documentos internos da empresa (neste caso, um manual de reciclagem),
eliminando a necessidade de buscar as informações manualmente no PDF.

## Descrição geral

O agente usa a técnica de **RAG (Retrieval-Augmented Generation)**: em vez de "decorar" o
documento no modelo, ele indexa o conteúdo em um banco vetorial e, a cada pergunta, busca os
trechos mais relevantes para montar o contexto que é enviado ao modelo de linguagem (Gemini).
Isso permite respostas fundamentadas no texto real do documento, com baixa chance de alucinação.

## Arquitetura da solução

Duas fases:

1. **Ingestão (offline)** — roda uma vez (ou sempre que os documentos mudarem):
   PDF → divisão em *chunks* → embeddings (Gemini) → índice vetorial FAISS salvo em disco.
2. **Consulta (online, na interface)** — a cada pergunta do usuário:
   pergunta → busca os *chunks* mais similares no FAISS → monta prompt com o contexto encontrado
   → Gemini gera a resposta em português → interface exibe resposta + trechos-fonte usados.

```
documentos/*.pdf ──► PyPDFLoader ──► RecursiveCharacterTextSplitter ──► GoogleGenerativeAIEmbeddings
                                                                              │
                                                                              ▼
                                                                        FAISS (vectorstore/)
                                                                              │
   pergunta ──► retriever (top-k) ──► prompt + contexto ──► ChatGoogleGenerativeAI ──► resposta + fontes
```

## Tecnologias e ferramentas

- **Python 3.11+**
- **LangChain** (`langchain`, `langchain-community`, `langchain-google-genai`) — orquestração do RAG
- **Google Gemini** — `gemini-flash-latest` (geração de respostas) e `models/gemini-embedding-001`
  (embeddings)
- **pypdf** — leitura do PDF (via `PyPDFLoader`)
- **FAISS** (`faiss-cpu`) — índice vetorial local, sem depender de banco de dados externo
- **Streamlit** — interface web de chat
- **python-dotenv** — carregamento da chave de API a partir de `.env`

## Estrutura do projeto

```
.
├── app.py                    # interface de chat (Streamlit)
├── src/
│   ├── ingestao.py            # lê os PDFs e constrói o índice FAISS
│   └── agente.py               # cadeia de RAG (retriever + prompt + Gemini)
├── documentos/
│   └── manual_reciclagem.pdf  # documento de exemplo usado como fonte
├── vectorstore/               # índice FAISS gerado (não versionado)
├── requirements.txt
├── .env.example
└── .gitignore
```

## Como executar localmente

1. Clone o repositório e entre na pasta do projeto.
2. Crie e ative um ambiente virtual:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```
3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
4. Copie `.env.example` para `.env` e preencha sua chave do Gemini (crie uma gratuitamente em
   https://aistudio.google.com/apikey):
   ```
   GOOGLE_API_KEY=sua_chave_aqui
   ```
5. Gere o índice vetorial a partir dos documentos em `documentos/`:
   ```bash
   python src/ingestao.py
   ```
6. Rode a aplicação:
   ```bash
   streamlit run app.py
   ```
7. Acesse http://localhost:8501 no navegador.

Para usar outro documento, basta colocar o PDF em `documentos/` e rodar novamente o passo 5.

## Exemplos de perguntas e respostas

**Pergunta:** Quais materiais podem ser reciclados segundo o manual?
**Resposta:** Com base no manual, os materiais recicláveis listados e geridos pela empresa são:
Papel e Papelão, Plástico, Metal, Vidro e Eletrônicos (e-lixo), incluindo cabos, baterias,
periféricos e equipamentos.

**Pergunta:** Qual a cor da lixeira para plástico?
**Resposta:** A cor da lixeira para plástico é vermelha.

**Pergunta:** Qual a capital da França?
**Resposta:** Não encontrei essa informação nos documentos disponíveis.

*(O último exemplo mostra o agente recusando responder sobre algo fora do documento, em vez de
inventar uma resposta.)*

## Deploy na Oracle Cloud Infrastructure (OCI)

Passo a passo para publicar a aplicação em uma VM **OCI Compute** (elegível ao Always Free):

1. Crie uma instância de Compute (imagem Ubuntu 22.04, shape `VM.Standard.A1.Flex` ou
   `VM.Standard.E2.1.Micro`, ambos no nível Always Free).
2. Conecte via SSH e instale os pré-requisitos:
   ```bash
   sudo apt update && sudo apt install -y python3-venv git
   ```
3. Clone o repositório e configure o ambiente (repita os passos 2–5 de "Como executar localmente").
4. Libere a porta 8501:
   - Na **Security List/NSG** da VCN: adicione uma regra de ingresso para a porta 8501 (TCP).
   - No firewall do sistema operacional:
     ```bash
     sudo iptables -I INPUT -p tcp --dport 8501 -j ACCEPT
     ```
5. Suba a aplicação expondo-a publicamente:
   ```bash
   streamlit run app.py --server.address 0.0.0.0 --server.port 8501
   ```
6. Acesse `http://<IP_PÚBLICO_DA_VM>:8501`.

Opcionalmente, configure um serviço `systemd` para manter a aplicação no ar após reinicializações.

**Evidência do deploy:** _(adicionar aqui o link público da aplicação e/ou uma captura de tela da
aplicação em execução na OCI)_.
