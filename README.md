<p align="center">
  <img src="assets/banner-challenge.png" alt="Banner do desafio Alura + Oracle" width="480">
</p>

# ♻️ Alura Agent — Agente de IA para Documentos Internos

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Streamlit](https://img.shields.io/badge/streamlit-app-ff4b4b)
![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC)

Projeto do desafio final **Alura Agent** (Alura + Oracle). Um agente de inteligência artificial
que responde perguntas sobre documentos internos da **Alpande Tech** — o manual de política de
gestão de materiais recicláveis (PDF) e o relatório mensal de reciclagem (CSV) — eliminando a
necessidade de buscar as informações manualmente.

## Sumário

- [Descrição geral](#descrição-geral)
- [Arquitetura da solução](#arquitetura-da-solução)
- [Gestão de documentos pela interface](#gestão-de-documentos-pela-interface)
- [Tecnologias e ferramentas](#tecnologias-e-ferramentas)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Como executar localmente](#como-executar-localmente)
- [Testes](#testes)
- [Exemplos de perguntas e respostas](#exemplos-de-perguntas-e-respostas)
- [Deploy na Oracle Cloud Infrastructure (OCI)](#deploy-na-oracle-cloud-infrastructure-oci)
- [Licença](#licença)

## Descrição geral

O agente combina duas técnicas, escolhendo a certa para cada tipo de pergunta:

- **RAG (Retrieval-Augmented Generation)** para perguntas sobre políticas e procedimentos do
  manual em PDF: o conteúdo é indexado em um banco vetorial e, a cada pergunta, os trechos mais
  relevantes são recuperados para servir de contexto ao modelo.
- **Consulta estruturada com pandas** para perguntas numéricas/percentuais sobre o relatório
  mensal em CSV (médias, totais, valores por mês/material): os dados são consultados diretamente
  com pandas, garantindo que os números sejam calculados corretamente, não "adivinhados" pelo
  modelo.

O agente é implementado como um **agente com tool-calling** (Gemini function calling): a cada
pergunta, o próprio modelo decide qual ferramenta usar (busca no manual, consulta ao CSV, ou
nenhuma, quando a informação não está disponível).

A interface traz um botão para o repositório no GitHub e, na primeira visita, uma lista de
**perguntas sugeridas clicáveis** (agrupadas por fonte — manual em PDF ou relatório em CSV) que já
envia a pergunta ao ser clicada, para facilitar o primeiro contato com o agente.

## Arquitetura da solução

1. **Ingestão (offline)** — roda uma vez (ou sempre que o PDF mudar):
   PDF → divisão em *chunks* → embeddings (Gemini) → índice vetorial FAISS salvo em disco.
2. **Consulta (online, na interface)** — a cada pergunta do usuário, o Gemini decide qual
   ferramenta chamar:
   - `buscar_no_manual`: busca os *chunks* mais similares no FAISS (manual em PDF).
   - `consultar_dados_reciclagem`: calcula média/total/máximo/mínimo sobre o CSV com pandas,
     usando apenas parâmetros escolhidos pelo modelo (mês, material, métrica, operação) — o
     modelo **nunca executa código livre**, só valores de um conjunto fixo permitido.
   O resultado da ferramenta volta para o modelo, que gera a resposta final em português. A
   interface exibe a resposta e, num painel expansível, qual ferramenta foi usada e com que dados.

```mermaid
flowchart LR
    A["documentos/*.pdf"] --> B[PyPDFLoader]
    B --> C[RecursiveCharacterTextSplitter]
    C --> D[GoogleGenerativeAIEmbeddings]
    D --> E[("FAISS\nvectorstore/")]
```

```mermaid
flowchart TD
    Q[Pergunta do usuário] --> M["ChatGoogleGenerativeAI\n(decide a ferramenta)"]
    M -- "política / procedimento" --> T1["buscar_no_manual\n(busca no FAISS)"]
    M -- "número / percentual" --> T2["consultar_dados_reciclagem\n(pandas sobre o CSV)"]
    M -- "fora do escopo" --> N[Nenhuma ferramenta]
    T1 --> R[Resposta final + fontes usadas]
    T2 --> R
    N --> R2["Resposta: não encontrado nos documentos"]
```

### Por que não um "agente pandas" com execução livre de código?

O LangChain oferece um padrão pronto (`create_pandas_dataframe_agent`) que deixa o modelo executar
código Python livremente sobre o CSV. Como esta aplicação fica **pública na internet** após o
deploy, isso abriria uma brecha real de execução remota de código (qualquer visitante poderia
tentar manipular a pergunta para rodar código arbitrário no servidor). Por isso, a ferramenta
`consultar_dados_reciclagem` foi implementada de forma restrita: o modelo só escolhe parâmetros
(mês, material, métrica, operação) dentro de listas fixas, e uma função Python comum faz o cálculo
com pandas — sem nenhum código gerado pelo modelo sendo executado.

### Por que um loop de tool-calling manual em vez do `AgentExecutor` do LangChain?

A primeira versão usava `create_tool_calling_agent` + `AgentExecutor`, a forma "oficial" do
LangChain para agentes com ferramentas. Na prática, essa combinação com os modelos Gemini mais
recentes (via `langchain-google-genai`) falhava com o erro `Function call is missing a
thought_signature in functionCall parts` — um requisito novo da API do Gemini para chamadas de
função que o `AgentExecutor` ainda não propaga corretamente entre turnos (ele reconstrói o
histórico de mensagens ao chamar `.stream()` internamente e perde essa informação). Em vez de
mascarar o problema, o agente foi reescrito com um **loop de tool-calling manual** (`llm.bind_tools`
+ um laço que executa as ferramentas chamadas e devolve o resultado ao modelo), o que resolveu o
erro e deixou o fluxo do agente mais explícito e fácil de acompanhar (veja `responder()` em
`src/agente.py`).

### Cadeia de fallback entre modelos (cota gratuita esgotada)

Na prática, o nível gratuito do Gemini é bem restrito para modelos de chat (algumas contas têm
limite de ~20 requisições/dia para um único modelo). Como cada modelo tem uma cota separada, o
agente tenta uma **cadeia de modelos em ordem** e só passa para o próximo quando o erro é
especificamente de cota esgotada ou modelo indisponível (nunca mascarando um bug real):

```
gemini-flash-latest → gemini-3.5-flash-lite → gemini-3.1-flash-lite → gemini-3.5-flash → gemma-4-26b-a4b-it
```

Os quatro primeiros são modelos Gemini "de verdade" (qualidade equivalente, testada com as mesmas
perguntas deste README); o `gemma-4-26b-a4b-it` (família de modelo aberta, com tool-calling
confirmado) fica por último, como rede de segurança — melhor uma resposta um pouco mais simples do
que o app parar de funcionar. Se todos os modelos da cadeia esgotarem, a interface mostra uma
mensagem clara pedindo para tentar novamente mais tarde, em vez de travar ou quebrar.

### Resiliência da cota de embeddings

A cota gratuita do modelo de embeddings (`gemini-embedding-001`) tem um limite **por minuto**,
separado da cota de chat. Como cada pergunta faz uma chamada de embedding (para buscar o trecho
relevante) e reconstruir o índice pode gerar várias chamadas de uma vez, é comum esbarrar nesse
limite em uma sessão de testes mais intensa. Em vez de propagar o erro `429 RESOURCE_EXHAUSTED` na
primeira falha, `GoogleGenerativeAIEmbeddingsComRetry` (`src/embeddings.py`) lê o tempo de espera
sugerido pela própria API (`retryDelay`) e tenta de novo automaticamente, tanto na busca quanto na
reconstrução do índice.

Além disso, como `vectorstore/` é gerado (não é versionado no Git — veja `.gitignore`), a aplicação
constrói o índice sozinha na primeira execução caso ele não exista, a partir dos PDFs já presentes
em `documentos/`. Isso evita que um clone novo do repositório (por exemplo, o primeiro deploy na
OCI) exija rodar `python src/ingestao.py` manualmente antes do primeiro `streamlit run`.

## Gestão de documentos pela interface

Além do manual padrão, a barra lateral permite adicionar, listar e remover PDFs sem precisar rodar
nenhum comando — o próprio Streamlit cuida da reindexação:

- **Upload** — arraste um ou mais PDFs; cada um passa por quatro validações antes de ser salvo:
  - **Duplicado** — se já existe um arquivo com esse nome em `documentos/`, o upload é ignorado
    (evita reprocessar/sobrescrever à toa).
  - **Arquivo inválido** — se o PDF estiver corrompido ao ponto de nem o `pypdf` conseguir abrir a
    estrutura do arquivo, o upload é recusado com uma mensagem clara em vez de derrubar a página.
  - **Texto corrompido** — alguns PDFs (comum em documentos de sites de legislação mais antigos)
    embutem uma fonte com codificação customizada que o `pypdf` não consegue mapear de volta para
    texto legível — o conteúdo extraído vem com caracteres deslocados/ilegíveis. Como indexar esse
    texto geraria embeddings inúteis (nunca encontrados em nenhuma busca), esses arquivos são
    detectados e recusados também, em vez de aceitos silenciosamente.
  - **Assunto** — uma chamada rápida ao Gemini verifica se o conteúdo é sobre reciclagem, gestão de
    resíduos ou sustentabilidade; documentos fora desse escopo são recusados antes de indexar.
- **Documentos atuais** — lista os PDFs indexáveis com um botão 🗑️ para remover cada um. O manual
  padrão (`manual_reciclagem.pdf`) aparece com 🔒 em vez do botão de remover — é o documento base do
  desafio, protegido contra exclusão acidental pela interface.
- **Reconstruir índice FAISS** — reconstrói o índice a partir do que estiver em `documentos/` no
  momento. Um indicador (✅/⚠️) mostra se o índice está em dia com os documentos atuais; excluir
  **todos** os PDFs (exceto o manual protegido) e reconstruir remove o índice por completo (a busca
  em documentos fica indisponível até um novo upload).
- O relatório de reciclagem (`relatorio_reciclagem_mensal.csv`) **não** é gerenciável por upload
  aqui — a ferramenta de cálculo espera colunas fixas (mês, material, % reciclado, kg), então trocar
  o CSV livremente quebraria essa ferramenta.

## Tecnologias e ferramentas

- **Python 3.11+**
- **LangChain** (`langchain`, `langchain-community`, `langchain-text-splitters`,
  `langchain-google-genai`) — orquestração do agente e das ferramentas
- **Google Gemini** — `gemini-flash-latest` (geração de respostas e tool-calling) e
  `models/gemini-embedding-001` (embeddings)
- **pypdf** — leitura do PDF (via `PyPDFLoader`)
- **pandas** — leitura e agregação do relatório mensal de reciclagem (CSV)
- **FAISS** (`faiss-cpu`) — índice vetorial local, sem depender de banco de dados externo
- **Streamlit** — interface web de chat
- **python-dotenv** — carregamento da chave de API a partir de `.env`

## Estrutura do projeto

```
.
├── app.py                    # interface de chat (Streamlit)
├── src/
│   ├── ingestao.py                    # lê os PDFs e constrói o índice FAISS
│   ├── agente.py                       # agente com tool-calling (busca no manual + pandas)
│   └── embeddings.py                   # embeddings com retry automático (cota por minuto)
├── tests/
│   └── test_agente.py                  # testes unitários da lógica de cálculo (pandas)
├── documentos/
│   ├── manual_reciclagem.pdf           # manual de reciclagem (fonte para o RAG)
│   └── relatorio_reciclagem_mensal.csv # relatório mensal de % reciclado (fonte para o pandas)
├── assets/
│   └── banner-challenge.png            # banner do desafio (Alura + Oracle)
├── vectorstore/               # índice FAISS gerado (não versionado)
├── requirements.txt
├── requirements-dev.txt       # dependências extras para rodar os testes
├── .env.example
├── LICENSE
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
5. (Opcional) Gere o índice vetorial manualmente a partir do(s) PDF(s) em `documentos/`:
   ```bash
   python src/ingestao.py
   ```
   *(o CSV de reciclagem é lido diretamente com pandas em tempo de consulta, não precisa de
   ingestão/índice. Esse passo também é opcional para os PDFs: se o índice não existir, a
   aplicação o constrói sozinha na primeira execução do passo 6.)*
6. Rode a aplicação:
   ```bash
   streamlit run app.py
   ```
7. Acesse http://localhost:8501 no navegador.

Para adicionar, listar ou remover PDFs depois, use a seção "Gestão de documentos" na barra lateral
da própria aplicação (veja [Gestão de documentos pela interface](#gestão-de-documentos-pela-interface))
— não precisa rodar comandos manualmente. Para usar outro CSV, ajuste o nome do arquivo e as
colunas esperadas em `src/agente.py`.

## Testes

A lógica de cálculo sobre o CSV (`calcular_dados_reciclagem`, em `src/agente.py`) tem testes
unitários que **não** chamam a API do Gemini nem precisam do índice FAISS — rodam em segundos e
sem chave de API configurada:

```bash
pip install -r requirements-dev.txt
pytest
```

## Exemplos de perguntas e respostas

**Pergunta:** Qual a cor da lixeira para plástico?
**Resposta:** A cor da lixeira para plástico é vermelha.
*(ferramenta usada: `buscar_no_manual`)*

**Pergunta:** Como devo descartar equipamentos eletrônicos?
**Resposta:** De acordo com o manual de reciclagem, os equipamentos eletrônicos (como notebooks,
monitores, celulares corporativos, cabos e baterias) devem ser descartados seguindo o
procedimento: 1) entrega exclusiva ao setor de TI; 2) preenchimento do Formulário de Baixa de
Ativo (FBA-07); 3) o setor de TI realiza a limpeza dos dados antes de enviar o material à empresa
parceira especializada em descarte de e-lixo (GreenTech Reciclagem).
*(ferramenta usada: `buscar_no_manual`)*

**Pergunta:** Qual a média do percentual reciclado de papel nos últimos meses?
**Resposta:** A média do percentual reciclado de papel nos últimos meses é de 72,07%.
*(ferramenta usada: `consultar_dados_reciclagem`, calculado com pandas sobre o CSV)*

**Pergunta:** Qual o percentual reciclado de metal em março de 2026?
**Resposta:** O percentual reciclado de metal em março de 2026 foi de 76,10%.
*(ferramenta usada: `consultar_dados_reciclagem`)*

**Pergunta:** Qual a capital da França?
**Resposta:** Desculpe, não encontrei a informação nos documentos disponíveis. Minhas ferramentas
são restritas a responder dúvidas sobre o manual de reciclagem e os relatórios de reciclagem da
empresa.
*(nenhuma ferramenta chamada — o agente reconhece que a pergunta está fora do escopo, em vez de
inventar uma resposta)*

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

### Evidência do deploy

**Link público:** http://163.176.181.35:8501

A aplicação está em produção na **Oracle Cloud Infrastructure (OCI)**, em uma VM **Oracle Linux 9**
(shape `VM.Standard.E2.1.Micro`, Always Free). A imagem abaixo mostra a aplicação respondendo
corretamente uma pergunta sobre o manual de reciclagem, recuperando e citando as fontes:

![Streamlit Agent rodando na OCI](deployment-evidence/streamlit-deployed-oci-2026-08-01.png)

Você pode acessar o link acima para testar a aplicação ao vivo. Exemplos de perguntas para
experimentar:
- "Quais materiais podem ser reciclados segundo o manual?"
- "Como devo separar o lixo orgânico?"
- "Qual a média do percentual reciclado de papel nos últimos meses?"

## Licença

Distribuído sob a licença MIT — veja [LICENSE](LICENSE) para mais detalhes.
