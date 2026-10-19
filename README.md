# YouTube RAG Intelligence 🎬

An interactive **Retrieval-Augmented Generation (RAG)** application that
lets users ask questions about a YouTube video or podcast after
processing its transcript. Built with Streamlit, LangChain, Jina AI,
Pinecone, Tavily, and Google Gemini.

## ✨ Features

-   **YouTube URL input** for supported video, Shorts, and live URL
    formats.
-   **Transcript extraction and cleaning** using
    `youtube-transcript-api`.
-   **English transcript translation** when a suitable translatable
    transcript is available.
-   **Token-based chunking** with Chonkie and `tiktoken`.
-   **Semantic vector search** using Jina Embeddings v3 and Pinecone.
-   **Hybrid retrieval** combining dense retrieval, BM25 keyword search,
    and Tavily web search.
-   **Reranking** using Jina Reranker.
-   **Conversational Q&A and summaries** powered by Google Gemini.
-   **Streamlit chat interface** with quick prompts and chat history for
    the current session.
-   **Per-connection Pinecone namespace** to keep each processed video's
    vectors logically separated.

## 🧰 Technology Stack

  Component                 Technology
  ------------------------- ------------------------
  User interface            Streamlit
  Transcript                YouTube Transcript API
  Chunking / tokenization   Chonkie, tiktoken
  Embeddings                Jina Embeddings v3
  Vector database           Pinecone
  Keyword retrieval         BM25
  Web retrieval             Tavily
  Reranking                 Jina Reranker
  LLM                       Google Gemini
  RAG orchestration         LangChain

## 🏗️ Architecture

``` text
YouTube URL + API credentials
             |
             v
   Fetch and clean transcript
             |
             v
      Split into chunks
             |
             v
       Jina embeddings
             |
             v
    Store vectors in Pinecone
             |
             v
 Question --> Hybrid retrieval
              |      |      |
           Pinecone BM25  Tavily
              \      |      /
               Jina reranker
                    |
                    v
          Retrieved context + question
                    |
                    v
                Gemini LLM
                    |
                    v
             Answer in Streamlit
```

## 🚀 Getting Started

### 1. Requirements

-   Python 3.10 or newer is recommended.
-   A Google AI API key with access to a supported Gemini model.
-   A Pinecone API key and permission to create/use a serverless index.
-   A Tavily API key.
-   A Jina API key.

YouTube transcripts must be available for the selected video. Some
videos disable transcripts or restrict access.

### 2. Clone the repository

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd youtube_rag_chatbot
```

Replace the placeholder with your repository URL.

### 3. Create and activate a virtual environment

**Windows**

``` powershell
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

Create a `requirements.txt` file with the following packages, then
install them:

``` text
streamlit
python-dotenv
youtube-transcript-api
chonkie
tiktoken
langchain-core
langchain-community
langchain-classic
langchain-google-genai
langchain-pinecone
pinecone
rank-bm25
```

Install:

``` bash
pip install -r requirements.txt
```

Integration classes can move between package versions. If an import
fails, check the current documentation for the installed LangChain
integration and install the compatible package version. Pin tested
dependency versions before deploying.

### 5. Configure API keys

The app accepts the following credentials in its sidebar:

-   `GOOGLE_API_KEY`
-   `PINECONE_API_KEY`
-   `TAVILY_API_KEY`
-   `JINA_API_KEY`

For local development, you may alternatively create a `.env` file:

``` dotenv
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
TAVILY_API_KEY=your_tavily_api_key
JINA_API_KEY=your_jina_api_key
```

Never commit `.env` or real API keys to GitHub. Add this to
`.gitignore`:

``` text
.venv/
__pycache__/
.env
.streamlit/secrets.toml
*.pyc
```

For deployed environments, prefer the hosting platform's secret manager.
Do not paste API keys into source code, screenshots, README files, or
LinkedIn posts.

### 6. Run the app

``` bash
streamlit run app.py
```

Open the local URL printed in the terminal (usually
`http://localhost:8501`), enter a YouTube URL and the required API keys,
then select **Connect & Process**. When processing succeeds, use the
chatbox to ask questions or request a numbered summary.

## 💡 Example Prompts

-   `Summarize this YouTube podcast in 5 numbered points.`
-   `What are the main topics discussed?`
-   `List the key takeaways from the podcast.`
-   `What examples or stories are mentioned?`
-   `What does the speaker say about [topic]?`

## 🔐 Security and Privacy

-   API credentials are sensitive secrets. Keep them out of source
    control and logs.
-   Password fields mask credentials visually, but this alone does not
    provide full secret isolation.
-   The current implementation sets API keys as environment variables
    for integrations; those values remain accessible to the server
    process.
-   Use managed secrets, least-privilege credentials, access controls,
    and appropriate retention policies before deploying publicly.
-   Avoid displaying raw exception details to end users in a public
    deployment.

## ⚠️ Important Notes and Limitations

1.  **Grounding:** The prompt instructs the model to answer from
    retrieved context and return `I don't know.` if the context does not
    contain the answer. This is a prompt-level constraint, not a
    guarantee that the model will never make an unsupported claim.
2.  **Tavily results:** The configured hybrid retriever includes web
    search. Retrieved context can therefore contain web content, not
    only the YouTube transcript. For strict transcript-only Q&A, remove
    Tavily from the ensemble and retrieve only from the transcript
    index.
3.  **Gemini model ID:** Confirm that the model name configured in
    `app.py` is available to your Google AI account. Replace it with a
    currently supported model ID if necessary.
4.  **Pinecone dimension:** The index dimension must match the
    configured embedding output dimension (1024 in this implementation).
    An existing index with another dimension cannot accept these
    vectors.
5.  **Index configuration:** The example uses a Pinecone serverless
    index named `testing-pinecode-emb` in AWS `us-east-1`. Change these
    settings to suit your account and deployment region.
6.  **Cost and rate limits:** Transcript processing, embedding,
    reranking, web retrieval, and LLM calls may incur charges or be
    subject to quotas.
7.  **Production readiness:** Before public deployment, test package
    compatibility, add retry and timeout handling, validate input URLs,
    limit resource use, improve structured logging, and consider caching
    previously processed transcripts.

## 🗺️ Future Improvements

-   Add source citations and expandable retrieved passages to each
    answer.
-   Cache processed transcripts and embeddings to reduce repeated
    processing and cost.
-   Add transcript download and summary export (TXT / Markdown / PDF).
-   Add configurable retrieval weights and a transcript-only mode.
-   Add automated tests, typed configuration, dependency lock files, and
    CI checks.
-   Deploy to Streamlit Community Cloud or another secured hosting
    platform.

## 🤝 Contributing

Issues, ideas, and pull requests are welcome. Please do not include API
keys, private transcripts, or other sensitive data in issues or
examples.

## 📄 License

Choose and add a license before publishing this repository. Until a
license is added, reuse and redistribution permissions are not
explicitly granted.

------------------------------------------------------------------------

**Built as a hands-on project exploring hybrid retrieval, vector
databases, reranking, and conversational RAG for YouTube content.**
