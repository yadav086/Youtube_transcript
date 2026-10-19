import os
import re
import uuid
import time
import streamlit as st

from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi

from chonkie import TokenChunker
import tiktoken

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel,
)

from langchain_community.embeddings import JinaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_compressors import JinaRerank
from langchain_community.retrievers import TavilySearchAPIRetriever

from langchain_classic.retrievers import (
    EnsembleRetriever,
    ContextualCompressionRetriever,
)

from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="YouTube RAG Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0b1020 0%, #111827 100%);
        color: #f8fafc;
    }

    [data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #273449;
    }

    .hero {
        padding: 25px;
        border-radius: 18px;
        background: linear-gradient(120deg, #312e81, #4338ca, #0891b2);
        margin-bottom: 22px;
        color: white;
    }

    .hero h1 {
        margin-bottom: 8px;
        font-size: 2rem;
    }

    .hero p {
        margin-bottom: 0;
        color: #e0f2fe;
    }

    .status-card {
        background: #172033;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 10px;
    }

    div[data-testid="stChatMessage"] {
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 12px;
        background: #172033;
    }

    .stButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

defaults = {
    "connected": False,
    "rag_chain": None,
    "chat_history": [],
    "video_url": "",
    "video_id": "",
    "namespace": None,
    "connection_info": {},
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# YOUTUBE HELPERS
# --------------------------------------------------

def extract_video_id(url: str) -> str | None:
    """Extract an 11-character YouTube video ID."""

    patterns = [
        r"(?:youtube\.com/watch\?.*?v=)([A-Za-z0-9_-]{11})",
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/embed/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/shorts/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/live/)([A-Za-z0-9_-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def clean_transcript_text(text: str) -> str:
    """Remove common transcript noise and normalize whitespace."""

    if not text:
        return ""

    text = re.sub(
        r"\[(?:music|laughter|applause|inaudible)[^\]]*\]",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\s+'\s*([a-zA-Z])", r"'\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    if text:
        text = text[0].upper() + text[1:]

    return text


def get_transcript(youtube_url: str, target_lang: str = "en") -> str:
    """Fetch a transcript, translating it into English when supported."""

    video_id = extract_video_id(youtube_url)

    if not video_id:
        raise ValueError(
            "Invalid YouTube URL. Enter a valid video, Shorts, or live URL."
        )

    api = YouTubeTranscriptApi()
    transcript_list = api.list(video_id)

    try:
        transcript = transcript_list.find_transcript([target_lang])
    except Exception:
        # Prefer a translatable transcript if English is unavailable.
        candidates = list(transcript_list)
        transcript = next(
            (item for item in candidates if item.is_translatable),
            None,
        )

        if transcript is None:
            raise ValueError(
                "No English or translatable transcript is available "
                "for this video."
            )

        transcript = transcript.translate(target_lang)

    fetched = transcript.fetch()

    text = " ".join(
        item.text if hasattr(item, "text") else item["text"]
        for item in fetched
    )

    text = clean_transcript_text(text)

    if not text:
        raise ValueError("The transcript is empty.")

    return text


# --------------------------------------------------
# BUILD DOCUMENTS
# --------------------------------------------------

def create_documents(transcript: str) -> list[Document]:
    """Chunk the transcript into LangChain documents."""

    tokenizer = tiktoken.get_encoding("cl100k_base")

    chunker = TokenChunker(
        tokenizer=tokenizer,
        chunk_size=512,
        chunk_overlap=50,
    )

    chunks = chunker.chunk(transcript)

    documents = []

    for number, chunk in enumerate(chunks):
        chunk_text = chunk.text.strip()

        if not chunk_text:
            continue

        documents.append(
            Document(
                page_content=chunk_text,
                metadata={
                    "chunk_number": number + 1,
                    "start_index": getattr(chunk, "start_index", None),
                    "end_index": getattr(chunk, "end_index", None),
                },
            )
        )

    if not documents:
        raise ValueError("Unable to create transcript chunks.")

    return documents


# --------------------------------------------------
# PINECONE INDEX
# --------------------------------------------------

def get_or_create_index(
    pinecone_client: Pinecone,
    index_name: str,
    dimension: int = 1024,
):
    """Create the index if necessary and wait until it is ready."""

    if not pinecone_client.has_index(index_name):
        pinecone_client.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

    # Wait for index readiness before writing vectors.
    for _ in range(60):
        description = pinecone_client.describe_index(index_name)

        if description.status["ready"]:
            return

        time.sleep(2)

    raise TimeoutError("Pinecone index did not become ready in time.")


# --------------------------------------------------
# BUILD RAG PIPELINE
# --------------------------------------------------

def build_rag_pipeline(
    youtube_url: str,
    google_api_key: str,
    pinecone_api_key: str,
    tavily_api_key: str,
    jina_api_key: str,
):
    """Create the transcript index and hybrid RAG chain."""

    # Set credentials for integrations that read environment variables.
    os.environ["GOOGLE_API_KEY"] = google_api_key
    os.environ["PINECONE_API_KEY"] = pinecone_api_key
    os.environ["TAVILY_API_KEY"] = tavily_api_key
    os.environ["JINA_API_KEY"] = jina_api_key

    video_id = extract_video_id(youtube_url)

    if not video_id:
        raise ValueError("Please enter a valid YouTube URL.")

    # 1. Transcript
    with st.status("Processing YouTube transcript...", expanded=True) as status:
        st.write("Fetching and cleaning transcript")
        transcript = get_transcript(youtube_url)

        st.write("Creating token chunks")
        documents = create_documents(transcript)

        st.write(f"Created {len(documents)} transcript chunks")

        # 2. Embeddings
        st.write("Initializing Jina embeddings")
        embedding_model = JinaEmbeddings(
            model_name="jina-embeddings-v3",
            dimensions=1024,
        )

        # 3. Pinecone
        st.write("Connecting to Pinecone")
        pinecone_client = Pinecone(api_key=pinecone_api_key)

        index_name = "testing-pinecode-emb"
        get_or_create_index(pinecone_client, index_name)

        # Use a separate namespace for each connection.
        namespace = f"youtube-{video_id}-{uuid.uuid4().hex[:12]}"

        st.write("Uploading transcript vectors")
        vector_store = PineconeVectorStore.from_documents(
            documents=documents,
            embedding=embedding_model,
            index_name=index_name,
            namespace=namespace,
            ids=[str(uuid.uuid4()) for _ in documents],
            batch_size=100,
        )

        # 4. Dense retriever
        dense_retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5},
        )

        # 5. Sparse BM25 retriever
        sparse_retriever = BM25Retriever.from_documents(documents)
        sparse_retriever.k = 5

        # 6. Tavily web retriever
        web_retriever = TavilySearchAPIRetriever(
            include_domains=["google.com"],
            search_depth="advanced",
        )
        web_retriever.k = 5

        # 7. Hybrid retrieval
        hybrid_retriever = EnsembleRetriever(
            retrievers=[
                dense_retriever,
                sparse_retriever,
                web_retriever,
            ],
            weights=[0.4, 0.2, 0.4],
        )

        # 8. Reranking
        reranker = JinaRerank(
            model="jina-reranker-v2-base-multilingual",
            top_n=5,
        )

        compressor = ContextualCompressionRetriever(
            base_retriever=hybrid_retriever,
            base_compressor=reranker,
        )

        # 9. Gemini
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            temperature=0,
            max_tokens=1000,
        )

        # Ground answers in retrieved context.
        prompt = ChatPromptTemplate.from_template(
            """
You are a precise YouTube transcript question-answering assistant.

Rules:
1. Answer using ONLY the Context below.
2. Do not use outside knowledge or assumptions.
3. If the Context does not contain enough information to answer,
   respond exactly: I don't know.
4. Keep answers clear, direct, and relevant to the question.
5. For summary requests, follow the requested number of points and format.

Context:
{context}

Question:
{question}

Answer:
"""
        )

        def format_documents(docs):
            if not docs:
                return ""

            return "\n\n".join(
                doc.page_content for doc in docs if doc.page_content
            )

        rag_chain = (
            RunnableParallel(
                {
                    "context": compressor | RunnableLambda(format_documents),
                    "question": RunnablePassthrough(),
                }
            )
            | prompt
            | llm
            | StrOutputParser()
        )

        status.update(
            label="YouTube RAG pipeline connected",
            state="complete",
            expanded=False,
        )

    return rag_chain, video_id, namespace, len(documents)


# --------------------------------------------------
# SIDEBAR: INPUTS AND CONNECTION
# --------------------------------------------------

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.caption("Enter your YouTube URL and API credentials.")

    youtube_url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        key="youtube_url_input",
    )

    with st.expander("🔐 API Credentials", expanded=True):
        google_key = st.text_input(
            "GOOGLE_API_KEY",
            type="password",
            help="Used by Gemini.",
        )
        pinecone_key = st.text_input(
            "PINECONE_API_KEY",
            type="password",
            help="Used for vector storage and retrieval.",
        )
        tavily_key = st.text_input(
            "TAVILY_API_KEY",
            type="password",
            help="Used by Tavily web search.",
        )
        jina_key = st.text_input(
            "JINA_API_KEY",
            type="password",
            help="Used for embeddings and reranking.",
        )

    connect_clicked = st.button(
        "🔌 Connect & Process",
        type="primary",
        disabled=not all(
            [
                youtube_url.strip(),
                google_key.strip(),
                pinecone_key.strip(),
                tavily_key.strip(),
                jina_key.strip(),
            ]
        ),
    )

    if st.button("🗑️ Reset Chat"):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.caption("Credentials are entered for this app session. "
               "Do not share screenshots containing secrets.")


# --------------------------------------------------
# CONNECT BUTTON
# --------------------------------------------------

if connect_clicked:
    video_id = extract_video_id(youtube_url.strip())

    if not video_id:
        st.error("Please enter a valid YouTube URL.")
    else:
        # Clear the old connection before processing a new video.
        st.session_state.connected = False
        st.session_state.rag_chain = None
        st.session_state.chat_history = []

        try:
            with st.spinner("Building your YouTube RAG application..."):
                chain, video_id, namespace, chunk_count = build_rag_pipeline(
                    youtube_url=youtube_url.strip(),
                    google_api_key=google_key.strip(),
                    pinecone_api_key=pinecone_key.strip(),
                    tavily_api_key=tavily_key.strip(),
                    jina_api_key=jina_key.strip(),
                )

            st.session_state.rag_chain = chain
            st.session_state.connected = True
            st.session_state.video_url = youtube_url.strip()
            st.session_state.video_id = video_id
            st.session_state.namespace = namespace
            st.session_state.connection_info = {
                "chunks": chunk_count,
                "video_id": video_id,
            }

            st.success("Connected successfully. You can now chat!")

        except Exception as exc:
            # Do not display credentials or environment variables.
            st.session_state.connected = False
            st.session_state.rag_chain = None
            st.error(
                "Connection failed. Verify the URL, API credentials, "
                "transcript availability, model access, Pinecone index "
                "configuration, and installed package versions."
            )
            st.exception(exc)


# --------------------------------------------------
# MAIN PAGE
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>🎬 YouTube RAG Intelligence</h1>
        <p>
            Ask questions, summarize podcasts, and explore video transcripts
            using hybrid retrieval and generative AI.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([2, 1])

with left:
    st.subheader("💬 Chat with your YouTube video")

    if not st.session_state.connected:
        st.info(
            "Enter a YouTube URL and all four API keys in the sidebar, "
            "then click Connect & Process to activate the chat."
        )

    else:
        info = st.session_state.connection_info

        st.success("Pipeline status: Connected")

        st.markdown(
            f"""
            <div class="status-card">
                <b>Video ID:</b> {info["video_id"]}<br>
                <b>Transcript chunks:</b> {info["chunks"]}<br>
                <b>Retrieval:</b> Pinecone + BM25 + Tavily<br>
                <b>Reranking:</b> Jina
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Display existing conversation.
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input is active only after successful connection.
        user_question = st.chat_input(
            "Ask about this podcast or request a summary..."
        )

        if user_question:
            st.session_state.chat_history.append(
                {"role": "user", "content": user_question}
            )

            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):
                try:
                    with st.spinner("Searching and generating an answer..."):
                        answer = st.session_state.rag_chain.invoke(
                            user_question
                        )

                    st.markdown(answer)

                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer}
                    )

                except Exception:
                    st.error(
                        "I couldn't generate an answer. Check your API "
                        "quotas, network connection, and service access."
                    )

with right:
    st.subheader("✨ Quick prompts")

    quick_prompts = [
        "Summarize this YouTube podcast in 5 numbered points.",
        "What are the main topics discussed?",
        "List the key takeaways from the podcast.",
        "What examples or stories are mentioned?",
    ]

    for prompt_text in quick_prompts:
        if st.button(prompt_text, key=f"quick_{prompt_text}"):
            if st.session_state.connected:
                st.session_state.chat_history.append(
                    {"role": "user", "content": prompt_text}
                )
                try:
                    with st.spinner("Generating summary..."):
                        answer = st.session_state.rag_chain.invoke(
                            prompt_text
                        )

                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer}
                    )
                    st.rerun()

                except Exception:
                    st.error("Unable to generate a response.")
            else:
                st.warning("Connect to a video first.")

    st.divider()
    st.caption(
        "Note: Answers depend on retrieved transcript chunks. "
        "Web results may contain information outside the transcript, "
        "so the assistant is instructed to answer only from its context."
    )
