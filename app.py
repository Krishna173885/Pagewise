import json
import os
import re
import time
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

import auth

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_community.vectorstores import FAISS


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CHAT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")  # override in .env if needed
EMBEDDING_MODEL = "models/gemini-embedding-001"

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
TOTAL_K = 8          # total chunks sent to the model
EMBED_BATCH = 40     # chunks embedded per batch
EMBED_INTERVAL = 30  # min seconds between batches (free tier: 100 requests/min)
HISTORY_TURNS = 6    # past messages used to resolve follow-up questions

if not GOOGLE_API_KEY:
    st.set_page_config(page_title="Pagewise", page_icon="📚")
    st.error("Google API key not found. Please add GOOGLE_API_KEY to your .env file.")
    st.stop()


# --------------------------------------------------
# 2. Models (created once per session)
# --------------------------------------------------

@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )


@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY,
        max_retries=1,  # default retries hammer the quota when you're rate limited
    )


# --------------------------------------------------
# 3. PDF -> Documents (with filename + page metadata)
# --------------------------------------------------

def build_documents(pdf_files):
    """Return (chunks, filenames, unreadable_filenames).

    Each page is split separately so every chunk keeps an accurate
    {"source": filename, "page": page_number} in its metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    chunks, filenames, unreadable = [], [], []

    for pdf in pdf_files:
        filenames.append(pdf.name)
        reader = PdfReader(pdf)
        file_chunks = 0

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if not page_text:
                continue

            page_doc = Document(
                page_content=page_text,
                metadata={"source": pdf.name, "page": page_number},
            )
            split_docs = splitter.split_documents([page_doc])
            chunks.extend(split_docs)
            file_chunks += len(split_docs)

        if file_chunks == 0:
            unreadable.append(pdf.name)

    return chunks, filenames, unreadable


# --------------------------------------------------
# 4. Retrieval (balanced across files)
# --------------------------------------------------

def retrieve(question, vector_store, filenames):
    """Single file: MMR search for diverse chunks.
    Multiple files: retrieve from each file separately so no document is
    drowned out, which fixes questions like "what are the 2 PDFs about?".
    """
    if len(filenames) <= 1:
        return vector_store.max_marginal_relevance_search(
            question, k=TOTAL_K, fetch_k=TOTAL_K * 4
        )

    per_file = max(2, TOTAL_K // len(filenames))
    fetch_k = vector_store.index.ntotal  # search everything, then filter

    docs = []
    for name in filenames:
        docs.extend(
            vector_store.similarity_search(
                question,
                k=per_file,
                fetch_k=fetch_k,
                filter={"source": name},
            )
        )
    return docs


def format_context(docs):
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[Source: {source}, page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def unique_sources(docs):
    seen = sorted(
        {(d.metadata.get("source", "unknown"), d.metadata.get("page", 0)) for d in docs}
    )
    return [f"{name} (p. {page})" for name, page in seen]


# --------------------------------------------------
# 5. Chains (modern LCEL, no load_qa_chain)
# --------------------------------------------------

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are a friendly assistant that helps people understand their uploaded
PDF documents. Follow these rules:

1. Greetings, thanks, small talk, and questions about this conversation
   (for example "hello" or "what did I ask before?"): reply naturally and
   briefly using the earlier conversation, and invite the user to ask about
   their documents. Do not use the context for these.
2. Questions about the content of the documents: answer using only the
   context below. The context comes from one or more documents; each excerpt
   is labeled with its source file and page. When the question concerns
   several documents, address each one separately and name it. Give a clear,
   detailed answer.
3. If a question is about the documents but the answer is not in the context,
   respond exactly: "Answer is not available in the context."
4. Questions that are unrelated to the documents and that you cannot answer
   from them (for example the user's API quota): say briefly that you can only
   help with the uploaded documents.
5. Never invent information about the documents.

Earlier conversation:
{history}

Context:
{context}

Question:
{question}

Answer:"""
)


def format_history(messages):
    recent = messages[-HISTORY_TURNS:]
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in recent)


def answer_question(question, messages):
    """One LLM call per question. Follow-ups are handled without an extra
    rewrite call: the previous user question is added to the search query,
    and the recent history is given to the model in the prompt."""
    search_query = question
    previous_user = [m["content"] for m in messages if m["role"] == "user"]
    if previous_user:
        search_query = f"{previous_user[-1]} {question}"

    docs = retrieve(
        search_query,
        st.session_state["vector_store"],
        st.session_state["filenames"],
    )
    if not docs:
        return "No relevant information was found.", []

    chain = ANSWER_PROMPT | get_llm() | StrOutputParser()
    answer = chain.invoke(
        {
            "history": format_history(messages) or "(none)",
            "context": format_context(docs),
            "question": question,
        }
    )
    return answer, unique_sources(docs)


def build_vector_store(chunks, progress=None):
    """Embed chunks in throttled batches so the free-tier per-minute limit
    isn't exceeded, and wait out any 429 the API still returns."""
    embeddings = get_embeddings()
    vector_store = None
    last_start = 0.0

    for i in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[i : i + EMBED_BATCH]

        if i > 0:
            wait = EMBED_INTERVAL - (time.time() - last_start)
            if wait > 0:
                if progress:
                    progress(i / len(chunks), f"Waiting {int(wait)}s to respect rate limits...")
                time.sleep(wait)
        last_start = time.time()

        for attempt in range(4):
            try:
                if vector_store is None:
                    vector_store = FAISS.from_documents(batch, embeddings)
                else:
                    vector_store.add_documents(batch)
                break
            except Exception as e:
                message = str(e)
                is_rate_limit = "429" in message or "ResourceExhausted" in type(e).__name__
                if not is_rate_limit or attempt == 3:
                    raise
                match = re.search(r"retry in (\d+(?:\.\d+)?)s", message)
                delay = float(match.group(1)) + 2 if match else 30
                if progress:
                    progress(i / len(chunks), f"Rate limited, retrying in {int(delay)}s...")
                time.sleep(delay)

        if progress:
            progress(min(i + EMBED_BATCH, len(chunks)) / len(chunks),
                     f"Embedded {min(i + EMBED_BATCH, len(chunks))}/{len(chunks)} chunks")

    return vector_store


# --------------------------------------------------
# 6. Streamlit app
# --------------------------------------------------

def user_dir(user):
    # Safe as a folder name: auth.py only allows [a-z0-9_] usernames.
    path = os.path.join("data", "users", user)
    os.makedirs(path, exist_ok=True)
    return path


def save_index(user):
    folder = user_dir(user)
    st.session_state["vector_store"].save_local(os.path.join(folder, "index"))
    with open(os.path.join(folder, "files.json"), "w") as f:
        json.dump(st.session_state["filenames"], f)


def save_messages(user):
    with open(os.path.join(user_dir(user), "messages.json"), "w") as f:
        json.dump(st.session_state["messages"], f)


def load_user_state(user):
    """Restore this user's saved PDFs and chat once per login session."""
    if st.session_state.get("loaded_for") == user:
        return

    folder = user_dir(user)
    index_path = os.path.join(folder, "index")
    files_path = os.path.join(folder, "files.json")
    messages_path = os.path.join(folder, "messages.json")

    try:
        if os.path.isdir(index_path) and os.path.exists(files_path):
            # Safe to deserialize: this index was written by this app,
            # inside this user's own folder.
            st.session_state["vector_store"] = FAISS.load_local(
                index_path, get_embeddings(), allow_dangerous_deserialization=True
            )
            with open(files_path) as f:
                st.session_state["filenames"] = json.load(f)
        if os.path.exists(messages_path):
            with open(messages_path) as f:
                st.session_state["messages"] = json.load(f)
    except Exception as e:
        st.warning(f"Could not restore your saved documents: {e}")

    st.session_state["loaded_for"] = user


def process_pdfs(pdf_docs):
    chunks, filenames, unreadable = build_documents(pdf_docs)

    for name in unreadable:
        st.warning(f"No readable text in '{name}' (it may be a scanned image).")

    if not chunks:
        st.error("No readable text was found in any of the PDFs.")
        return

    # Kept in memory per session: no shared folder on disk, and no
    # need for allow_dangerous_deserialization.
    bar = st.progress(0.0, text="Embedding chunks...")
    st.session_state["vector_store"] = build_vector_store(
        chunks, progress=lambda frac, text: bar.progress(frac, text=text)
    )
    bar.empty()
    st.session_state["filenames"] = [f for f in filenames if f not in unreadable]
    st.session_state["messages"] = []
    save_index(st.session_state["user"])
    save_messages(st.session_state["user"])

    st.success(
        f"Processed {len(st.session_state['filenames'])} file(s) "
        f"into {len(chunks)} chunks."
    )


def main():
    st.set_page_config(page_title="Pagewise", page_icon="📚", layout="wide")
    user = auth.require_login()  # stops here until the user is signed in
    st.header("📚 Pagewise using Gemini")

    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("vector_store", None)
    st.session_state.setdefault("filenames", [])
    load_user_state(user)

    # ---------- Sidebar ----------
    with st.sidebar:
        st.title("Menu")
        st.caption(f"Signed in as **{user}**")
        if st.button("Log out"):
            auth.logout()

        pdf_docs = st.file_uploader(
            "Upload your PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if st.button("Submit & Process", type="primary"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Processing PDF files..."):
                    try:
                        process_pdfs(pdf_docs)
                    except Exception as e:
                        st.error(f"PDF processing failed: {e}")

        if st.session_state["filenames"]:
            st.markdown("**Loaded documents**")
            for name in st.session_state["filenames"]:
                st.markdown(f"- {name}")

        if st.session_state["messages"] and st.button("Clear chat"):
            st.session_state["messages"] = []
            save_messages(user)
            st.rerun()

    # ---------- Chat history ----------
    if st.session_state["vector_store"] is None:
        st.info("Upload your PDFs in the sidebar and click **Submit & Process** to begin.")

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.markdown(f"- {source}")

    # ---------- New question ----------
    question = st.chat_input(
        "Ask a question about your PDF files",
        disabled=st.session_state["vector_store"] is None,
    )

    if question:
        history = list(st.session_state["messages"])  # history before this turn

        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Finding the answer..."):
                try:
                    answer, sources = answer_question(question, history)
                except Exception as e:
                    if "429" in str(e) or "ResourceExhausted" in type(e).__name__:
                        answer = (
                            "Gemini API quota exceeded for this model. Wait for the "
                            "quota to reset, set a different GEMINI_MODEL in your "
                            ".env, or enable billing."
                        )
                    else:
                        answer = f"An error occurred: {e}"
                    sources = []

            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.markdown(f"- {source}")

        st.session_state["messages"].append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
        save_messages(user)


if __name__ == "__main__":
    main()