# app.py
import streamlit as st
import os
import time
import json
from pathlib import Path
from dotenv import load_dotenv
from embed_index import EMBED_MODEL
from retrieve import Retriever, RetrieverConfig

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)
from ingest import (
    extract_pages,
    normalize_text,
    detect_headers_footers,
    remove_headers_footers,
    chunk_text_by_words,
)
from embed_index import (
    load_model,
    batch_encode_texts,
    create_faiss_index,
    save_embeddings,
)
from retrieve import Retriever
from mixtral_client import assemble_prompt, call_mixtral

# ------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------
st.set_page_config(page_title="StudyMate — PDF Q&A", layout="wide")
st.title("📘 StudyMate — AI-Powered PDF Q&A")

# ------------------------------------------------------
# INITIALIZE SESSION STATE
# ------------------------------------------------------
if "retriever" not in st.session_state:
    st.session_state["retriever"] = None
if "last_call" not in st.session_state:
    st.session_state["last_call"] = 0.0

# ------------------------------------------------------
# SIDEBAR: PDF Upload + Index Management
# ------------------------------------------------------
with st.sidebar:
    st.header("📂 Document Upload & Indexing")

    uploaded = st.file_uploader(
        "Upload one or more PDF files", type=["pdf"], accept_multiple_files=True
    )
    ingest_btn = st.button("🚀 Ingest PDFs and Build Index")
    load_btn = st.button("📥 Load Existing Index")

    st.markdown("---")
    st.markdown("⚠️ **Guidelines:**")
    st.markdown("- Total size ≤ 100 MB")
    st.markdown("- Avoid uploading sensitive or personal data")

# ------------------------------------------------------
# INGESTION PIPELINE
# ------------------------------------------------------
if ingest_btn and uploaded:
    os.makedirs("data", exist_ok=True)
    all_chunks = []

    for f in uploaded:
        fname = f.name
        save_path = os.path.join("data", fname)
        with open(save_path, "wb") as out:
            out.write(f.getbuffer())
        st.info(f"📄 Saved `{fname}` to /data")

        pages = extract_pages(save_path)
        headers, footers = detect_headers_footers(pages)

        for p in pages:
            p["text"] = normalize_text(remove_headers_footers(p["text"], headers, footers))

        doc_chunks = chunk_text_by_words(doc_id=fname, pages=pages)
        st.write(f"✅ Ingested {fname}: {len(doc_chunks)} chunks")
        all_chunks.extend(doc_chunks)

    if all_chunks:
        with st.spinner("🔎 Embedding and indexing text (this may take a moment)..."):
            model = load_model()
            texts = [c["text"] for c in all_chunks]
            embeddings = batch_encode_texts(model, texts)
            save_embeddings(embeddings, all_chunks, prefix="corpus")
            create_faiss_index(
                embeddings, index_path="indexes/corpus.index", use_hnsw=False
            )

        st.success("✅ Index built successfully and saved to `/indexes/`.")
        config = RetrieverConfig(
            index_path="indexes/corpus.index",
            metadata_path="indexes/corpus_metadata.json",
            model_name=EMBED_MODEL
        )
        st.session_state["retriever"] = Retriever(config)# ------------------------------------------------------
# LOAD EXISTING INDEX
# ------------------------------------------------------
if load_btn:
    try:
        config = RetrieverConfig(
            index_path="indexes/corpus.index",
            metadata_path="indexes/corpus_metadata.json",
            model_name=EMBED_MODEL
        )
        retriever = Retriever(config)
        st.session_state["retriever"] = retriever
        st.success("✅ Existing FAISS index loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load index: {e}")

# ------------------------------------------------------
# QUERY SECTION
# ------------------------------------------------------
st.header("💬 Ask a Question")

query = st.text_input("Type your academic question here:")
ask_btn = st.button("🔍 Get Answer")
TOP_K = 4

if ask_btn and query:
    now = time.time()
    if now - st.session_state["last_call"] < 0.5:
        st.warning("⏳ Please wait a moment before your next query.")
    else:
        st.session_state["last_call"] = now

        retriever = st.session_state.get("retriever")
        if retriever is None:
            try:
                config = RetrieverConfig(
                    index_path="indexes/corpus.index",
                    metadata_path="indexes/corpus_metadata.json",
                    model_name=EMBED_MODEL
                )
                retriever = Retriever(config)
                st.session_state["retriever"] = retriever
            except Exception as e:
                st.error("❌ No index found. Please upload and ingest PDFs first.")
                st.stop()

        # Retrieve top-k chunks (fixed value)
        with st.spinner("🔎 Retrieving relevant sections..."):
            top_chunks = retriever.retrieve(query, top_k=TOP_K)

        if not top_chunks:
            st.warning("⚠️ No relevant chunks found in your uploaded documents.")
            st.stop()

        # Display retrieved chunks
        st.subheader("📚 Retrieved Context Chunks")
        for idx, c in enumerate(top_chunks):
            st.markdown(f"**{idx+1}. [{c.doc_id} | page {c.page_num}]** — *(score={c.score:.3f})*")
            st.write(c.text[:500] + ("..." if len(c.text) > 500 else ""))

        # Assemble the prompt for LLM
        prompt = assemble_prompt(top_chunks, query)
        with st.expander("🧠 Show Assembled Prompt (debug view)"):
            st.code(prompt[:2000])

        # Check Mixtral API credentials
        if not os.environ.get("MISTRAL_API_KEY"):
            st.error("⚠️ Mixtral API key not set. Please configure `MISTRAL_API_KEY` environment variable.")
            st.stop()

        # Call Mixtral
        with st.spinner("🤖 Generating answer using Mixtral..."):
            try:
                answer = call_mixtral(prompt, max_tokens=400, temperature=0.1)
                st.markdown("### 🧩 Answer")
                st.write(answer)

                st.markdown("### 🔖 Supporting Context (Sources)")
                for c in top_chunks:
                    st.write(f"[{c.doc_id} | page {c.page_num}] — {c.text[:400]}...")
            except Exception as ex:
                st.error(f"❌ LLM call failed: {ex}")

# ------------------------------------------------------
# SIDEBAR: Feedback & API Test
# ------------------------------------------------------
st.sidebar.header("💡 Feedback")
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("👍 Helpful"):
        st.sidebar.success("Thanks for your feedback!")
with col2:
    if st.button("👎 Not Helpful"):
        st.sidebar.info("We'll use this to improve StudyMate.")

st.sidebar.markdown("---")
with st.sidebar.expander("🔧 Test Mixtral API Connection"):
    if st.button("Test API"):
        try:
            result = call_mixtral("This is a test prompt to verify API connectivity.")
            st.success("✅ Mixtral API reachable!")
            st.write(result[:300])
        except Exception as e:
            st.error(f"❌ API test failed: {e}")
