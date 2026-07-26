"""
streamlit_app.py
==================
Final user interface: a RAG-based customer support assistant, built on top
of the whole pipeline (steps 01 -> 07).

Note on deployment (Deploy / Streamlit Cloud):
    This file checks on startup whether the pipeline artifacts
    (data/embedding_model.pkl, chroma_db/...) already exist. If they don't
    (e.g. the very first run on a fresh cloud server that has no copy of
    them yet), it automatically runs steps 01 -> 05 before opening the
    chat. That means you don't have to run those scripts manually before
    deploying - the very first load will just take a bit longer.

    If you'd rather run them manually yourself locally (faster for
    development):
        python 01_documents.py
        python 02_preprocessing.py
        python 03_chunking.py
        python 04_vector_representation.py
        python 05_create_chroma_store.py

Then run:
    streamlit run streamlit_app.py

Features:
  - Full chat memory: every customer message and reply stays saved for the
    whole session (session_state), and is passed as context to each new
    reply.
  - Optional brand selection to customize the answer (filters the archive
    to that brand only).
  - Transparent output: under every reply, the source chunks it was built
    from are shown, along with a clear message when the reply was
    escalated to a human agent due to low retrieval confidence.
"""

import os
import pickle
from importlib import import_module

import pandas as pd
import streamlit as st
import chromadb

# The pipeline files start with a digit in their name, so we import them
# with importlib instead of a normal "import" statement.
retrieve_module = import_module("06_retrieve_context")
prompting_module = import_module("07_prompting")

retrieve_context = retrieve_module.retrieve_context
generate_response = prompting_module.generate_response
evaluate_generated_answer = prompting_module.evaluate_generated_answer

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "support_replies"
MODEL_PATH = "data/embedding_model.pkl"
CHUNKS_PATH = "data/chunks.csv"


def _pipeline_artifacts_exist() -> bool:
    chroma_ready = os.path.isdir(CHROMA_DIR) and len(os.listdir(CHROMA_DIR)) > 0
    return os.path.exists(MODEL_PATH) and os.path.exists(CHUNKS_PATH) and chroma_ready


def ensure_pipeline_ready():
    """If the pipeline artifacts (data/, chroma_db/) don't exist yet -
    e.g. the first run on a brand-new Streamlit Cloud server - run steps
    01 through 05 automatically here before trying to open them, instead
    of raising a FileNotFoundError."""
    if _pipeline_artifacts_exist():
        return
    with st.spinner("First run: building the knowledge base (this may take a bit)..."):
        for step in [
            "01_documents", "02_preprocessing", "03_chunking",
            "04_vector_representation", "05_create_chroma_store",
        ]:
            import_module(step).main()


@st.cache_resource
def load_resources():
    ensure_pipeline_ready()
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    chunks_df = pd.read_csv(CHUNKS_PATH)
    brands = sorted(chunks_df["brand"].dropna().unique().tolist())
    return model, collection, brands


def init_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []  # [{role, content}, ...] for the current session
    if "customer_id" not in st.session_state:
        st.session_state.customer_id = "guest"


def main():
    st.set_page_config(page_title="Customer Support Assistant (RAG)", page_icon="💬")
    st.title("💬 Customer Support Assistant - RAG")
    st.caption(
        "Searches the archive of past support replies, merges the closest "
        "matching ones, and gives you one suitable answer to your question. "
        "The chat remembers your conversation for the whole session."
    )

    model, collection, brands = load_resources()
    init_session_state()

    with st.sidebar:
        st.subheader("Settings")
        brand_choice = st.selectbox(
            "Customize reply by brand (optional)",
            options=["All brands"] + brands,
        )
        brand_filter = None if brand_choice == "All brands" else brand_choice

        k = st.slider("Number of similar replies used (k)", min_value=1, max_value=6, value=3)

        confidence_threshold = st.slider(
            "Confidence threshold",
            min_value=0.05, max_value=1.5, value=0.55, step=0.05,
            help=(
                "Any distance above this value is considered a 'weak match' "
                "and gets escalated to a human agent. If the chat always says "
                "'no confident answer' even for clear questions, try raising "
                "this value gradually and watch the actual distance shown "
                "under each reply."
            ),
        )

        if st.button("🗑️ Start a new conversation (clear memory)"):
            st.session_state.history = []
            st.rerun()

    # Show the full previous conversation (memory)
    for turn in st.session_state.history:
        role = "user" if turn["role"] == "customer" else "assistant"
        with st.chat_message(role):
            st.write(turn["content"])

    query = st.chat_input("Type the customer's complaint or question here...")
    if not query:
        return

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching the support archive..."):
            retrieval_result = retrieve_context(
                query, collection, model, k=k, brand=brand_filter,
                confidence_threshold=confidence_threshold,
            )
            result = generate_response(
                query, retrieval_result, history=st.session_state.history
            )

        st.write(result["answer"])

        # The actual distance is always shown, so you can tune the threshold correctly
        st.caption(
            f"🔎 Best distance = {retrieval_result['best_distance']:.3f}  "
            f"| Current threshold = {confidence_threshold:.2f}  "
            f"| {'✅ Under threshold (confident)' if retrieval_result['is_confident'] else '❌ Over threshold (not confident)'}"
        )

        if result["used_fallback"] and not retrieval_result["is_confident"]:
            st.warning("⚠️ No strong match found in the archive - escalated to a human agent.")
        elif result["used_fallback"]:
            st.info("ℹ️ Answer merged directly from the archive (no LLM API key set).")
            if result.get("generation_error"):
                with st.expander("🐞 Why didn't the LLM answer? (debug info)"):
                    st.code(result["generation_error"])

        with st.expander("📚 Sources used to build this reply"):
            if retrieval_result["chunks"]:
                for c in retrieval_result["chunks"]:
                    st.write(f"**[{c['brand']}]** (distance={c['distance']:.3f}) — {c['text']}")
            else:
                st.write("No matching sources found.")

    # Update memory (same idea as ChatSession.ask, but here Streamlit manages
    # session_state itself, so we append the turns manually)
    st.session_state.history.append({"role": "customer", "content": query})
    st.session_state.history.append({"role": "assistant", "content": result["answer"]})


if __name__ == "__main__":
    main()
