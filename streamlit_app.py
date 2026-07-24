import pickle
from importlib import import_module

import pandas as pd
import streamlit as st
import chromadb

# ملفات الـ pipeline أسماؤها بتبدأ برقم، فبنستوردها بـ importlib
retrieve_module = import_module("06_retrieve_context")
prompting_module = import_module("07_prompting")

retrieve_context = retrieve_module.retrieve_context
generate_response = prompting_module.generate_response
evaluate_generated_answer = prompting_module.evaluate_generated_answer

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "support_replies"
MODEL_PATH = "data/embedding_model.pkl"
CHUNKS_PATH = "data/chunks.csv"


@st.cache_resource
def load_resources():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    chunks_df = pd.read_csv(CHUNKS_PATH)
    brands = sorted(chunks_df["brand"].dropna().unique().tolist())
    return model, collection, brands


def init_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []  # [{role, content}, ...] لكل الجلسة الحالية
    if "customer_id" not in st.session_state:
        st.session_state.customer_id = "guest"


def main():
    st.set_page_config(page_title="Customer Support Assistant (RAG)", page_icon="💬")
    st.title("💬 Customer Support Assistant - RAG Assistant")
    st.caption(
        "This assistant searches through a database of support replies, integrates the most similar responses, "
        "and presents you with an appropriate answer to your question. The chat remembers your conversation "
        "throughout the session."
    )

    model, collection, brands = load_resources()
    init_session_state()

    with st.sidebar:
        st.subheader("Settings")
        brand_choice = st.selectbox(
            "Customize response by brand (optional)",
            options=["All Brands"] + brands,
        )
        brand_filter = None if brand_choice == "All Brands" else brand_choice

        k = st.slider("Number of similar responses used (k)", min_value=1, max_value=6, value=3)

        confidence_threshold = st.slider(
            "Confidence threshold (distance) for fallback to human support",
            min_value=0.05, max_value=1.5, value=0.55, step=0.05,
            help=(
                "Any distance higher than this value is considered 'poor match' "
                "and will trigger a fallback to human support. If the chat consistently "
                "responds with 'I couldn't find an answer' even for clear questions, "
                "try adjusting this value incrementally and observe the actual distances "
                "under each response."
            ),
        )

        if st.button("🗑️ Start New Conversation (Clear Memory)"):
            st.session_state.history = []
            st.rerun()

    # عرض المحادثة السابقة كاملة (الذاكرة)
    for turn in st.session_state.history:
        role = "user" if turn["role"] == "customer" else "assistant"
        with st.chat_message(role):
            st.write(turn["content"])

    query = st.chat_input("Enter your question or concern here...")
    if not query:
        return

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching the reply archive..."):
            retrieval_result = retrieve_context(
                query, collection, model, k=k, brand=brand_filter,
                confidence_threshold=confidence_threshold,
            )
            result = generate_response(
                query, retrieval_result, history=st.session_state.history
            )

        st.write(result["answer"])

        # رقم المسافة الفعلي بيتعرض دايمًا، عشان تعرف تظبط العتبة صح
        st.caption(
            f"🔎 Closest Distance (best_distance) = {retrieval_result['best_distance']:.3f}  "
            f"| Current Threshold = {confidence_threshold:.2f}  "
            f"| {'✅ Above Threshold (Confident)' if retrieval_result['is_confident'] else '❌ Below Threshold (Not Confident)'}"
        )

        if result["used_fallback"] and not retrieval_result["is_confident"]:
            st.warning("⚠️ Poor Match with Archive - Fallback to Human Support Initiated.")
        elif result["used_fallback"]:
            st.info("ℹ️ Direct Integration Used from Archive (No LLM API Key Required).")

        with st.expander("📚 The sources on which I base the response"):
            if retrieval_result["chunks"]:
                for c in retrieval_result["chunks"]:
                    st.write(f"**[{c['brand']}]** (distance={c['distance']:.3f}) — {c['text']}")
            else:
                st.write("There are no matching sources.")

    # تحديث الذاكرة (نفس الشيء اللي بيعمله ChatSession.ask， لكن هنا Streamlit
    # بيدير الـ session_state بنفسه فبنضيف الأدوار يدويًا)
    st.session_state.history.append({"role": "customer", "content": query})
    st.session_state.history.append({"role": "assistant", "content": result["answer"]})


if __name__ == "__main__":
    main()
