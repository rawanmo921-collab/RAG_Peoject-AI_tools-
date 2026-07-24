import numpy as np
import pandas as pd
import chromadb

CHUNKS_PATH = "data/chunks.csv"
EMBEDDINGS_PATH = "data/embeddings.npy"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "support_replies"


def load_chunks_and_embeddings():
    chunks_df = pd.read_csv(CHUNKS_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)
    assert len(chunks_df) == embeddings.shape[0], (

    )
    return chunks_df, embeddings


def build_chroma_store(chunks_df: pd.DataFrame, embeddings: np.ndarray):
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # لو الـ collection موجودة من تشغيل سابق، امسحها وابنيها من جديد
    # عشان دايمًا نبني الـ store من أحدث نسخة للداتا
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

    # نستخدم embeddings جاهزة من عندنا، فمش محتاجين embedding_function من Chroma
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = chunks_df["chunk_id"].astype(str).tolist()
    documents = chunks_df["text"].astype(str).tolist()
    metadatas = [
        {"doc_id": int(row["doc_id"]), "brand": str(row["brand"])}
        for _, row in chunks_df.iterrows()
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas,
    )
    return client, collection


def main():
    print(f"[05] loading: {CHUNKS_PATH} و {EMBEDDINGS_PATH}")
    chunks_df, embeddings = load_chunks_and_embeddings()

    print(f"[05] building Chroma persistent store in: {CHROMA_DIR}")
    client, collection = build_chroma_store(chunks_df, embeddings)

    print(f"[05] files saved to: {CHROMA_DIR}")
    print(f"[05] the store is ready for use in 06_retrieve_context.py")


if __name__ == "__main__":
    main()
