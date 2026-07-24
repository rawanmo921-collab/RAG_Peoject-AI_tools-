import pickle

import chromadb

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "support_replies"
MODEL_PATH = "data/embedding_model.pkl"

# عتبة الثقة: تشابه cosine أقل من كده يعتبر "عدم تطابق قوي"
# (chroma بيرجع cosine *distance* = 1 - similarity، فكل ما القيمة أقل كل ما التشابه أعلى)
CONFIDENCE_DISTANCE_THRESHOLD = 0.55


def load_embedding_model(path: str = MODEL_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_collection(chroma_dir: str = CHROMA_DIR, collection_name: str = COLLECTION_NAME):
    client = chromadb.PersistentClient(path=chroma_dir)
    return client.get_collection(collection_name)


def retrieve_context(
    query: str,
    collection,
    model,
    k: int = 3,
    brand: str | None = None,
    confidence_threshold: float = CONFIDENCE_DISTANCE_THRESHOLD,
):
    
    query_embedding = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )[0].tolist()

    where_filter = {"brand": brand} if brand else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where_filter,
    )

    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []
    dists = results["distances"][0] if results["distances"] else []

    chunks = [
        {
            "text": doc,
            "doc_id": meta.get("doc_id"),
            "brand": meta.get("brand"),
            "distance": dist,
        }
        for doc, meta, dist in zip(docs, metas, dists)
    ]

    best_distance = dists[0] if dists else 1.0
    is_confident = bool(chunks) and best_distance <= confidence_threshold

    return {
        "chunks": chunks,
        "is_confident": is_confident,
        "best_distance": best_distance,
    }


def main():
    print("[06] loading Chroma collection...")
    model = load_embedding_model()
    collection = load_collection()

    demo_queries = [
        "my order hasn't arrived yet, what should I do?",
        "app keeps crashing when I open it",
    ]
    for q in demo_queries:
        print("-" * 70)
        print("QUERY:", q)
        result = retrieve_context(q, collection, model, k=3)
        print(f"is_confident={result['is_confident']}  best_distance={result['best_distance']:.3f}")
        for c in result["chunks"]:
            print(f"  [{c['brand']}] (dist={c['distance']:.3f}) {c['text'][:100]}")


if __name__ == "__main__":
    main()
