import os
import pickle

import numpy as np
import pandas as pd

from embedding_utils import get_embedding_model  # موديول مشترك، عشان الـ pickle يتفك من أي ملف

INPUT_PATH = "data/chunks.csv"
EMBEDDINGS_PATH = "data/embeddings.npy"
MODEL_PATH = "data/embedding_model.pkl"


def main():
    print(f"[04] loading: {INPUT_PATH}")
    chunks_df = pd.read_csv(INPUT_PATH)
    texts = chunks_df["text"].astype(str).tolist()

    model = get_embedding_model(texts)
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    print(f"[04] shape of the embeddings matrix: {embeddings.shape}")

    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"[04] files saved to: {EMBEDDINGS_PATH}")
    print(f"[04] files saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
