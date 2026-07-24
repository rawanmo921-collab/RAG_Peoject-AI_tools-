import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class SVDEmbeddingModel:
   

    def __init__(self, documents, n_components=30):
        self.vectorizer = TfidfVectorizer()
        doc_tfidf = self.vectorizer.fit_transform(documents)
        n_components = min(n_components, min(doc_tfidf.shape) - 1)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.svd.fit(doc_tfidf)

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        tfidf_vecs = self.vectorizer.transform(texts)
        dense_vecs = self.svd.transform(tfidf_vecs)
        if normalize_embeddings:
            norms = np.linalg.norm(dense_vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            dense_vecs = dense_vecs / norms
        return dense_vecs


def get_embedding_model(documents):
    
    if HAS_SENTENCE_TRANSFORMERS:
        try:
            print("[embedding_utils] Loading SentenceTransformer (all-MiniLM-L6-v2)...")
            return SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print(f"[embedding_utils] Failed to load the model ({type(e).__name__}). "
                  f"Switching to SVD offline.")
    return SVDEmbeddingModel(documents)
