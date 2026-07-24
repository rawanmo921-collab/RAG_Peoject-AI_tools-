import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sklearn.decomposition import TruncatedSVD

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False

pd.set_option("display.max_colwidth", 120)

CSV_PATH = "sample.csv"


# ===========================================================================
# القسم ده زي الملف الأصلي بالظبط (تحميل الداتا + كل طرق البحث الأربعة)
# ===========================================================================

def load_data(path):
    df = pd.read_csv(path)
    return df


def clean_tweet(text):
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_corpus_and_queries(df):
    df = df.copy()
    df["text_clean"] = df["text"].apply(clean_tweet)

    support_df = df[df["inbound"] == False].reset_index(drop=True)
    customer_df = df[df["inbound"] == True].reset_index(drop=True)

    tweet_id_to_doc_id = {row["tweet_id"]: i for i, row in support_df.iterrows()}

    documents = support_df["text_clean"].tolist()
    doc_brands = support_df["author_id"].tolist()

    queries = []
    ground_truth = {}
    for _, row in customer_df.iterrows():
        resp_field = row["response_tweet_id"]
        if pd.isna(resp_field):
            continue
        candidate_ids = [int(x) for x in str(resp_field).split(",") if x.strip().isdigit()]
        relevant_doc_ids = [tweet_id_to_doc_id[i] for i in candidate_ids if i in tweet_id_to_doc_id]
        if not relevant_doc_ids:
            continue
        query_text = row["text_clean"]
        queries.append(query_text)
        ground_truth[query_text] = relevant_doc_ids

    return documents, doc_brands, queries, ground_truth


def simple_tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def build_bm25(documents):
    tokenized = [simple_tokenize(doc) for doc in documents]
    return BM25Okapi(tokenized)


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
    if _HAS_SENTENCE_TRANSFORMERS:
        try:
            print("Loading sentence embedding model (all-MiniLM-L6-v2)...")
            return SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print(f"Could not download all-MiniLM-L6-v2 ({type(e).__name__}). Falling back to SVD.")
    return SVDEmbeddingModel(documents)


def build_embeddings(documents, model):
    return model.encode(documents, convert_to_numpy=True, normalize_embeddings=True)


def min_max_normalize(scores):
    scores = np.array(scores, dtype=float)
    if scores.max() == scores.min():
        return np.zeros_like(scores)
    return (scores - scores.min()) / (scores.max() - scores.min())


def retrieve_top_k_hybrid_scored(query, bm25, model, doc_embeddings, alpha=0.6, k=3, allowed_indices=None):
   
    bm25_scores = bm25.get_scores(simple_tokenize(query))
    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    semantic_scores = cosine_similarity(query_embedding, doc_embeddings).flatten()

    bm25_norm = min_max_normalize(bm25_scores)
    semantic_norm = min_max_normalize(semantic_scores)
    hybrid_scores = alpha * semantic_norm + (1 - alpha) * bm25_norm

    if allowed_indices is not None:
        mask = np.full(hybrid_scores.shape, -np.inf)
        mask[list(allowed_indices)] = hybrid_scores[list(allowed_indices)]
        hybrid_scores = mask

    ranking = np.argsort(hybrid_scores)[::-1][:k]
    scores = hybrid_scores[ranking]
    return list(ranking), list(scores)


# ===========================================================================
# RAG LAYER: الميزات الخمسة المطلوبة
# ===========================================================================

CONFIDENCE_THRESHOLD = 0.35   # حد أدنى لدرجة الـ hybrid score عشان نعتبرها "تطابق قوي"
MAX_HISTORY_TURNS = 6         # أقصى عدد أدوار سابقة نضيفهم للـ prompt


# --- 3) تخصيص الرد حسب البراند --------------------------------------------

def brand_filtered_indices(doc_brands, brand):
    """بيرجع أندكسات الردود اللي بتاعة براند معين بس. لو brand=None بيرجع None
    (يعني من غير فلترة - كل الردود مسموحة)."""
    if brand is None:
        return None
    return [i for i, b in enumerate(doc_brands) if b == brand]


# --- 2) التعامل مع الحالات اللي مفيش فيها تطابق قوي ------------------------

def is_low_confidence(best_score, threshold=CONFIDENCE_THRESHOLD):
    return best_score < threshold


# --- 1) دمج ردود متعددة في رد واحد ذكي --------------------------------------

def merge_multiple_replies(query, documents, retrieved_indices):
    
    texts = []
    seen = set()
    for idx in retrieved_indices:
        t = documents[idx].strip()
        if t and t not in seen:
            texts.append(t)
            seen.add(t)
    if not texts:
        return "I don't have enough information to answer this question right now."

    primary = texts[0]
    extra = _distinct_extra_snippets(primary, texts[1:])
    if not extra:
        return primary

    extra_block = "\n".join(f"• {snippet}" for snippet in extra)
    return f"{primary}\n\nYou might also find this helpful:\n{extra_block}"


def _distinct_extra_snippets(primary, candidates, max_extra=2):
  
    def tokens(t):
        return set(t.lower().split())

    primary_tokens = tokens(primary)
    kept, kept_tokens = [], []
    for snippet in candidates:
        snippet_tokens = tokens(snippet)
        if not snippet_tokens:
            continue
        if len(snippet_tokens & primary_tokens) / len(snippet_tokens) > 0.5:
            continue
        if any(len(snippet_tokens & kt) / len(snippet_tokens) > 0.5 for kt in kept_tokens):
            continue
        kept.append(snippet)
        kept_tokens.append(snippet_tokens)
        if len(kept) >= max_extra:
            break
    return kept


def build_prompt(query, documents, retrieved_indices, history):
    context_lines = [f"[{i+1}] {documents[idx]}" for i, idx in enumerate(retrieved_indices)]
    context_block = "\n".join(context_lines)

    history_block = ""
    if history:
        recent = history[-MAX_HISTORY_TURNS:]
        history_block = "محادثة سابقة مع نفس العميل:\n" + "\n".join(
            f"{h['role']}: {h['content']}" for h in recent
        ) + "\n\n"

    return (
        "You are a customer support associate. Combine the information from previous similar responses under "
        "In one short and appropriate response to the customer's question, without inventing information that does not exist.\n\n"
        f"{history_block}"
        f"Customer Question: {query}\n\n"
        f"Previous Similar Responses:\n{context_block}\n\n"
        "Final Answer only:"
    )


def call_llm_generate(prompt):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY غير موجود")
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in response.content if getattr(b, "type", "") == "text").strip()


def generate_response(query, documents, doc_brands, bm25, model, doc_embeddings,
                       history=None, brand=None, k=3, alpha=0.6):
    
    history = history or []
    allowed_indices = brand_filtered_indices(doc_brands, brand)

    if allowed_indices is not None and len(allowed_indices) == 0:
        return {
            "answer": f"I do not currently have an archive of previous responses for the brand '{brand}'. I will refer you to human support.",
            "used_fallback": True,
            "is_confident": False,
            "retrieved_indices": [],
            "best_score": 0.0,
        }

    retrieved_indices, scores = retrieve_top_k_hybrid_scored(
        query, bm25, model, doc_embeddings, alpha=alpha, k=k, allowed_indices=allowed_indices
    )
    best_score = scores[0] if scores else 0.0

    if is_low_confidence(best_score):
        return {
            "answer": (
                "I'm sorry, but I couldn't find a sufficiently similar response in the current archive for your question. "
                "I will connect you with a human customer service representative who can assist you further."
            ),
            "used_fallback": True,
            "is_confident": False,
            "retrieved_indices": retrieved_indices,
            "best_score": best_score,
        }

    prompt = build_prompt(query, documents, retrieved_indices, history)
    try:
        answer = call_llm_generate(prompt)
        used_fallback = False
    except Exception:
        answer = merge_multiple_replies(query, documents, retrieved_indices)
        used_fallback = True

    return {
        "answer": answer,
        "used_fallback": used_fallback,
        "is_confident": True,
        "retrieved_indices": retrieved_indices,
        "best_score": best_score,
    }


# --- 4) تقييم جودة الرد المُولَّد (مش بس دقة الاسترجاع) ---------------------

def evaluate_generated_answer(generated_answer, reference_answers, model):
    
    if not reference_answers:
        return {"semantic_similarity": None, "best_reference": None}

    gen_emb = model.encode([generated_answer], convert_to_numpy=True, normalize_embeddings=True)
    ref_embs = model.encode(reference_answers, convert_to_numpy=True, normalize_embeddings=True)
    sims = cosine_similarity(gen_emb, ref_embs).flatten()
    return {
        "semantic_similarity": float(sims.max()),
        "best_reference": reference_answers[int(sims.argmax())],
    }


# --- 5) ذاكرة المحادثة (Chat Memory) ----------------------------------------

class ChatSession:
   
    def __init__(self, customer_id="guest", brand=None):
        self.customer_id = customer_id
        self.brand = brand
        self.history = []

    def add_turn(self, role, content):
        self.history.append({"role": role, "content": content})

    def ask(self, query, documents, doc_brands, bm25, model, doc_embeddings, k=3):
        result = generate_response(
            query, documents, doc_brands, bm25, model, doc_embeddings,
            history=self.history, brand=self.brand, k=k,
        )
        self.add_turn("customer", query)
        self.add_turn("assistant", result["answer"])
        return result


# ===========================================================================
# main(): تجربة حية توضح الميزات الخمسة مع بعض
# ===========================================================================

def main():
    df = load_data(CSV_PATH)
    documents, doc_brands, queries, ground_truth = build_corpus_and_queries(df)

    print(f"Support documents (knowledge base): {len(documents)}")
    print(f"Customer queries with known ground truth: {len(queries)}")
    print()

    bm25 = build_bm25(documents)
    model = get_embedding_model(documents)
    doc_embeddings = build_embeddings(documents, model)
    print()

    # --- تجربة 1: عميل بيسأل سؤالين في نفس الجلسة (يوضح الذاكرة + الدمج) ---
    print("=" * 70)
    print("Experiment: An entire chat session for one customer (with chat memory)")
    session = ChatSession(customer_id="customer_1")  # من غير تخصيص براند
    demo_questions = [queries[0] if queries else "my order hasn't arrived yet"]
    if len(queries) > 1:
        demo_questions.append(queries[1])

    for q in demo_questions:
        result = session.ask(q, documents, doc_brands, bm25, model, doc_embeddings, k=3)
        print("-" * 70)
        print("QUERY:", q)
        print("ANSWER:", result["answer"])
        print(f"confident={result['is_confident']}  best_score={result['best_score']:.3f}  "
              f"used_fallback={result['used_fallback']}")

    # --- تجربة 2: تخصيص حسب البراند ---
    print()
    print("=" * 70)
    print("Experiment: Filtering responses by a specific brand")
    if queries:
        sample_brand = doc_brands[0]
        session_brand = ChatSession(customer_id="customer_2", brand=sample_brand)
        result = session_brand.ask(queries[0], documents, doc_brands, bm25, model, doc_embeddings, k=3)
        print(f"Brand filter: {sample_brand}")
        print("ANSWER:", result["answer"])

    # --- تجربة 3: تقييم جودة الرد المُولَّد مقابل الإجابة الصحيحة الحقيقية ---
    print()
    print("=" * 70)
    print("Experiment: Evaluating the quality of generated responses (Generation Quality)")
    if queries:
        q = queries[0]
        reference_docs = [documents[i] for i in ground_truth[q]]
        result = generate_response(q, documents, doc_brands, bm25, model, doc_embeddings, k=3)
        quality = evaluate_generated_answer(result["answer"], reference_docs, model)
        print("QUERY:", q)
        print("GENERATED ANSWER:", result["answer"])
        print("SEMANTIC SIMILARITY TO GROUND TRUTH:", quality["semantic_similarity"])


if __name__ == "__main__":
    main()
