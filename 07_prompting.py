import os
from typing import List, Dict, Optional

MAX_HISTORY_TURNS = 6  # أقصى عدد ردود سابقة نبعتها للموديل كسياق (عشان الطول)


# ---------------------------------------------------------------------------
# 1) دمج ردود متعددة + بناء الـ prompt
# ---------------------------------------------------------------------------

def build_context_block(chunks: List[Dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        brand = c.get("brand", "unknown")
        lines.append(f"[{i}] (from support {brand}): {c['text']}")
    return "\n".join(lines)


def build_prompt(query: str, chunks: List[Dict], history: List[Dict]) -> str:
    context_block = build_context_block(chunks)

    history_block = ""
    if history:
        recent = history[-MAX_HISTORY_TURNS:]
        history_lines = [f"{h['role']}: {h['content']}" for h in recent]
        history_block = "Previous conversation with the same customer:\n" + "\n".join(history_lines) + "\n\n"

    prompt = (
        "You are a customer support assistant. Your job is to integrate information from previous responses."
        "The following (attached below) are similar responses to the current customer query, "
        "from the same/previous support archive:\n\n"
        f"{history_block}"
        f"Current customer query: {query}\n\n"
        f"Similar previous responses:\n{context_block}\n\n"
        "Write the final response appropriate for the current customer query (without additional preamble):"
    )
    return prompt


# ---------------------------------------------------------------------------
# 2) توليد الرد: Claude API لو متاح، وإلا قالب استخراجي (fallback)
# ---------------------------------------------------------------------------

def _extractive_fallback(query: str, chunks: List[Dict]) -> str:
   
    if not chunks:
        return "I don't have enough information to answer this question right now."
    parts = []
    seen = set()
    for c in chunks:
        snippet = c["text"].strip()
        if snippet and snippet not in seen:
            parts.append(snippet)
            seen.add(snippet)

    if not parts:
        return "I don't have enough information to answer this question right now."

    primary = parts[0]
    extra = _distinct_extra_snippets(primary, parts[1:])

    if not extra:
        return primary

    extra_block = "\n".join(f"• {snippet}" for snippet in extra)
    return f"{primary}\n\n It might also help you:\n{extra_block}"


def _distinct_extra_snippets(primary: str, candidates: List[str], max_extra: int = 2) -> List[str]:
    
    def tokens(t: str):
        return set(t.lower().split())

    primary_tokens = tokens(primary)
    kept: List[str] = []
    kept_tokens: List[set] = []

    for snippet in candidates:
        snippet_tokens = tokens(snippet)
        if not snippet_tokens:
            continue
        overlap_primary = len(snippet_tokens & primary_tokens) / len(snippet_tokens)
        if overlap_primary > 0.5:
            continue  # قريب جدًا من الرد الأساسي، هيبقى تكرار
        too_similar_to_kept = any(
            len(snippet_tokens & kt) / len(snippet_tokens) > 0.5 for kt in kept_tokens
        )
        if too_similar_to_kept:
            continue
        kept.append(snippet)
        kept_tokens.append(snippet_tokens)
        if len(kept) >= max_extra:
            break
    return kept

    if len(parts) == 1:
        return parts[0]
    return " like that، ".join(parts[:3])


def call_llm_generate(prompt: str) -> str:
    
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY Not")

    import anthropic  # استيراد داخلي: مايكسرش الكود لو المكتبة مش متثبتة أصلًا

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    ).strip()


# ---------------------------------------------------------------------------
# 3) الدالة الرئيسية: تجمع كل حاجة مع بعض
# ---------------------------------------------------------------------------

def generate_response(
    query: str,
    retrieval_result: Dict,
    history: Optional[List[Dict]] = None,
) -> Dict:
    
    history = history or []
    chunks = retrieval_result.get("chunks", [])
    is_confident = retrieval_result.get("is_confident", False)

    # الحالة 2: عدم تطابق قوي -> رسالة واضحة بدل رد مخترع
    if not is_confident:
        return {
            "answer": (
                "I'm sorry, but I'm not sure I have enough information to answer this question accurately "
                "from the current archive. I'll connect you with a customer service representative who can "
                "assist you directly."
            ),
            "used_fallback": True,
            "source_chunks": chunks,
        }

    prompt = build_prompt(query, chunks, history)

    try:
        answer = call_llm_generate(prompt)
        used_fallback = False
    except Exception:
        answer = _extractive_fallback(query, chunks)
        used_fallback = True

    return {
        "answer": answer,
        "used_fallback": used_fallback,
        "source_chunks": chunks,
    }


# ---------------------------------------------------------------------------
# 4) تقييم جودة الرد المُولَّد (مش بس دقة الاسترجاع)
# ---------------------------------------------------------------------------

def evaluate_generated_answer(generated_answer: str, reference_answers: List[str], model) -> Dict:
    
    from sklearn.metrics.pairwise import cosine_similarity

    if not reference_answers:
        return {"semantic_similarity": None}

    gen_emb = model.encode([generated_answer], convert_to_numpy=True, normalize_embeddings=True)
    ref_embs = model.encode(reference_answers, convert_to_numpy=True, normalize_embeddings=True)
    sims = cosine_similarity(gen_emb, ref_embs).flatten()

    return {
        "semantic_similarity": float(sims.max()),
        "best_reference": reference_answers[int(sims.argmax())],
    }


# ---------------------------------------------------------------------------
# ذاكرة المحادثة (Chat Memory) لكل عميل
# ---------------------------------------------------------------------------

class ChatSession:
   
    def __init__(self, customer_id: str = "guest"):
        self.customer_id = customer_id
        self.history: List[Dict] = []

    def add_turn(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def ask(self, query: str, collection, model, retrieve_context_fn, brand: Optional[str] = None, k: int = 3):
        retrieval_result = retrieve_context_fn(query, collection, model, k=k, brand=brand)
        result = generate_response(query, retrieval_result, history=self.history)

        self.add_turn("customer", query)
        self.add_turn("assistant", result["answer"])
        return result


if __name__ == "__main__":
    # ملف 06 اسمه بيبدأ برقم فمينفعش نعمله import عادي هنا، فالتجربة
    # الحية لكل الـ pipeline موجودة في streamlit_app.py بدل ما نكررها هنا.
        print("[07] This file is a template for calling(generate_response, ChatSession, evaluate_generated_answer).")
        print("[07] For the full live experience, play: streamlit run streamlit_app.py")
