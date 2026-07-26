"""
07_prompting.py
=================
Step 7 of the pipeline: building the prompt and generating the final
answer (Generation).

The job here is more than just "retrieve an old reply":
  1. Merge several similar replies (chunks) into one coherent answer
     instead of returning just a single one.
  2. If there's no strong match (is_confident=False from step 06), return
     a clear "escalate to a human agent" message instead of "inventing"
     an inaccurate answer (hallucination).
  3. Customize the answer by brand: filter the context to the same brand
     (when known) so the tone and solution match that specific company.
  4. Keep chat memory (chat history) so the answer takes into account
     what the customer said earlier in the same session.

Generating the answer itself:
  - If the ANTHROPIC_API_KEY environment variable is set, it actually
    calls the Claude API (model: claude-haiku-4-5-20251001) to generate a
    natural answer grounded in the retrieved context.
  - If it's not set, it falls back to an "extractive template" that
    merges the most relevant snippet from each retrieved reply, so the
    script keeps working without any API key (same idea as the offline
    fallback in 04_vector_representation.py).
"""

import os
from typing import List, Dict, Optional

MAX_HISTORY_TURNS = 6  # max number of previous turns sent to the model as context (to control length)


# ---------------------------------------------------------------------------
# 1) Merging multiple replies + building the prompt
# ---------------------------------------------------------------------------

def build_context_block(chunks: List[Dict]) -> str:
    """Joins the retrieved chunks into one numbered block of text, to be
    inserted into the prompt."""
    lines = []
    for i, c in enumerate(chunks, start=1):
        brand = c.get("brand", "unknown")
        lines.append(f"[{i}] (from {brand} support): {c['text']}")
    return "\n".join(lines)


def build_prompt(query: str, chunks: List[Dict], history: List[Dict]) -> str:
    context_block = build_context_block(chunks)

    history_block = ""
    if history:
        recent = history[-MAX_HISTORY_TURNS:]
        history_lines = [f"{h['role']}: {h['content']}" for h in recent]
        history_block = "Previous conversation with the same customer:\n" + "\n".join(history_lines) + "\n\n"

    prompt = (
        "You are a customer support assistant. Your task is to merge the "
        "information from the similar past replies below into one short, "
        "clear answer suitable for the customer's current question, "
        "without inventing information that isn't present in these replies.\n\n"
        f"{history_block}"
        f"Customer's current question: {query}\n\n"
        f"Similar past replies from the support archive:\n{context_block}\n\n"
        "Write only the final answer suitable for the customer's question "
        "(no extra preamble):"
    )
    return prompt


# ---------------------------------------------------------------------------
# 2) Generating the answer: Claude API if available, otherwise an extractive fallback
# ---------------------------------------------------------------------------

def _extractive_fallback(query: str, chunks: List[Dict]) -> str:
    """Simple fallback without an LLM. Instead of gluing all retrieved
    replies into one clunky paragraph (which can end up repetitive or
    oddly mixed), it takes the best reply as the primary answer, and only
    adds the extra information that is genuinely different (not close to
    the primary reply's wording) as separate, clearly labeled bullets."""
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
    return f"{primary}\n\nThis might also help:\n{extra_block}"


def _distinct_extra_snippets(primary: str, candidates: List[str], max_extra: int = 2) -> List[str]:
    """Returns only the extra replies that aren't similar to each other or
    to the primary reply (not roughly the same words), to avoid repeating
    the same idea in slightly different wording."""
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
            continue  # too close to the primary reply, would be a repeat
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


def _get_secret(name: str) -> Optional[str]:
    """Looks for a variable in os.environ first, then in st.secrets (if
    running inside a Streamlit app) as a fallback."""
    value = os.environ.get(name)
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(name)
    except Exception:
        return None


def _call_anthropic(prompt: str, api_key: str) -> str:
    import anthropic  # local import: doesn't break the script if the package isn't installed at all

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    ).strip()


def _call_openrouter(prompt: str, api_key: str) -> str:
    """A fully free alternative: OpenRouter offers free models (IDs ending
    in :free) with no balance or credit card required. Its API is
    OpenAI-compatible, so we don't use the anthropic library here, just a
    direct HTTP call."""
    import requests

    model = os.environ.get("OPENROUTER_MODEL", "openrouter/free")
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300,
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def call_llm_generate(prompt: str) -> str:
    """Calls a real LLM if an API key is available (Anthropic takes
    priority, OpenRouter as a free fallback), otherwise raises an
    exception so the calling code falls back to the extractive template
    instead of crashing."""
    anthropic_key = _get_secret("ANTHROPIC_API_KEY")
    if anthropic_key:
        return _call_anthropic(prompt, anthropic_key)

    openrouter_key = _get_secret("OPENROUTER_API_KEY")
    if openrouter_key:
        return _call_openrouter(prompt, openrouter_key)

    raise RuntimeError(
        "No API key available (ANTHROPIC_API_KEY or OPENROUTER_API_KEY)"
    )


# ---------------------------------------------------------------------------
# 3) The main function: ties everything together
# ---------------------------------------------------------------------------

def generate_response(
    query: str,
    retrieval_result: Dict,
    history: Optional[List[Dict]] = None,
) -> Dict:
    """
    Input: retrieval_result from 06_retrieve_context.retrieve_context()
    Output: a dict containing:
      - answer: the final answer (text)
      - used_fallback: whether the extractive fallback was used (low
        confidence, or the LLM call failed) instead of a generated answer
      - source_chunks: the chunks the answer was built from (for
        transparency/traceability)
      - generation_error: the underlying exception message if the LLM
        call failed, or None otherwise (useful for debugging)
    """
    history = history or []
    chunks = retrieval_result.get("chunks", [])
    is_confident = retrieval_result.get("is_confident", False)

    # Case 2: no strong match -> a clear message instead of an invented answer
    if not is_confident:
        return {
            "answer": (
                "Sorry, I'm not confident enough in an answer to this question "
                "based on the current archive. I'll route you to a human support "
                "agent so they can help you directly."
            ),
            "used_fallback": True,
            "source_chunks": chunks,
            "generation_error": None,
        }

    prompt = build_prompt(query, chunks, history)

    generation_error = None
    try:
        answer = call_llm_generate(prompt)
        used_fallback = False
    except Exception as e:
        answer = _extractive_fallback(query, chunks)
        used_fallback = True
        generation_error = f"{type(e).__name__}: {e}"

    return {
        "answer": answer,
        "used_fallback": used_fallback,
        "source_chunks": chunks,
        "generation_error": generation_error,
    }


# ---------------------------------------------------------------------------
# 4) Evaluating the quality of the generated answer (not just retrieval accuracy)
# ---------------------------------------------------------------------------

def evaluate_generated_answer(generated_answer: str, reference_answers: List[str], model) -> Dict:
    """
    Measures how close the generated answer is to the known correct
    reply/replies (ground truth) using cosine similarity between
    embeddings, rather than just evaluating retrieval. This tells us
    whether the final answer is actually close in meaning to a real
    answer that would have satisfied the customer, not just whether we
    fetched the right chunk.
    """
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
# Chat memory, per customer
# ---------------------------------------------------------------------------

class ChatSession:
    """Keeps the full conversation history with a single customer from the
    moment the chat starts until it ends, so every new answer takes their
    earlier messages into account."""

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
    # File 06's name starts with a digit so it can't be imported normally
    # here, so the full live pipeline demo lives in streamlit_app.py
    # instead of being duplicated here.
    print("[07] This file is a module meant to be imported (generate_response, ChatSession, evaluate_generated_answer).")
    print("[07] For a full live demo, run: streamlit run streamlit_app.py")
