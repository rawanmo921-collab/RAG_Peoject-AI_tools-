import os
import pandas as pd

INPUT_PATH = "data/documents_clean.csv"
OUTPUT_PATH = "data/chunks.csv"

MAX_WORDS_PER_CHUNK = 60   # مناسب لتغريدات قصيرة، وقابل للتعديل لمستندات أطول
OVERLAP_WORDS = 10         # تداخل بسيط بين الأجزاء عشان محافظش على السياق


def chunk_text(text: str, max_words: int = MAX_WORDS_PER_CHUNK, overlap: int = OVERLAP_WORDS):
   
    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + max_words
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start = end - overlap   # التداخل
    return chunks


def build_chunks(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    chunk_counter = 0
    for doc_id, row in df.iterrows():
        text_chunks = chunk_text(row["text_clean"])
        for chunk in text_chunks:
            rows.append({
                "chunk_id": f"chunk_{chunk_counter}",
                "doc_id": int(doc_id),
                "brand": row["author_id"],
                "text": chunk,
            })
            chunk_counter += 1
    return pd.DataFrame(rows)


def main():
    print(f"[03] loading: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)

    chunks_df = build_chunks(df)
    print(f"[03] number of original documents: {len(df)}")
    print(f"[03] number of resulting chunks: {len(chunks_df)}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    chunks_df.to_csv(OUTPUT_PATH, index=False)
    print(f"[03] files saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
