import os
import re
import pandas as pd

INPUT_PATH = "data/documents_raw.csv"
OUTPUT_PATH = "data/documents_clean.csv"


def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", "", text)   
    text = re.sub(r"@\w+", "", text)                
    text = re.sub(r"\s+", " ", text).strip()        
    return text


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text_clean"] = df["text"].apply(clean_text)

    before = len(df)
    df = df[df["text_clean"].str.len() > 3]          # شيل النصوص الفاضية/التافهة بعد التنظيف
    df = df.drop_duplicates(subset="text_clean")      # شيل التكرارات
    df = df.reset_index(drop=True)
    after = len(df)

    print(f"[02] number of rows before cleaning: {before}")
    print(f"[02] number of rows after removing empty and duplicate: {after}")
    return df


def main():
    print(f"[02] loading: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)

    clean_df = preprocess(df)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    clean_df.to_csv(OUTPUT_PATH, index=False)
    print(f"[02] files saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
