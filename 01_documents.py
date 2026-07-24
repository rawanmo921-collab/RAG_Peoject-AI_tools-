import os
import pandas as pd

RAW_CSV_PATH = "sample.csv"
OUTPUT_DIR = "data"


def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def split_documents_and_queries(df: pd.DataFrame):
    
    support_docs = df[df["inbound"] == False].reset_index(drop=True)
    customer_queries = df[df["inbound"] == True].reset_index(drop=True)
    return support_docs, customer_queries


def save_outputs(support_docs: pd.DataFrame, customer_queries: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    support_path = os.path.join(out_dir, "documents_raw.csv")
    queries_path = os.path.join(out_dir, "customer_queries_raw.csv")
    support_docs.to_csv(support_path, index=False)
    customer_queries.to_csv(queries_path, index=False)
    return support_path, queries_path


def main():
    print(f"[01] downloading {RAW_CSV_PATH}")
    df = load_raw_data(RAW_CSV_PATH)
    print(f"[01] total number of rows: {len(df)}")

    support_docs, customer_queries = split_documents_and_queries(df)
    print(f"[01] number of support documents (documents/knowledge base): {len(support_docs)}")
    print(f"[01] number of customer queries: {len(customer_queries)}")

    support_path, queries_path = save_outputs(support_docs, customer_queries, OUTPUT_DIR)
    print(f"[01] files saved to:\n     - {support_path}\n     - {queries_path}")


if __name__ == "__main__":
    main()
