# scripts/truncate_prompts.py
#
# Purpose:
# Some models can’t handle giant prompts (e.g., >16k tokens).
# This script takes a CSV of prompts and trims them
# so they fit the max input size for a given model/tokenizer.

import argparse, pandas as pd
from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser("Truncate augmented prompts to a token limit")
    p.add_argument("--in_csv", type=str, required=True, help="Input CSV (from build_retrieval_dataset.py)")
    p.add_argument("--out_csv", type=str, required=True, help="Output CSV with truncated prompts")
    p.add_argument("--tokenizer_name", type=str, required=True, help="Model tokenizer name")
    p.add_argument("--max_input_tokens", type=int, default=16384)
    return p.parse_args()


def main():
    a = parse_args()
    tok = AutoTokenizer.from_pretrained(a.tokenizer_name, trust_remote_code=True)

    df = pd.read_csv(a.in_csv)

    def truncate(prompt: str) -> str:
        ids = tok.encode(str(prompt), add_special_tokens=False)
        if len(ids) > a.max_input_tokens:
            # Chop tokens and decode back to text
            ids = ids[:a.max_input_tokens]
            return tok.decode(ids, skip_special_tokens=True)
        return prompt

    df["augmented_prompt"] = df["augmented_prompt"].astype(str).apply(truncate)
    df.to_csv(a.out_csv, index=False)
    print(f" Wrote truncated CSV → {a.out_csv}")


if __name__ == "__main__":
    main()
