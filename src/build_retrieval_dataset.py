# scripts/build_retrieval_dataset.py
#
# Purpose:
# Take the raw dataset (Arrow shards) and build a retrieval benchmark.
# We split into Class A / B (based on function size),
# then for each function we add distractor functions around it.
# The output is a CSV we can feed into evaluation.
#
# Why: Makes it easier to test models on multi-function retrieval.

import argparse, random
from pathlib import Path
import pandas as pd
from datasets import load_from_disk
from transformers import AutoTokenizer
from src.data_path import get_dataset_dir


def parse_args():
    # Command line args so we don’t hardcode anything
    p = argparse.ArgumentParser("Build retrieval dataset with distractors")
    p.add_argument("--data_dir", type=str, default=None,
                   help="Folder with Arrow shards (load_from_disk format)")
    p.add_argument("--size_threshold", type=int, default=10_000,
                   help="Boundary between Class A (< threshold) and Class B (>= threshold)")
    p.add_argument("--class_type", type=str, choices=["A","B"], default="B",
                   help="Which class split to use")
    p.add_argument("--num_tasks", type=int, default=300,
                   help="How many tasks to sample (rows)")
    p.add_argument("--num_distractors", type=int, default=20,
                   help="How many distractor functions to add per prompt")
    p.add_argument("--tokenizer_name", type=str, required=True,
                   help="Model name for tokenizer, e.g. EleutherAI/gpt-neox-20b")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_csv", type=str, default="retrieval_dataset.csv")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    # Figure out where the dataset lives (default: data/raw/function_dataset/)
    ds_dir = get_dataset_dir(args.data_dir)

    # Load Hugging Face Arrow dataset
    dataset = load_from_disk(str(ds_dir))

    # Basic cleanup: drop rows missing code or function name
    dataset = dataset.filter(lambda x: x.get("unoptimized") is not None and x.get("function_name") is not None)

    # Split into class A or B depending on size threshold
    if args.class_type.upper() == "A":
        subset = dataset.filter(lambda x: x.get("unoptimized_size", 0) < args.size_threshold)
    else:
        subset = dataset.filter(lambda x: x.get("unoptimized_size", 0) >= args.size_threshold)

    # Pick N rows for our benchmark
    n = min(args.num_tasks, len(subset))
    subset = subset.shuffle(seed=args.seed).select(range(n))

    # Load tokenizer for token counting
    tok = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)

    pool = subset.to_list()  # Convert dataset to Python list of dicts

    def build_prompt_with_distractors(target_row, pool, k):
        """Take one function and surround it with distractors."""
        target_code = target_row["unoptimized"]
        target_name = target_row["function_name"]

        # Exclude distractors with the same name to avoid cheating
        candidates = [d for d in pool if d["function_name"] != target_name]
        distractors = random.sample(candidates, min(k, len(candidates)))

        # Smash distractors into one big block of code
        distractor_code = "\n\n".join(d["unoptimized"] for d in distractors)

        # Randomly put the target before or after distractors
        if random.choice(["before", "after"]) == "before":
            full_prompt = target_code + "\n\n" + distractor_code
        else:
            full_prompt = distractor_code + "\n\n" + target_code

        return {
            "task_id": f"{target_name}_{random.randint(1000,9999)}",
            "augmented_prompt": full_prompt,
            "target_function_name": target_name,
            "expected_output": target_code,
            "token_count": len(tok.encode(full_prompt, add_special_tokens=False)),
        }

    # Actually build the dataset
    rows = [build_prompt_with_distractors(r, pool, args.num_distractors) for r in pool]
    df = pd.DataFrame(rows)

    # Save to CSV so we can reuse later
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"✅ Saved dataset to {args.out_csv}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
