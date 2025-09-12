# src/run_retrieval_eval.py
#
# Purpose:
# Evaluate if a model can retrieve the right function from a noisy prompt.
# We check: did the model output the header? Did it output the right logic?
# We also log "lost-in-the-middle" position info.

import argparse
from collections import Counter
from typing import List, Tuple, Optional
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser("LLM Function Retrieval Evaluation")
    p.add_argument("--model_name", type=str, required=True,
                   help="Hugging Face model name or local path")
    p.add_argument("--data_path", type=str, required=True,
                   help="CSV with prompts and expected outputs")
    p.add_argument("--output_path", type=str, default="retrieval_results.csv")
    p.add_argument("--max_tokens", type=int, default=128,
                   help="Max new tokens to generate")
    p.add_argument("--max_input_tokens", type=int, default=16384,
                   help="Max input tokens (truncate prompts)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# --- Small helpers ---

def token_overlap_ratio(a: str, b: str) -> float:
    """How much of function A overlaps with function B (rough % match)."""
    toks_a = Counter(a.split())
    toks_b = Counter(b.split())
    overlap = sum((toks_a & toks_b).values())
    den = max(1, sum(toks_a.values()))
    return overlap / den


def find_subsequence(hay: List[int], needle: List[int]) -> Optional[Tuple[int, int]]:
    """Look for a tiny token subsequence inside a big sequence (for header locating)."""
    if not needle or not hay or len(needle) > len(hay):
        return None
    first = needle[0]
    for i, t in enumerate(hay):
        if t != first:
            continue
        j = 1
        while j < len(needle) and i + j < len(hay) and hay[i + j] == needle[j]:
            j += 1
        if j == len(needle):
            return i, i + j
    return None


# --- Model loading ---
def load_model_and_tokenizer(model_name: str, device: str):
    print(f"\nLoading model: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device)

    # Ensure padding token is set properly
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    return tok, model


# --- Inference ---
def generate_output(prompt: str, tok, model, device: str,
                    max_tokens: int, max_input_tokens: int):
    # Truncate manually for determinism
    full_ids = tok.encode(prompt, add_special_tokens=False)
    if len(full_ids) > max_input_tokens:
        full_ids = full_ids[:max_input_tokens]
        prompt = tok.decode(full_ids, skip_special_tokens=True)

    encoded = tok(prompt, return_tensors="pt", truncation=True,
                  max_length=max_input_tokens).to(device)
    input_len = int(encoded["input_ids"].shape[1])

    with torch.no_grad():
        gen = model.generate(
            **encoded,
            max_new_tokens=max_tokens,
            do_sample=False,  # greedy decode
            eos_token_id=None,
            pad_token_id=tok.pad_token_id,
            early_stopping=False,
            return_dict_in_generate=True,
            output_scores=False,
        )
    decoded = tok.decode(gen.sequences[0], skip_special_tokens=True)
    prompt_ids = encoded["input_ids"][0].tolist()
    return decoded, input_len, prompt_ids


# --- Evaluation loop ---
def evaluate(data_path: str, model_name: str, output_path: str,
             max_tokens: int, max_input_tokens: int, device: str):

    df = pd.read_csv(data_path)
    tok, model = load_model_and_tokenizer(model_name, device)
    max_expected_tokens = 1024  # don’t let GT blow up

    def truncate_output(text: str) -> str:
        ids = tok.encode(str(text), add_special_tokens=False)
        if len(ids) > max_expected_tokens:
            ids = ids[:max_expected_tokens]
            return tok.decode(ids, skip_special_tokens=True)
        return str(text)

    if "expected_output" in df.columns:
        df["expected_output"] = df["expected_output"].astype(str).apply(truncate_output)

    results = []
    print("\nStarting evaluation loop...")
    for i, row in tqdm(df.iterrows(), total=len(df)):
        prompt = str(row["augmented_prompt"])
        true_name = str(row["target_function_name"])

        header_text = f"def {true_name}"
        header_ids = tok.encode(header_text, add_special_tokens=False)

        try:
            output, input_len, prompt_ids = generate_output(
                prompt, tok, model, device, max_tokens, max_input_tokens
            )

            # Locate header position → used for LITM analysis
            loc = find_subsequence(prompt_ids, header_ids)
            if loc is not None:
                sig_start, sig_end = loc
                mid = sig_start + 0.5 * (sig_end - sig_start)
                signal_pos = float(mid / max(1, input_len))
                signal_pos = max(0.0, min(1.0, signal_pos))
            else:
                idx = prompt.find(header_text)
                signal_pos = float((idx + len(header_text) / 2) / max(1, len(prompt))) if idx >= 0 else None
                sig_start = sig_end = None

            # Retrieval checks
            header_retrieved = header_text in output
            exact_logic = False
            partial_logic = False
            if "expected_output" in row:
                gt = str(row["expected_output"]).strip()
                exact_logic = gt in output
                partial_logic = token_overlap_ratio(gt, output) >= 0.50

        except Exception as e:
            # Fail safe: log but don’t crash the run
            print(f" Error on row {i}: {e}")
            output = "ERROR"
            input_len = -1
            sig_start = sig_end = None
            signal_pos = None
            header_retrieved = False
            exact_logic = False
            partial_logic = False

        results.append({
            "task_id": row.get("task_id", i),
            "retrieved_header": bool(header_retrieved),
            "retrieved_logic_exact": bool(exact_logic),
            "retrieved_logic_partial": bool(partial_logic),
            "token_count": input_len,
            "output": output[:500],  # save truncated output for sanity check
            "input_len_tokens": input_len,
            "signal_tok_start": sig_start if sig_start is not None else -1,
            "signal_tok_end": sig_end if sig_end is not None else -1,
            "signal_pos": signal_pos,
        })

    out = pd.DataFrame(results)
    out.to_csv(output_path, index=False)
    print(f"\nFinished. Results saved to {output_path}")
    print("Columns:", list(out.columns))


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        data_path=args.data_path,
        model_name=args.model_name,
        output_path=args.output_path,
        max_tokens=args.max_tokens,
        max_input_tokens=args.max_input_tokens,
        device=args.device,
    )
