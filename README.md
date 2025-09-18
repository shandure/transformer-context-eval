Transformer Context Evaluation

This project evaluates long-context transformer models on **synthetic function** retrieval tasks, comparing baseline vs RoPE-scaled models.
It was developed as part of my MSc project.

## 🚀 Quickstart

1. **Install dependencies**
   ```bash         
   pip install -r requirements.txt
   ```            

2. **Download the dataset**
   ```bash         
   python scripts/download_data.py
   ```            
   This will place the full dataset in `data/raw/function_dataset/`.

   👉 For a quick demo, you can instead generate a small toy dataset:
   ```bash         
   python scripts/toy_dataset.py
   ```            

3. **Build retrieval dataset**
   ```bash         
   python scripts/build_retrieval_dataset.py \
     --tokenizer_name EleutherAI/gpt-neox-20b \
     --class_type B --num_tasks 300 --num_distractors 20 \
     --out_csv data/processed/class_b_retrieval_dataset.csv
   ```            

4. **Truncate prompts**
   ```bash         
   python scripts/truncate_prompts.py \
     --in_csv data/processed/class_b_retrieval_dataset.csv \
     --out_csv data/processed/class_b_truncated_16384.csv \
     --tokenizer_name lmsys/longchat-7b-16k \
     --max_input_tokens 16384
   ```            

5. **Run evaluation**
   ```bash         
   python -m src.run_retrieval_eval \
     --model_name lmsys/longchat-7b-16k \
     --data_path data/processed/class_b_truncated_16384.csv \
     --output_path results/class_b_results_7b_LITM_16384.csv \
     --max_tokens 1024 \
     --max_input_tokens 16384
   ```            

Results will be saved to:


## 📂 Repository Structure

```bash
data/         # datasets (raw, processed, sample) → see data/README.md
notebooks/    # demo Jupyter notebooks (baseline vs RoPE) → see notebooks/README.md
results/      # CSV evaluation outputs
scripts/      # dataset preparation scripts
src/          # core evaluation + plotting code
```

📓 Notebooks

Interactive notebooks are available for quick experimentation:

- baseline_eval.ipynb → Baseline model (DeepSeek-R1-Distill-Qwen-7B)
- RoPE_Eval.ipynb → RoPE-scaled model (LongChat-7B-16k)

See ```notebooks/README.md``` for details.


📊 Data

- Full dataset is downloaded via GitHub release using scripts/download_data.py.
- Processed datasets are generated via build/truncate scripts.
- A sample toy dataset is included for lightweight testing.

See ```data/README.md``` for details.


📜 License

This project is licensed under the terms of the MIT License (```LICENSE```).
