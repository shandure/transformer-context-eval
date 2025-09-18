
---

## 📄 `notebooks/README.md`

```markdown
# 📂 Notebooks

This folder contains example Jupyter notebooks that demonstrate how to run the evaluation pipeline.  
They are intended for **interactive exploration and demos** (useful for students, reviewers, and quick experiments).

### Notebooks
- **baseline_eval.ipynb**  
  Runs the pipeline using the baseline model (**DeepSeek-R1-Distill-Qwen-7B**).  
  - Builds a small retrieval dataset  
  - Truncates prompts  
  - Runs evaluation  
  - Saves results to `results/demo_baseline_results.csv`

- **RoPE_Eval.ipynb**  
  Runs the pipeline using the RoPE-scaled model (**LongChat-7B-16k**).  
  - Same steps as baseline, but with extended context length.  
  - Saves results to `results/demo_rope_results.csv`

### Usage
Open either notebook in VS Code or Jupyter Lab and run the cells in order.  
Make sure dependencies are installed:
```bash
pip install -r requirements.txt
