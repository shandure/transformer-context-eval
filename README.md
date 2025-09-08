# MSc Project: Evaluating Long-Range Dependency Handling in Transformer Models  

This repository contains the code, data, and results from my MSc Computer Science dissertation at Queen Mary University of London. The project investigates whether **RoPE scaling** improves transformer performance on **long-range multi-step code retrieval tasks**.  

---

## 📌 Project Overview
- **Goal:** Evaluate transformer models’ ability to resolve long-range dependencies in code generation.  
- **Baseline:** DeepSeek-R1-Distill-Qwen-7B (4k tokens).  
- **Extended:** LongChat-7B/13B with RoPE scaling (16k tokens).  
- **Benchmark:** Synthetic multi-step function retrieval tasks with distractors.  

---

## 📂 Repository Structure
- `src/` – core scripts for preprocessing, evaluation, and plotting.  
- `notebooks/` – Jupyter notebooks for exploratory analysis and experiments.  
- `data/` – benchmark datasets.  
- `results/` – CSVs and figures from evaluations.  

---

## ⚡ Key Results
- RoPE scaling improves retrieval accuracy at longer context windows (+10.4pp exact match).  
- Reduced truncation failures compared to baseline 4k models.  
- “Lost in the Middle” effect is alleviated but not eliminated.  

---

## 🔧 Installation
```bash
git clone https://github.com/username/msc-long-context-llm.git
cd msc-long-context-llm
pip install -r requirements.txt

