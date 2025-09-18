# 📂 Data Folder

This folder stores datasets used in the project.  
Large raw data files are **not committed to GitHub** (see `.gitignore`).  

### Structure
data/

├─ raw/ # full dataset (downloaded via scripts/download_data.py)

├─ processed/ # generated CSVs (retrieval tasks, truncated prompts, etc.)

├─ sample/ # tiny demo dataset (safe to commit for testing)

└─ .gitkeep # keeps folder tracked in Git


### Notes
- **Raw data** (`data/raw/`) is downloaded from the GitHub release:
  ```bash
  python scripts/download_data.py

Processed data (data/processed/) is created by:

scripts/build_retrieval_dataset.py

scripts/truncate_prompts.py

Sample data (data/sample/) contains a small toy_dataset.csv for quick testing:
python scripts/toy_dataset.py
