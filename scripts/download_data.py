# scripts/download_data.py
from pathlib import Path
from urllib.request import urlretrieve
import hashlib, zipfile


URL = "https://github.com/shandure/transformer-context-eval/releases/download/v1-data/function_dataset-v1.zip"
SHA256 = "5B1149569BDA059DF9E94587C93884C65959DEE417005A8D6AB921B350F69B5C"

TARGET_SUBDIR = "function_dataset"  # folder name inside data/raw/

def sha256sum(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    root = Path(__file__).resolve().parents[1]
    data_raw = root / "data" / "raw"
    data_raw.mkdir(parents=True, exist_ok=True)

    zip_path = data_raw / "function_dataset-v1.zip"
    if not zip_path.exists():
        print(f"Downloading dataset from {URL} …")
        urlretrieve(URL, zip_path)

    print("Verifying checksum …")
    actual = sha256sum(zip_path)
    if actual != SHA256:
        raise SystemExit(f"Checksum mismatch:\n expected {SHA256}\n got      {actual}")

    dest = data_raw / TARGET_SUBDIR
    if not dest.exists():
        print("Extracting …")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(data_raw)

    print(f"Ready: {dest}")

if __name__ == "__main__":
    main()
