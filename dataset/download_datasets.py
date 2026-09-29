"""
Dataset Downloader & Merger
============================
Downloads and merges two datasets:
1. MRL Eye Dataset     — 84,898 images (open/closed eyes, multiple subjects, lighting conditions)
2. CEW Dataset         — 2,420 images  (Closed Eyes in the Wild, real-world photos)

Final merged dataset structure:
  dataset/
  └── merged/
      ├── open/     (~45,000+ images)
      └── closed/   (~45,000+ images)
"""

import os
import sys
import shutil
import zipfile
import requests
import subprocess
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "raw"
MERGED_DIR  = BASE_DIR / "merged"

MRL_GDRIVE_IDS = {
    "mrl_open":   "1p-FsAOuGWkiHrA3pFKVQCuC0VCQawfaF",   # MRL open-eye subset
    "mrl_closed": "1p-FsAOuGWkiHrA3pFKVQCuC0VCQawfaF",   # MRL closed-eye subset
}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _download_file(url: str, dest: Path, desc: str = "Downloading"):
    """Stream-download a file with a progress bar."""
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=desc) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"✅  Saved → {dest}")


def _gdown(file_id: str, dest: Path):
    """Download from Google Drive using gdown."""
    try:
        import gdown
        gdown.download(id=file_id, output=str(dest), quiet=False)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "gdown", "-q"], check=True)
        import gdown
        gdown.download(id=file_id, output=str(dest), quiet=False)


def _unzip(zip_path: Path, dest: Path):
    print(f"📦  Extracting {zip_path.name} …")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)
    print(f"✅  Extracted → {dest}")


# ─── MRL Eye Dataset ─────────────────────────────────────────────────────────

def download_mrl():
    """
    MRL Eye Dataset official page: http://mrl.cs.vsb.cz/eyedataset
    
    The dataset is split into 16 subsets (s0001–s0016).
    Each subset ZIP contains two folders: open/ and close/.
    
    We download via the direct HTTP links provided on the official page.
    """
    print("\n" + "="*60)
    print("  STEP 1 — MRL Eye Dataset (84,898 images)")
    print("="*60)

    mrl_base = "http://mrl.cs.vsb.cz/data/eyedataset"
    subsets  = [f"mrlEyes_2018_01.zip"]   # Main archive (~1.2 GB)

    mrl_dir = DATASET_DIR / "mrl"
    mrl_dir.mkdir(parents=True, exist_ok=True)

    for fname in subsets:
        dest = mrl_dir / fname
        if dest.exists():
            print(f"⚡  {fname} already downloaded, skipping.")
        else:
            try:
                _download_file(f"{mrl_base}/{fname}", dest, desc=fname)
            except Exception as e:
                print(f"⚠️  Direct download failed ({e}).")
                print("    Trying Kaggle mirror …")
                _download_kaggle_mrl(mrl_dir)
                return mrl_dir

        _unzip(dest, mrl_dir)

    return mrl_dir


def _download_kaggle_mrl(dest: Path):
    """Kaggle mirror: prasadvpatil/mrl-dataset (same data, 84k images)."""
    dest.mkdir(parents=True, exist_ok=True)
    print("📥  Downloading MRL via Kaggle API …")
    print("    Make sure kaggle.json is in ~/.kaggle/")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", "prasadvpatil/mrl-dataset", "--unzip", "-p", str(dest)],
        check=True
    )
    print("✅  MRL Kaggle download complete.")


# ─── CEW Dataset ─────────────────────────────────────────────────────────────

def download_cew():
    """
    CEW — Closed Eyes in the Wild
    Original: http://parnec.nuaa.edu.cn/xtan/data/ClosedEyes.zip
    Kaggle mirror: serenaraju/closed-eyes-in-the-wild-cew-dataset
    """
    print("\n" + "="*60)
    print("  STEP 2 — CEW Dataset (2,420 images)")
    print("="*60)

    cew_dir = DATASET_DIR / "cew"
    cew_dir.mkdir(parents=True, exist_ok=True)

    # Try direct download first
    primary_url = "http://parnec.nuaa.edu.cn/xtan/data/ClosedEyes.zip"
    dest = cew_dir / "ClosedEyes.zip"

    if dest.exists():
        print("⚡  CEW already downloaded, skipping.")
    else:
        try:
            _download_file(primary_url, dest, desc="CEW Dataset")
        except Exception as e:
            print(f"⚠️  Direct download failed ({e}). Trying Kaggle …")
            _download_kaggle_cew(cew_dir)
            return cew_dir

    _unzip(dest, cew_dir)
    return cew_dir


def _download_kaggle_cew(dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    print("📥  Downloading CEW via Kaggle API …")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", "serenaraju/closed-eyes-in-the-wild-cew-dataset",
         "--unzip", "-p", str(dest)],
        check=True
    )
    print("✅  CEW Kaggle download complete.")


# ─── Merge & Preprocess ──────────────────────────────────────────────────────

def _is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".pgm"}


def _copy_images(src_dir: Path, dest_dir: Path, label: str, start_idx: int = 0) -> int:
    """Copy all images from src_dir → dest_dir, renaming sequentially."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    imgs  = [p for p in src_dir.rglob("*") if _is_image(p)]
    count = 0
    for i, img_path in enumerate(tqdm(imgs, desc=f"  Copying {label}")):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.resize(img, (224, 224))
            new_name = f"{label}_{start_idx + i:06d}.jpg"
            cv2.imwrite(str(dest_dir / new_name), img)
            count += 1
        except Exception:
            pass
    return count


def merge_datasets(mrl_dir: Path, cew_dir: Path):
    """
    Scan MRL and CEW raw folders, classify images into open/closed,
    resize to 224×224, and copy into merged/open & merged/closed.
    """
    print("\n" + "="*60)
    print("  STEP 3 — Merging Datasets into merged/open & merged/closed")
    print("="*60)

    open_dir   = MERGED_DIR / "open"
    closed_dir = MERGED_DIR / "closed"
    open_dir.mkdir(parents=True, exist_ok=True)
    closed_dir.mkdir(parents=True, exist_ok=True)

    total_open = total_closed = 0

    # ── MRL: folder names contain 'open'/'close' ──────────────────────────
    print("\n[MRL] Scanning …")
    for folder in mrl_dir.rglob("*"):
        if not folder.is_dir():
            continue
        fname = folder.name.lower()
        if "open" in fname:
            n = _copy_images(folder, open_dir, "mrl_open", total_open)
            total_open += n
        elif "close" in fname or "closed" in fname:
            n = _copy_images(folder, closed_dir, "mrl_closed", total_closed)
            total_closed += n

    # ── CEW: folder names contain 'openFace'/'closedFace' ─────────────────
    print("\n[CEW] Scanning …")
    for folder in cew_dir.rglob("*"):
        if not folder.is_dir():
            continue
        fname = folder.name.lower()
        if "open" in fname:
            n = _copy_images(folder, open_dir, "cew_open", total_open)
            total_open += n
        elif "close" in fname or "closed" in fname:
            n = _copy_images(folder, closed_dir, "cew_closed", total_closed)
            total_closed += n

    print(f"""
╔══════════════════════════════════════╗
║   ✅  DATASET MERGE COMPLETE          ║
╠══════════════════════════════════════╣
║  📂 Open   images : {total_open:>6,}           ║
║  📂 Closed images : {total_closed:>6,}           ║
║  📊 Total          : {total_open+total_closed:>6,}           ║
╚══════════════════════════════════════╝
    """)
    return total_open, total_closed


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("""
╔══════════════════════════════════════════════════════╗
║   DROWSINESS GUARD — Dataset Downloader & Merger     ║
╚══════════════════════════════════════════════════════╝

Datasets to download:
  1. MRL Eye Dataset  — 84,898 images  (academic, multi-condition)
  2. CEW Dataset      — 2,420 images   (real-world closed eyes)

Total expected: ~87,000+ images → merged into open/closed classes
    """)

    mrl_dir = download_mrl()
    cew_dir = download_cew()
    merge_datasets(mrl_dir, cew_dir)

    print("\n🎉  All done! Now run:  python train/train_model.py")


if __name__ == "__main__":
    main()
