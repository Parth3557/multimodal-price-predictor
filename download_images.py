"""Download all product images once, so training reads them from disk.

Usage:  python download_images.py --data dataset --out images --workers 16
Saves each image as images/{sample_id}.jpg
"""
import argparse, os
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd, requests
from tqdm import tqdm


def fetch(sample_id, url, out_dir):
    path = os.path.join(out_dir, f"{sample_id}.jpg")
    if os.path.exists(path):
        return True
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--out", default="images")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = pd.concat([pd.read_csv(os.path.join(a.data, f)) for f in ("train.csv", "test.csv")])
    ok = 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(fetch, r.sample_id, r.image_link, a.out) for r in df.itertuples()]
        for f in tqdm(as_completed(futs), total=len(futs)):
            ok += f.result()
    print(f"downloaded {ok}/{len(df)} images (missing ones become blank images in training)")


if __name__ == "__main__":
    main()
