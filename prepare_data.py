"""Download official Kaggle archive and cache resized tensors; no test leakage."""
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
import numpy as np
import requests
import torch
from PIL import Image, ImageOps

URL = 'https://www.kaggle.com/api/v1/datasets/download/paultimothymooney/chest-xray-pneumonia'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--archive', type=Path, default=Path('data/chest-xray.zip'))
    p.add_argument('--output', type=Path, default=Path('data'))
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if not a.archive.exists():
        a.archive.parent.mkdir(parents=True, exist_ok=True)
        partial = a.archive.with_suffix('.partial')
        with requests.get(URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            with partial.open('wb') as f:
                for chunk in r.iter_content(8 * 1024 * 1024):
                    f.write(chunk)
        if not zipfile.is_zipfile(partial):
            raise RuntimeError('Kaggle did not return a ZIP. Download the dataset manually.')
        partial.replace(a.archive)
    manifest = {}
    with zipfile.ZipFile(a.archive) as z:
        for split in ['train', 'val', 'test']:
            candidates = [n for n in z.namelist() if f'/{split}/' in n and
                          n.lower().endswith(('.jpeg', '.jpg', '.png')) and '__MACOSX' not in n]
            # Kaggle ZIP may contain a duplicate nested chest_xray/chest_xray tree.
            by_name = {}
            for name in sorted(candidates, key=lambda s: (s.count('/'), s)):
                key = name.split(f'/{split}/', 1)[1]
                by_name.setdefault(key, name)
            images, entries = [], []
            for key, name in sorted(by_name.items()):
                raw = z.read(name)
                im = Image.open(io.BytesIO(raw)).convert('L')
                # Preserve aspect ratio, letterbox to square, then resize.
                im = ImageOps.pad(im, (64, 64), method=Image.Resampling.BILINEAR, color=0)
                images.append(np.array(im, dtype=np.uint8))
                entries.append({'file': key, 'sha256': hashlib.sha256(raw).hexdigest()})
            if not images:
                raise RuntimeError(f'No images in {split}')
            torch.save(torch.from_numpy(np.stack(images))[:, None], a.output / f'{split}.pt')
            manifest[split] = entries
            print(split, len(images), flush=True)
    hashes = {k: {v['sha256'] for v in vs} for k, vs in manifest.items()}
    overlaps = {f'{a}-{b}': len(hashes[a] & hashes[b])
                for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]}
    (a.output / 'manifest.json').write_text(json.dumps({'source': URL, 'splits': manifest,
                                                     'exact_hash_overlap': overlaps}, indent=2))
    print('Exact overlap:', overlaps, flush=True)


if __name__ == '__main__':
    main()
