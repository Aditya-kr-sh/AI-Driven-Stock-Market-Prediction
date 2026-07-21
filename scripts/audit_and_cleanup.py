#!/usr/bin/env python
import shutil
from pathlib import Path

base = Path('d:/stockproject/data')
canonical_dir = base / 'stocks'
canonical_dir.mkdir(parents=True, exist_ok=True)

# Gather all CSV files under data (including subfolders)
csv_paths = list(base.rglob('*.csv'))

# Map ticker -> list of paths
ticker_map = {}
for p in csv_paths:
    ticker = p.stem.upper()
    ticker_map.setdefault(ticker, []).append(p)

print('=== Duplicate report before cleanup ===')
for ticker, paths in ticker_map.items():
    if len(paths) > 1:
        print(f'{ticker}: {len(paths)} copies')
        for pp in paths:
            print('  ', pp)

# Perform cleanup
for ticker, paths in ticker_map.items():
    # Desired canonical path
    canon_path = canonical_dir / f'{ticker}.csv'
    # Determine which file to keep (prefer one already in canonical dir)
    keep = None
    for p in paths:
        if p.parent == canonical_dir:
            keep = p
            break
    if not keep:
        # Move the first occurrence to canonical location
        keep = paths[0]
        print(f'Moving {keep} -> {canon_path}')
        shutil.move(str(keep), str(canon_path))
        # also move its metadata if present
        meta_src = keep.with_suffix(keep.suffix + '.metadata.json')
        if meta_src.exists():
            shutil.move(str(meta_src), str(canon_path.with_suffix(canon_path.suffix + '.metadata.json')))
    # Delete all other copies
    for p in paths:
        if p == keep:
            continue
        print(f'Deleting duplicate {p}')
        p.unlink(missing_ok=True)
        meta = p.with_suffix(p.suffix + '.metadata.json')
        if meta.exists():
            print(f'Deleting duplicate metadata {meta}')
            meta.unlink(missing_ok=True)

# Clean up empty index subfolders under data (e.g., data/nifty50)
for sub in base.iterdir():
    if sub.is_dir() and sub.name not in ('stocks', 'processed'):
        try:
            if not any(sub.iterdir()):
                sub.rmdir()
                print(f'Removed empty directory {sub}')
        except Exception:
            pass

print('=== Final CSV listing in canonical directory ===')
for p in sorted(canonical_dir.glob('*.csv')):
    print(p)
print('Cleanup complete')
