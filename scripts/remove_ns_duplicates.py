#!/usr/bin/env python
"""Audit and clean duplicate ticker CSV files.
Keeps the canonical file without suffix (e.g., RELIANCE.csv) and removes
any duplicate with .NS suffix (e.g., RELIANCE.NS.csv) and associated metadata.
"""
import os
from pathlib import Path

def find_duplicates(data_dir: Path):
    duplicates = []
    for csv_path in data_dir.rglob('*.csv'):
        name = csv_path.stem  # without extension
        # Check if name ends with .NS
        if name.endswith('.NS'):
            base_name = name[:-3]
            canonical = data_dir / f"{base_name}.csv"
            if canonical.exists():
                duplicates.append((canonical, csv_path))
    return duplicates

def main():
    base = Path('d:/stockproject/data/stocks')
    dup_pairs = find_duplicates(base)
    if not dup_pairs:
        print('No duplicate .NS CSV files found.')
        return
    print('Found duplicate CSVs:')
    for canonical, redundant in dup_pairs:
        print(f'  Canonical: {canonical}\n  Redundant: {redundant}')
        # Delete redundant CSV
        redundant.unlink(missing_ok=True)
        # Delete its metadata if exists
        meta = redundant.with_suffix(redundant.suffix + '.metadata.json')
        if meta.exists():
            print(f'  Deleting metadata: {meta}')
            meta.unlink(missing_ok=True)
    print('Cleanup complete.')

if __name__ == '__main__':
    main()
