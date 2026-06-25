"""
Split combined dataset files into DisProt-only and PDB_missing-only subsets.

Reads 9-line protein blocks from the combined train/val files and routes
each block to a source-specific output file based on the header tag
(>ProteinID|Source).

Usage:
    python scripts/data/split_dataset_by_source.py
    python scripts/data/split_dataset_by_source.py --input_dir data/final_cleaned_dataset
    python scripts/data/split_dataset_by_source.py --splits train val
"""
import argparse
import sys
from pathlib import Path

def parse_blocks(filepath: Path) -> list[list[str]]:
    blocks = []
    current: list[str] = []

    with open(filepath, 'r') as f:
        for raw_line in f:
            line = raw_line.rstrip('\n')
            if line.startswith('>'):
                if current:
                    blocks.append(current)
                current = [line]
            elif current:
                current.append(line)

        if current:
            blocks.append(current)

    return blocks

def get_source(block: list[str]) -> str:
    # Extract source tag from header line: >ID|Source -> Source
    header = block[0][1:].strip()  # strip '>'
    if '|' in header:
        return header.rsplit('|', 1)[1]
    return 'unknown'

def split_file(input_file: Path, output_dir: Path, prefix: str):
    # Split one dataset file by source. Returns per-source counts
    blocks = parse_blocks(input_file)

    disprot_blocks = []
    pdb_blocks = []

    for b in blocks:
        source = get_source(b)
        if 'DisProt' in source:
            disprot_blocks.append(b)
        elif 'PDB' in source:
            pdb_blocks.append(b)
        else:
            print(f"  WARNING: unknown source '{source}' in {b[0]}, skipping")

    disprot_out = output_dir / f"{prefix}_disprot_only.txt"
    pdb_out = output_dir / f"{prefix}_pdb_only.txt"

    for out_path, out_blocks, label in [
        (disprot_out, disprot_blocks, "DisProt"),
        (pdb_out, pdb_blocks, "PDB_missing"),
    ]:
        with open(out_path, 'w') as f:
            for block in out_blocks:
                f.write('\n'.join(block) + '\n')
        print(f"  {label}: {len(out_blocks)} proteins -> {out_path}")

    return len(disprot_blocks), len(pdb_blocks)

def main():
    parser = argparse.ArgumentParser(description="Split dataset by source (DisProt vs PDB_missing)")
    parser.add_argument('--input_dir', type=str, default='data/final_cleaned_dataset')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    for split in args.splits:
        input_file = input_dir / f"{split}_final_update_or_caid4_unaltered_data.txt"
        if not input_file.exists():
            print(f"SKIP: {input_file} not found", file=sys.stderr)
            continue

        print(f"\n=== {split} split: {input_file} ===")
        n_disprot, n_pdb = split_file(input_file, input_dir, split)
        print(f"  Total: {n_disprot + n_pdb} ({n_disprot} DisProt + {n_pdb} PDB_missing)")

if __name__ == '__main__':
    main()