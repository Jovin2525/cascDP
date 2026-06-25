"""
Filter dataset splits to Phase 2 (function-annotated) proteins.

Reads the full caid4 unaltered splits and writes *_phase2_any_data.txt files
containing only proteins that have at least one annotated functional residue
(linker OR any binding type).

PDB_missing entries are excluded because all their function tracks are '-'
(unknown), meaning they contribute nothing to function-head training.

Usage:
    python scripts/data/filter_phase2_dataset.py
    python scripts/data/filter_phase2_dataset.py --input_dir data/final_cleaned_dataset
    python scripts/data/filter_phase2_dataset.py --splits train val test
"""

import argparse
import sys
from pathlib import Path

def has_functional_annotation(block_lines: list[str]) -> bool:
    """
    Return True if any function track in the block contains at least one '1'.

    Block layout (9 lines, 0-indexed):
        0  >pid|source
        1  sequence
        2  disorder labels
        3  protein binding
        4  nucleic acid binding
        5  ion binding
        6  lipid binding
        7  combined binding
        8  flexible linker

    Lines 3-8 are the function tracks.
    A '-' character means unknown/unannotated; only '1' counts as annotated.
    """
    for track_line in block_lines[3:9]:
        if '1' in track_line:
            return True
    return False

def parse_blocks(filepath: Path) -> list[list[str]]:
    """
    Parse a dataset file into a list of 9-line blocks.
    Each block = [header, sequence, disorder, track1, ..., track6].
    """
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

def filter_split(input_file: Path, output_file: Path) -> tuple[int, int]:
    blocks = parse_blocks(input_file)
    total = len(blocks)

    kept_blocks = [b for b in blocks if len(b) >= 9 and has_functional_annotation(b)]
    kept = len(kept_blocks)

    with open(output_file, 'w') as f:
        for block in kept_blocks:
            for line in block:
                f.write(line + '\n')

    return total, kept

def main():
    parser = argparse.ArgumentParser(
        description='Filter dataset splits to Phase 2 annotated proteins.'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='data/final_cleaned_dataset',
        help='Directory containing the unaltered split files (default: data/final_cleaned_dataset)'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val', 'test'],
        help='Split names to process (default: train val test)'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='final_update_or_caid4',
        help='Filename infix, e.g. "final_update_or_caid4" → {split}_{prefix}_unaltered_data.txt'
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Input directory : {input_dir}")
    print(f"Splits          : {args.splits}")
    print(f"Prefix          : {args.prefix}")
    print()

    for split in args.splits:
        input_name  = f"{split}_{args.prefix}_unaltered_data.txt"
        output_name = f"{split}_{args.prefix}_phase2_any_data.txt"

        input_file  = input_dir / input_name
        output_file = input_dir / output_name

        if not input_file.exists():
            print(f"[{split}] SKIP — file not found: {input_file}")
            continue

        total, kept = filter_split(input_file, output_file)
        dropped = total - kept
        pct = 100.0 * kept / total if total > 0 else 0.0

        print(
            f"[{split}]  {total:>6,} total  →  {kept:>5,} kept "
            f"({pct:.1f}%)  |  {dropped:>5,} dropped  →  {output_file.name}"
        )

if __name__ == '__main__':
    main()