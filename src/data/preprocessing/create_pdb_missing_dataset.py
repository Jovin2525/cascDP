"""
Step 2 of PDB_missing pipeline.

Reads the entity ID list produced by fetch_pdb_entity_ids.py, batch-fetches
sequences and UNOBSERVED_RESIDUE_XYZ feature data from the RCSB GraphQL API,
computes union disorder labels across all chains, then clusters
at 30% sequence identity via MMseqs2 and writes the final dataset in the
project's 9-line format.

Usage:
    # Full pipeline (fetch + label + cluster)
    python scripts/data/create_pdb_missing_dataset.py \
        --ids    data/final_cleaned_dataset/pdb_entity_ids.txt \
        --output data/final_cleaned_dataset/pdb_missing_caid4.txt

    # Resume -- skip fetch if raw file already has content
    python scripts/data/create_pdb_missing_dataset.py \
        --ids    data/final_cleaned_dataset/pdb_entity_ids.txt \
        --output data/final_cleaned_dataset/pdb_missing_caid4.txt \
        --raw    data/final_cleaned_dataset/pdb_missing_raw.txt

    # Debug -- print first batch response and exit
    python scripts/data/create_pdb_missing_dataset.py ... --debug

Output 9-line format per entry:
    >{rcsb_id}|PDB_missing
    {sequence}
    {disorder_labels}        # 1=missing/disordered, 0=ordered
    {mask '-'*L}             # Protein_binding   (unannotated)
    {mask '-'*L}             # Nucleic_acid_binding
    {mask '-'*L}             # Ion_binding
    {mask '-'*L}             # Lipid_binding
    {mask '-'*L}             # Combined_binding
    {mask '-'*L}             # Flexible_linker
"""

import argparse
import json
import subprocess
import time
from pathlib import Path
import requests

GRAPHQL_URL = "https://data.rcsb.org/graphql"
BATCH_SIZE = 50
RETRY_LIMIT = 3
SLEEP_BETWEEN_BATCHES = 0.2 # seconds

DEBUG = False   # flipped by --debug flag

# Verified correct schema path:
#   polymer_entities -> polymer_entity_instances -> rcsb_polymer_instance_feature
#   type UNOBSERVED_RESIDUE_XYZ, positions {beg_seq_id, end_seq_id}
GRAPHQL_QUERY = """
query getEntities($ids: [String!]!) {
  polymer_entities(entity_ids: $ids) {
    rcsb_id
    entity_poly {
      pdbx_seq_one_letter_code_can
    }
    polymer_entity_instances {
      rcsb_id
      rcsb_polymer_instance_feature {
        type
        feature_positions {
          beg_seq_id
          end_seq_id
        }
      }
    }
  }
}
"""

# Helpers
def write_entry(f, rcsb_id: str, sequence: str, label_string: str) -> None:
    # Write one entity in the project 9-line dataset format
    mask = "-" * len(sequence)
    f.write(f">{rcsb_id}|PDB_missing\n")
    f.write(f"{sequence}\n")
    f.write(f"{label_string}\n")
    for _ in range(6):      # 6 functional tracks -- all masked
        f.write(f"{mask}\n")

def compute_label_string(sequence: str, instances: list) -> str:
    """
    Compute union disorder labels across all chains of an entity.

    instances: list of polymer_entity_instance dicts, each containing
               rcsb_polymer_instance_feature with UNOBSERVED_RESIDUE_XYZ ranges.

    Logic:
    - For each chain, collect positions from UNOBSERVED_RESIDUE_XYZ feature ranges.
    - Union across all chains: a position is '1' if unobserved in ANY chain.
      This captures real flexibility - a residue missing in one crystal copy
      has genuine disorder signal even if another copy resolves it.
    """
    all_missing = set()   # union of 1-based unobserved positions across all chains

    for inst in instances:
        features = inst.get("rcsb_polymer_instance_feature") or []

        for feat in features:
            if feat.get("type") != "UNOBSERVED_RESIDUE_XYZ":
                continue
            for pos in (feat.get("feature_positions") or []):
                beg = pos.get("beg_seq_id")
                end = pos.get("end_seq_id")
                if beg is not None and end is not None:
                    all_missing.update(range(int(beg), int(end) + 1))
                elif beg is not None:
                    all_missing.add(int(beg))

    return "".join("1" if (i + 1) in all_missing else "0"
                   for i in range(len(sequence)))

# Fetching
def fetch_batch(entity_ids: list, debug: bool = False) -> list:
    """
    POST one GraphQL query for a batch of entity IDs (e.g. ['4HHB_1', '1IDP_1']).
    Retries up to RETRY_LIMIT on failure.
    Returns list of polymer_entity dicts.
    """
    payload = {"query": GRAPHQL_QUERY, "variables": {"ids": entity_ids}}

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            resp = requests.post(GRAPHQL_URL, json=payload, timeout=120)
            if resp.status_code == 200:
                body = resp.json()
                if body is None:
                    print(f"    Empty JSON body on attempt {attempt} -- retrying...")
                    time.sleep(2 ** attempt)
                    continue
                if debug:
                    print("DEBUG -- raw response body:")
                    print(json.dumps(body, indent=2)[:4000])
                if body.get("errors"):
                    print(f"    GraphQL errors on attempt {attempt}: "
                          f"{[e['message'][:80] for e in body['errors'][:2]]}")
                data     = body.get("data") or {}
                entities = data.get("polymer_entities") or []
                return [e for e in entities if e]   # filter null entries
            print(f"    HTTP {resp.status_code} on attempt {attempt} -- retrying...")
        except (requests.RequestException, ValueError) as e:
            print(f"    Request/parse error on attempt {attempt}: {e} -- retrying...")
        time.sleep(2 ** attempt)

    print(f"    Giving up on batch starting with {entity_ids[0]}")
    return []

# Stage 1: Fetch + Label
def fetch_and_label(ids_file: str, raw_output: str) -> int:
    """
    Iterate all entity IDs, fetch via GraphQL, compute disorder labels,
    write to raw_output in 9-line format.  Returns number of entries written.
    """
    ids_path = Path(ids_file)
    if not ids_path.exists():
        raise FileNotFoundError(
            f"Entity IDs file not found: {ids_file}\n"
            "Run scripts/data/fetch_pdb_entity_ids.py first."
        )

    with open(ids_path) as f:
        entity_ids = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(entity_ids):,} entity IDs from {ids_file}")

    raw_path = Path(raw_output)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    total   = len(entity_ids)

    with open(raw_path, "w") as out_f:
        for i in range(0, total, BATCH_SIZE):
            batch      = entity_ids[i : i + BATCH_SIZE]
            debug_this = (DEBUG and i == 0)
            entities   = fetch_batch(batch, debug=debug_this)

            if debug_this:
                print("DEBUG -- exiting after first batch (--debug flag)")
                break

            for entity in entities:
                rcsb_id = entity.get("rcsb_id", "")

                ep = entity.get("entity_poly")
                if not ep:
                    skipped += 1
                    continue

                sequence = (ep.get("pdbx_seq_one_letter_code_can") or "").replace("\n", "").strip()
                if not sequence:
                    skipped += 1
                    continue

                instances    = entity.get("polymer_entity_instances") or []
                label_string = compute_label_string(sequence, instances)

                write_entry(out_f, rcsb_id, sequence, label_string)
                written += 1

            time.sleep(SLEEP_BETWEEN_BATCHES)

            done = min(i + BATCH_SIZE, total)
            if done % 5000 == 0 or done == total:
                print(f"  Progress: {done:,}/{total:,} entities "
                      f"({written:,} written, {skipped:,} skipped)...", flush=True)

    print(f"\nRaw labeled dataset -> {raw_path}")
    print(f"  Written : {written:,}")
    print(f"  Skipped : {skipped:,}")
    return written

# Stage 2: MMseqs2 30% clustering
def read_raw_entries(raw_file: str) -> dict:
    """
    Parse the raw 9-line dataset file.
    Returns {pid: (sequence, [9_raw_lines])}.
    """
    entries = {}
    with open(raw_file) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if not lines[i].startswith(">"):
            i += 1
            continue
        block = lines[i : i + 9]
        if len(block) < 9:
            break
        pid      = block[0].rstrip()[1:]    # strip '>'
        sequence = block[1].rstrip()
        entries[pid] = (sequence, block)
        i += 9

    return entries


def run_mmseqs_cluster(entries: dict, tmp_prefix: str) -> set:
    """
    Write a FASTA, run mmseqs easy-cluster at 30% identity,
    return set of representative IDs (original encoding).
    """
    fasta_file     = f"{tmp_prefix}.fasta"
    cluster_tsv    = f"{tmp_prefix}_cluster.tsv"
    tmp_mmseqs_dir = f"{tmp_prefix}_tmp"
    cluster_prefix = tmp_prefix

    with open(fasta_file, "w") as fasta_f:
        for pid, (seq, _) in entries.items():
            safe_id = pid.replace("|", "__")    # MMseqs2 dislikes '|'
            fasta_f.write(f">{safe_id}\n{seq}\n")

    cmd = [
        "mmseqs", "easy-cluster",
        fasta_file, cluster_prefix, tmp_mmseqs_dir,
        "--min-seq-id", "0.30",
        "-c", "0.8",
        "--cov-mode", "0",
    ]
    print(f"Running MMseqs2: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        raise RuntimeError(
            "mmseqs not found.  Install: conda install -c bioconda mmseqs2"
        )
    except subprocess.CalledProcessError as e:
        print(f"mmseqs failed:\n{e.stderr.decode()}")
        raise

    reps_safe = set()
    with open(cluster_tsv) as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts:
                reps_safe.add(parts[0])

    reps = {eid.replace("__", "|") for eid in reps_safe}
    print(f"MMseqs2: {len(entries):,} -> {len(reps):,} representatives at 30% identity")
    return reps

def dedup_exact(entries: dict) -> dict:
    """
    Remove exact-sequence duplicates (100% identity dedup).
    For each unique sequence, keeps the first entry seen.
    Returns new dict of {pid: (sequence, block)}.
    """
    seen_seqs = {}
    deduped   = {}
    for pid, (seq, block) in entries.items():
        if seq not in seen_seqs:
            seen_seqs[seq] = pid
            deduped[pid]   = (seq, block)
    removed = len(entries) - len(deduped)
    print(f"  100% identity dedup: {len(entries):,} -> {len(deduped):,} "
          f"({removed:,} exact duplicates removed)")
    return deduped


def cluster_and_write(raw_file: str, final_output: str) -> None:
    """Load raw file, dedup at 100%, cluster at 30%, write representatives."""
    print(f"\nReading raw entries from {raw_file}...")
    entries = read_raw_entries(raw_file)
    print(f"  Loaded {len(entries):,} entries")

    if not entries:
        raise RuntimeError(
            f"No valid entries in {raw_file}.\n"
            f"Delete it and re-run: rm {raw_file}"
        )

    # Stage 2a: 100% identity dedup
    print("\nStage 2a: 100% identity deduplication...")
    entries = dedup_exact(entries)

    tmp_dir    = Path(raw_file).parent / "tmp_pdb_cluster"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_prefix = str(tmp_dir / "pdb_missing")

    reps = run_mmseqs_cluster(entries, tmp_prefix)

    final_path = Path(final_output)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(final_path, "w") as out_f:
        for pid, (_, block) in entries.items():
            if pid in reps:
                out_f.writelines(block)
                written += 1

    print(f"Final PDB_missing dataset -> {final_path}")
    print(f"  Entries: {written:,}")

def main():
    parser = argparse.ArgumentParser(
        description="Build PDB_missing disorder dataset from RCSB X-ray structures"
    )
    parser.add_argument("--ids",    required=True,
                        help="Entity ID list from fetch_pdb_entity_ids.py")
    parser.add_argument("--output", required=True,
                        help="Final clustered output path "
                             "(e.g. data/final_cleaned_dataset/pdb_missing_caid4.txt)")
    parser.add_argument("--raw",    default=None,
                        help="Intermediate raw file path. "
                             "If it exists with content, fetch step is skipped. "
                             "Defaults to <output_dir>/pdb_missing_raw.txt")
    parser.add_argument("--debug",  action="store_true",
                        help="Print first batch GraphQL response and exit")
    args = parser.parse_args()

    global DEBUG
    DEBUG = args.debug

    output_path = Path(args.output)
    raw_file    = args.raw or str(output_path.parent / "pdb_missing_raw.txt")
    raw_path    = Path(raw_file)

    # Stage 1 -- skip if raw file already populated
    has_content = (raw_path.exists()
                   and raw_path.stat().st_size > 0
                   and any(l.startswith(">") for l in open(raw_path)))

    if has_content and not args.debug:
        print(f"Raw file already has content ({raw_file}) -- skipping fetch step.")
    else:
        if raw_path.exists() and not has_content:
            print("Raw file exists but is empty/invalid -- removing and re-fetching...")
            raw_path.unlink()
        print("=" * 70)
        print("STAGE 1: Fetching sequences and computing disorder labels")
        print("=" * 70)
        fetch_and_label(args.ids, raw_file)
        if args.debug:
            return

    # Stage 2
    print("\n" + "=" * 70)
    print("STAGE 2: 100% identity dedup + MMseqs2 clustering at 30% identity")
    print("=" * 70)
    cluster_and_write(raw_file, args.output)
    print("\nDone.")

if __name__ == "__main__":
    main()