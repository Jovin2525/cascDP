"""
Data cleaning and dataset splitting pipeline.
1. Load DisProt 2025_12.
2. Load CAID3 and remove matching sequences from DisProt 2025_12.
3. Cluster the filtered DisProt pool using MMseqs2.
4. Split clusters (60/20/20) -> Train / Val / Test.
5. Load PDB_missing, filter against DisProt + CAID3, merge into master.
6. Generate output files (Unaltered variant only).
"""

import os
import sys
import subprocess
import random
from collections import defaultdict
from pathlib import Path

# Add parent directory to path to allow imports if run as script
sys.path.append(str(Path(__file__).resolve().parent.parent))

from .utils import (
    parse_fasta,
    fetch_uniprot_sequence,
    load_sequence_cache,
    save_sequence_cache,
    parse_consensus_file,
    parse_region_file,
    create_disorder_labels_from_idpo,
    create_functional_labels
)
from .onto_mapping import FUNCTIONAL_CLASS_MAPPING_99


# Configuration
DATA_DIR = Path(__file__).resolve().parents[3] / 'data'
DISPROT_DIR = DATA_DIR / 'disprot'
TEMP_DIR = DATA_DIR / 'temp'
EVAL_DIR = DATA_DIR / 'eval'

# Input Files
# Use 2025_12 for all data (sequences, labels, annotations)
DP_2025_CONSENSUS = DISPROT_DIR / 'DisProt release_2025_12-consensus-regions.fasta'
DP_2025_REGIONS   = DISPROT_DIR / 'DisProt release_2025_12.fasta'

CAID3_FILE = EVAL_DIR / 'caid3/CAID3 v3.fasta'
CACHE_FILE = DATA_DIR / 'uniprot_sequences_cache.txt'

# Output Configuration
OUTPUT_DIR = DATA_DIR / 'final_cleaned_dataset'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

CLUSTERS_FILE = TEMP_DIR / 'disprot_caid4_clusters'
# use mmseqs to handle thresholds
CLUSTER_THRESHOLD = 0.25

def ensure_dirs():
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

def load_disprot_dataset(consensus_path, regions_path, cache):
    """
    Parses consensus for UniProt IDs and regions for annotations.
    Note: Disorder labels will come from IDPO:0000002, not consensus.
    """
    print(f"Loading dataset from {consensus_path} and {regions_path}...")
    consensus_data = parse_consensus_file(str(consensus_path))
    region_data = parse_region_file(str(regions_path))
    
    sequences = {}
    missing_count = 0
    
    print(f"Fetching sequences for {len(consensus_data)} entries...")
    for i, (pid, data) in enumerate(consensus_data.items()):
        uid = data['uniprot_id']
        if not uid:
            continue
            
        if uid in cache:
            sequences[pid] = cache[uid]
        else:
            seq = fetch_uniprot_sequence(uid)
            if seq:
                sequences[pid] = seq
                cache[uid] = seq
            else:
                missing_count += 1
        
        if i % 100 == 0:
            print(f"  Processed {i}/{len(consensus_data)}...", end='\r')
            
    print(f"\nLoaded {len(sequences)} sequences. Missing: {missing_count}")
    return sequences, region_data

def run_mmseqs(sequences, output_prefix, threshold):
    """Runs MMseqs2 to cluster sequences."""
    input_fasta = f"{output_prefix}.fasta"
    # mmseqs easy-cluster output suffix for clusters is _cluster.tsv
    cluster_tsv = f"{output_prefix}_cluster.tsv"
    tmp_dir = f"{output_prefix}_tmp"
    
    # Write temp fasta
    with open(input_fasta, 'w') as f:
        for pid, seq in sequences.items():
            f.write(f">{pid}\n{seq}\n")
            
    # mmseqs easy-cluster input output tmp --min-seq-id threshold
    cmd = [
        "mmseqs", "easy-cluster",
        input_fasta,
        output_prefix,
        tmp_dir,
        "--min-seq-id", str(threshold),
        "-c", "0.8", # Coverage 80%
        "--cov-mode", "0" # Bidirectional
    ]
    
    print(f"Running mmseqs: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        raise RuntimeError("mmseqs executable not found. Please install MMseqs2 (e.g. `apt install mmseqs2` or conda).")
    except subprocess.CalledProcessError as e:
        print(f"mmseqs failed: {e.stderr.decode()}")
        raise RuntimeError(f"mmseqs failed with exit code {e.returncode}")
            
    return cluster_tsv

def filter_caid_exact(query_seqs, target_seqs):
    """Filters out any sequences from query_seqs that exist exactly in target_seqs."""
    print("Filtering against CAID3 using exact sequence matching...")
    
    # Store just the raw, upper-cased sequences of target (CAID3)
    target_set = {seq.upper() for seq in target_seqs.values()}
    
    leaked_ids = set()
    for qid, qseq in query_seqs.items():
        if qseq.upper() in target_set:
            leaked_ids.add(qid)
            
    return leaked_ids

def parse_mmseqs_clusters(tsv_file):
    """
    Parses MMseqs2 cluster TSV file (representative <tab> member).
    Returns list of clusters, where each cluster is a list of protein IDs.
    """
    if not tsv_file or not os.path.exists(tsv_file):
        raise FileNotFoundError(f"Cluster file {tsv_file} not found.")

    clusters_map = defaultdict(list)
    with open(tsv_file, 'r') as f:
        for line in f:
            # rep_id \t member_id
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                rep, member = parts[0], parts[1]
                clusters_map[rep].append(member)
    
    # Sort for deterministic reproducibility (MMseqs2 output order is multi-threaded/random)
    for rep in clusters_map:
        clusters_map[rep].sort()
    sorted_reps = sorted(clusters_map.keys())
    
    return [clusters_map[rep] for rep in sorted_reps]


def parse_caid3(filepath):
    """Parses CAID3 fasta to get dict of sequences."""
    seqs = {}
    try:
        data = parse_fasta(str(filepath))
        seqs = data  # {id: sequence}
    except FileNotFoundError:
        print(f"Warning: CAID3 file not found at {filepath}")
    return seqs

# PDB_MISSING DATA UTILS
def load_and_filter_pdb_missing(pdb_missing_file, forbidden_seqs_set):
    """
    Load PDB_missing dataset from 9-line format file (pdb_missing_caid4.txt).
    Filter out any sequences already present in DisProt 2025 or CAID3.
    Returns list of (pid, sequence, block) where block is the 9 raw lines.
    """
    pdb_missing_file = Path(pdb_missing_file)
    if not pdb_missing_file.exists():
        print(f"Warning: PDB_missing file not found at {pdb_missing_file}")
        return []

    print(f"Loading PDB_missing from {pdb_missing_file}...")
    entries = []
    total = 0
    filtered = 0

    with open(pdb_missing_file) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if not lines[i].startswith(">"):
            i += 1
            continue
        block = lines[i : i + 9]
        if len(block) < 9:
            break
        pid      = block[0].rstrip()[1:]   # strip '>'
        sequence = block[1].rstrip()
        total   += 1
        if sequence not in forbidden_seqs_set:
            entries.append((pid, sequence, block))
        else:
            filtered += 1
        i += 9

    print(f"  Loaded {total:,} PDB_missing entries. "
          f"Filtered {filtered:,} matching DisProt/CAID3.")
    print(f"  Remaining PDB_missing pool: {len(entries):,}")
    return entries


# --------------------------------------------------------------------------------


def generate_entry_lines(pid, seq, binary_seq, functional_lbl=None):
    """Generates the 9 lines for a dataset entry."""
    # Functional masks/labels
    # If functional_lbl is None (e.g. PDB_missing), all are masks '-'
    # If functional_lbl provided, parse it
    
    if functional_lbl is None:
        mask_seq = '-' * len(seq)
        tracks = [mask_seq] * 6
    else:
        # Create strings
        prot_line = ['0'] * len(seq)
        nucl_line = ['0'] * len(seq)
        ion_line = ['0'] * len(seq)
        lipid_line = ['0'] * len(seq)
        link_line = ['0'] * len(seq)
        
        for pos, classes_str in enumerate(functional_lbl):
            if not classes_str: continue
            cls_list = classes_str.split(',')
            if 'Protein_binding' in cls_list: prot_line[pos] = '1'
            if 'Nucleic_acid_binding' in cls_list: nucl_line[pos] = '1'
            if 'Ion_binding' in cls_list: ion_line[pos] = '1'
            if 'Lipid_binding' in cls_list: lipid_line[pos] = '1'
            if 'Flexible_linker' in cls_list: link_line[pos] = '1'
        
        # Combined binding
        combined_binding = ['1' if (prot_line[k]=='1' or nucl_line[k]=='1' or ion_line[k]=='1' or lipid_line[k]=='1') else '0' for k in range(len(seq))]
        
        tracks = [
            "".join(prot_line),
            "".join(nucl_line),
            "".join(ion_line),
            "".join(lipid_line),
            "".join(combined_binding),
            "".join(link_line)
        ]
        
    lines = [f">{pid}\n", f"{seq}\n", f"{binary_seq}\n"]
    lines.extend([f"{t}\n" for t in tracks])
    return lines

def write_generic_dataset(name, pids, all_seqs, region_data, output_dir, func_map, pdb_missing_entries=None):
    """
    Write the unaltered dataset split for DisProt + PDB_missing entries.

    DisProt entries receive full label generation from IDPO annotations.
    PDB_missing entries are written verbatim (pre-built 9-line blocks).
    """
    f_unaltered = output_dir / f"{name}_unaltered_data.txt"

    # PDB_missing lookup: pid -> (sequence, block)
    pdb_missing_lookup = {}
    if pdb_missing_entries:
        pdb_missing_lookup = {pid: (seq, block) for pid, seq, block in pdb_missing_entries}

    print(f"[{name}] Writing {f_unaltered.name}...")

    with open(f_unaltered, 'w') as fu:
        count_dp          = 0
        count_pdb_missing = 0

        for i, pid in enumerate(sorted(pids)):
            if i % 100 == 0:
                print(f"  Processing {i+1}/{len(pids)}...", end='\r')

            # 1. PDB_missing Entry — write pre-built 9 lines verbatim
            if "|PDB_missing" in pid:
                entry = pdb_missing_lookup.get(pid)
                if not entry:
                    continue
                _, block = entry
                fu.write("".join(block))
                count_pdb_missing += 1

            # 2. DisProt Entry
            else:
                if pid not in all_seqs:
                    continue
                seq    = all_seqs[pid]
                regions = region_data.get(pid, [])

                binary_unaltered = create_disorder_labels_from_idpo(len(seq), regions)
                functional_lbl   = create_functional_labels(len(seq), regions, func_map)

                lines_u = generate_entry_lines(f"{pid}|DisProt", seq, binary_unaltered, functional_lbl)
                fu.write("".join(lines_u))
                count_dp += 1

    print(f"\n  Written {count_dp} DisProt + {count_pdb_missing} PDB_missing entries.")

def main():
    ensure_dirs()

    # 0. Load Cache
    cache = load_sequence_cache(CACHE_FILE)

    # 1. Load DisProt 2025_12
    print("--- Loading DisProt 2025_12 ---")
    if not DP_2025_CONSENSUS.exists() or not DP_2025_REGIONS.exists():
        print("Error: DisProt 2025_12 files not found")
        sys.exit(1)

    seqs_2025, reg_2025 = load_disprot_dataset(DP_2025_CONSENSUS, DP_2025_REGIONS, cache)
    save_sequence_cache(cache, CACHE_FILE)
    print(f"Loaded {len(seqs_2025)} proteins from 2025_12")

    # 2. Load CAID3 and filter matching sequences from DisProt 2025_12
    print("--- Loading CAID3 and filtering from DisProt 2025_12 ---")
    caid3_seqs_dict = parse_caid3(CAID3_FILE)
    caid3_seqs_set  = set(caid3_seqs_dict.values())
    print(f"CAID3: {len(caid3_seqs_dict)} sequences")

    MAX_LEN = 2000
    MIN_LEN = 11
    canonical = set('ACDEFGHIKLMNPQRSTVWY')

    seqs_pool = seqs_2025.copy()
    
    # Exact sequence match filtering against CAID3
    caid3_leaks = filter_caid_exact(seqs_pool, caid3_seqs_dict)
    seqs_pool = {pid: seq for pid, seq in seqs_pool.items() if pid not in caid3_leaks}
    
    removed_caid3 = len(seqs_2025) - len(seqs_pool)
    print(f"Removed {removed_caid3} DisProt proteins showing exact match with CAID3")

    len_before = len(seqs_pool)
    seqs_pool = {
        pid: seq for pid, seq in seqs_pool.items() 
        if len(seq) <= MAX_LEN
    }
    print(f"Removed {len_before - len(seqs_pool)} DisProt proteins (length > {MAX_LEN})")
    print(f"DisProt pool for clustering: {len(seqs_pool)} proteins")

    # 3. Load PDB_missing and filter against DisProt + CAID3
    pdb_missing_file    = DATA_DIR / 'final_cleaned_dataset' / 'pdb_missing_caid4.txt' # 'pdb_missing_notag.txt'
    pdb_missing_entries = []
    pdb_missing_seqs_dict = {}

    if pdb_missing_file.exists():
        # First filter out valid DisProt to avoid redundancies, and apply fast CAID3 exact match drops
        forbidden = set(seqs_2025.values()) | caid3_seqs_set
        print(f"Filtering PDB_missing against {len(forbidden):,} sequences (DisProt 2025 exact ∪ CAID3 exact)")
        pdb_missing_entries = load_and_filter_pdb_missing(pdb_missing_file, forbidden)
        
        # Exact sequence match filter against CAID3 leaks
        pdb_missing_dict = {pid: seq for pid, seq, _ in pdb_missing_entries}
        pdb_leaks = filter_caid_exact(pdb_missing_dict, caid3_seqs_dict)
        pdb_missing_entries = [(p, s, b) for p, s, b in pdb_missing_entries if p not in pdb_leaks]
        print(f"Removed {len(pdb_leaks)} PDB_missing entries showing exact match with CAID3")
        len_before_pdb = len(pdb_missing_entries)
        pdb_missing_entries = [
            (pid, seq, block) for pid, seq, block in pdb_missing_entries 
            if len(seq) <= MAX_LEN
        ]
        print(f"Removed {len_before_pdb - len(pdb_missing_entries):,} PDB_missing entries (length > {MAX_LEN})")
        pdb_missing_seqs_dict = {pid: seq for pid, seq, _ in pdb_missing_entries}
        print(f"Loaded {len(pdb_missing_seqs_dict):,} PDB_missing sequences (post-filtering)")
    else:
        print(f"Warning: PDB_missing file not found at {pdb_missing_file}. Proceeding without it.")

    # 4. Combine DisProt pool + PDB_missing for clustering
    master_seqs = seqs_pool.copy()
    master_seqs.update(pdb_missing_seqs_dict)

    print(f"--- Clustering master dataset ({len(master_seqs)} sequences) at k={CLUSTER_THRESHOLD} ---")
    clstr_file = run_mmseqs(master_seqs, str(CLUSTERS_FILE), threshold=CLUSTER_THRESHOLD)
    clusters   = parse_mmseqs_clusters(clstr_file)
    print(f"Generated {len(clusters)} clusters from {len(master_seqs)} sequences.")

    # 5. Split clusters 60 / 20 / 20
    random.seed(42)
    random.shuffle(clusters)

    total_clusters = len(clusters)
    n_train = int(total_clusters * 0.6)
    n_val   = int(total_clusters * 0.2)

    train_clusters = clusters[:n_train]
    val_clusters   = clusters[n_train:n_train + n_val]
    test_clusters  = clusters[n_train + n_val:]

    train_pids = [pid for cl in train_clusters for pid in cl]
    val_pids   = [pid for cl in val_clusters   for pid in cl]
    test_pids  = [pid for cl in test_clusters  for pid in cl]

    print(f"Split counts (60/20/20 clusters):")
    print(f"  Train: {len(train_pids)} seqs")
    print(f"  Val:   {len(val_pids)} seqs")
    print(f"  Test:  {len(test_pids)} seqs")

    # Filter test set to drop short and purely non-canonical sequences
    len_test_before = len(test_pids)
    test_pids = [
        pid for pid in test_pids 
        if MIN_LEN <= len(master_seqs[pid]) and any(c in canonical for c in master_seqs[pid].upper())
    ]
    print(f"\nRemoved {len_test_before - len(test_pids)} test sequences (length < {MIN_LEN} or purely non-canonical)")

    # Downsample Test Set to 1:1 Ratio
    test_pids_dp = [pid for pid in test_pids if "|PDB_missing" not in pid]
    test_pids_pdb = [pid for pid in test_pids if "|PDB_missing" in pid]
    
    target_pdb_count = len(test_pids_dp)
    if len(test_pids_pdb) > target_pdb_count:
        random.seed(42)
        test_pids_pdb.sort()  # Ensure deterministic sampling
        test_pids_pdb = random.sample(test_pids_pdb, target_pdb_count)
        
    test_pids = test_pids_dp + test_pids_pdb
    print(f"\nBalanced Test Set (1:1 Ratio -> {len(test_pids)} total):")
    print(f"  DisProt: {len(test_pids_dp)} seqs")
    print(f"  PDB_missing: {len(test_pids_pdb)} seqs")

    # Export ESM test FASTA
    test_fasta_path = OUTPUT_DIR / "test_clean.fasta"
    print(f"Exporting predictor-safe FASTA to {test_fasta_path}...")
    with open(test_fasta_path, 'w') as f:
        for pid in test_pids:
            seq = master_seqs[pid].upper()
            
            # Strip the tags pipeline headers
            clean_pid = pid.replace('|PDB_missing', '')
            f.write(f">{clean_pid}\n{seq}\n")

    # 6. Write datasets
    print("\n" + "=" * 80)
    print("WRITING OUTPUT FILES")
    print("=" * 80)

    write_generic_dataset("train_final_update_or_caid4", train_pids, master_seqs, reg_2025,
                          OUTPUT_DIR, FUNCTIONAL_CLASS_MAPPING_99, pdb_missing_entries=pdb_missing_entries)

    write_generic_dataset("val_final_update_or_caid4", val_pids, master_seqs, reg_2025,
                          OUTPUT_DIR, FUNCTIONAL_CLASS_MAPPING_99, pdb_missing_entries=pdb_missing_entries)

    write_generic_dataset("test_final_update_or_caid4", test_pids, master_seqs, reg_2025,
                          OUTPUT_DIR, FUNCTIONAL_CLASS_MAPPING_99, pdb_missing_entries=pdb_missing_entries)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nFinal dataset composition:")
    print(f"  Train: {len(train_pids):6,} proteins")
    print(f"  Val:   {len(val_pids):6,} proteins")
    print(f"  Test:  {len(test_pids):6,} proteins")
    print("\nAll splits contain DisProt + PDB_missing sequences (CAID3 exact-filtered).")

if __name__ == '__main__':
    main()