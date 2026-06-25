import re
import time
import urllib.request
import urllib.error
import os
from collections import defaultdict

def parse_fasta(filename):
    sequences = {}
    current_id = None
    current_seq = []

    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.rstrip('\n')
                if line.startswith('>'):
                    # Save previous entry
                    if current_id is not None:
                        sequences[current_id] = ''.join(current_seq)

                    # Start new entry
                    current_id = line
                    current_seq = []
                else:
                    # Accumulate sequence lines
                    current_seq.append(line)

            # Save last entry
            if current_id is not None:
                sequences[current_id] = ''.join(current_seq)
    except FileNotFoundError:
        pass

    return sequences

def fetch_uniprot_sequence(uniprot_id, retry=3):
    # Check if it's an isoform (contains hyphen)
    if '-' in uniprot_id:
        return fetch_uniprot_isoform(uniprot_id, retry)
    else:
        return fetch_uniprot_canonical(uniprot_id, retry)

def fetch_uniprot_canonical(uniprot_id, retry=3):
    url = f'https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta'

    for attempt in range(retry):
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                data = response.read().decode('utf-8')
                lines = data.strip().split('\n')
                if len(lines) > 1:
                    sequence = ''.join(lines[1:])
                    return sequence
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
        except Exception as e:
            if attempt < retry - 1:
                time.sleep(1)
                continue

    return None

def fetch_uniprot_isoform(uniprot_id, retry=3):
    # Parse isoform: P48740-3 -> base=P48740, isoform=3
    match = re.match(r'([A-Z0-9]+)-(\d+)', uniprot_id)
    if not match:
        return None

    base_id = match.group(1)
    isoform_num = match.group(2)

    # Try the isoform-specific endpoint first
    url = f'https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta'

    for attempt in range(retry):
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                data = response.read().decode('utf-8')
                lines = data.strip().split('\n')

                # Check if this is the correct isoform
                if len(lines) > 1:
                    header = lines[0]
                    # UniProt isoform headers contain "Isoform X" or the isoform ID
                    if f'Isoform {isoform_num}' in header or uniprot_id in header:
                        sequence = ''.join(lines[1:])
                        return sequence
                    # If it's the canonical (wrong), try alternative method
                    elif 'Isoform' not in header:
                        # This returned canonical, need to try alternative
                        break

        except urllib.error.HTTPError:
            pass
        except Exception as e:
            if attempt < retry - 1:
                time.sleep(1)
                continue

    # Alternative: Try fetching all isoforms from the entry
    return fetch_isoform_from_entry(base_id, isoform_num, retry)

def fetch_isoform_from_entry(base_id, isoform_num, retry=3):
    """
    Fetch all isoforms from UniProt entry and extract the specific one.
    Uses the text format which includes all isoforms.
    """
    url = f'https://rest.uniprot.org/uniprotkb/{base_id}.txt'

    for attempt in range(retry):
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                data = response.read().decode('utf-8')

                # Parse the text format for isoform sequences
                # Look for: CC   -!- ALTERNATIVE PRODUCTS:
                # Then find the specific isoform

                in_alt_products = False
                current_isoform = None
                isoform_sequences = {}

                for line in data.split('\n'):
                    if 'ALTERNATIVE PRODUCTS:' in line:
                        in_alt_products = True
                    elif in_alt_products and line.startswith('CC       Name='):
                        # Extract isoform name
                        match = re.search(r'Name=([^;]+)', line)
                        if match:
                            current_isoform = match.group(1).strip()
                    elif in_alt_products and 'IsoId=' in line:
                        # Extract isoform ID
                        match = re.search(r'IsoId=([^;,]+)', line)
                        if match:
                            iso_id = match.group(1).strip()
                            if iso_id.endswith(f'-{isoform_num}'):
                                # Found the right isoform, but we need sequence
                                # Continue to SQ section
                                pass
                    elif line.startswith('SQ   SEQUENCE'):
                        # Start of canonical sequence
                        in_alt_products = False

                # If isoforms aren't in description, return canonical
                # The disorder annotation might be for canonical
                return fetch_uniprot_canonical(base_id, retry=1)

        except Exception as e:
            if attempt < retry - 1:
                time.sleep(1)
                continue

    return None

def load_sequence_cache(cache_file):
    return parse_fasta(cache_file) if os.path.exists(cache_file) else {}

def save_sequence_cache(cache, cache_file):
    with open(cache_file, 'w') as f:
        for uniprot_id, sequence in sorted(cache.items()):
            f.write(f'>{uniprot_id}\n')
            # Write in 80-character lines (standard FASTA)
            for i in range(0, len(sequence), 80):
                f.write(f'{sequence[i:i+80]}\n')

def parse_consensus_file(filename):
    fasta_data = parse_fasta(filename)
    data = {}

    for header, consensus in fasta_data.items():
        # Parse: >disprot|DP00003|full acc=P03265
        if not header.startswith('>disprot|'):
            continue

        parts = header.split('|')
        if len(parts) < 2:
            continue

        protein_id = parts[1]

        # Extract UniProt accession (including isoforms)
        match = re.search(r'acc=([A-Z0-9\-]+)', header)
        uniprot_id = match.group(1) if match else None

        data[protein_id] = {
            'uniprot_id': uniprot_id,
            'consensus': consensus
        }

    return data

def parse_region_file(filename):
    fasta_data = parse_fasta(filename)
    # Group regions by protein_id and position
    region_groups = defaultdict(lambda: defaultdict(lambda: {'terms': [], 'go_terms': [], 'idpo_terms': []}))

    for header, sequence in fasta_data.items():
        if not header.startswith('>disprot|'):
            continue

        parts = header.split()
        if len(parts) < 1:
            continue

        region_id = parts[0].split('|')[1]
        match = re.match(r'(DP\d+)', region_id)
        if not match:
            continue

        protein_id = match.group(1)
        
        # Extract position and term
        start_pos = end_pos = None
        term = None
        
        for part in parts:
            if part.startswith('pos='):
                pos_str = part.split('=')[1]
                start, end = pos_str.split('-')
                start_pos = int(start)
                end_pos = int(end)
            elif part.startswith('term='):
                term = part.split('=')[1]
        
        if start_pos and end_pos and term:
            # Use position as key to group terms for same region
            pos_key = (start_pos, end_pos)
            region_data = region_groups[protein_id][pos_key]
            
            # Initialize position data if first time
            if 'start' not in region_data:
                region_data['start'] = start_pos
                region_data['end'] = end_pos
                region_data['sequence'] = sequence
                region_data['region_id'] = region_id
            
            # Collect all terms
            region_data['terms'].append(term)
            
            # Separate GO and IDPO terms
            if term.startswith('GO:'):
                if term not in region_data['go_terms']:
                    region_data['go_terms'].append(term)
            elif term.startswith('IDPO:'):
                if term not in region_data['idpo_terms']:
                    region_data['idpo_terms'].append(term)
    
    # Convert to list format
    annotations = defaultdict(list)
    for protein_id, regions in region_groups.items():
        for pos_key, region_data in regions.items():
            annotations[protein_id].append(region_data)
    
    return annotations

def create_disorder_labels_from_idpo(seq_length, regions):
    # Create disorder labels from IDPO:0000002 annotations.
    labels = ['0'] * seq_length
    
    for region in regions:
        idpo_terms = region.get('idpo_terms', [])
        
        # Check if this region is marked as disorder (IDPO:0000002)
        if 'IDPO:0000002' in idpo_terms:
            start = region['start'] - 1  # Convert to 0-indexed
            end = region['end']
            
            for i in range(start, end):
                if 0 <= i < seq_length:
                    labels[i] = '1'
    
    return ''.join(labels)

def create_functional_labels(seq_length, regions, functional_mapping):
    # Creates functional class labels using both GO and IDPO terms.
    labels = [set() for _ in range(seq_length)]
    
    for region in regions:
        # Get both GO and IDPO terms from the region
        go_terms = region.get('go_terms', [])
        idpo_terms = region.get('idpo_terms', [])
        all_terms = go_terms + idpo_terms
        
        if not all_terms:
            continue
            
        start = region['start'] - 1  # Convert to 0-indexed
        end = region['end']
        
        # Map terms to functional classes
        for term in all_terms:
            if term in functional_mapping:
                func_class = functional_mapping[term]
                for i in range(start, end):
                    if 0 <= i < seq_length:
                        labels[i].add(func_class)
    
    # Convert sets to sorted comma-separated strings
    return [','.join(sorted(s)) if s else '' for s in labels]

def create_go_labels(seq_length, region_annotations):
    labels = [set() for _ in range(seq_length)]

    for region in region_annotations:
        # Get GO terms from the region
        go_terms = region.get('go_terms', [])
        
        if not go_terms:
            continue
            
        start = region.get('start', 0) - 1
        end = region.get('end', 0)

        for term in go_terms:
            for pos in range(start, end):
                if 0 <= pos < seq_length:
                    labels[pos].add(term)

    return [';'.join(sorted(s)) if s else '' for s in labels]