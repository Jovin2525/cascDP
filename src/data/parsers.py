import numpy as np
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any

logger = logging.getLogger(__name__)

def parse_dataset_file(file_path: Path) -> Tuple[
    Dict[str, str], 
    Dict[str, Tuple[np.ndarray, np.ndarray]], 
    Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]],
    Dict[str, str]
]:
    """
    Parses a dataset file and returns all data components.
    
    Returns:
        sequences: Dict[str, str]
        disorder_labels: Dict[str, (labels, mask)]
        function_data: (labels_matrix, binding_mask, linker_mask, binding_type_masks)
            - binding_type_masks: Dict[str, np.ndarray (L, 4)] per-type valid masks
              (Protein, Nucleic, Ion, Lipid). 1 = annotated, 0 = unknown ('-').
        protein_sources: Dict[str, str]
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Parsing dataset file: {file_path}")

    sequences = {}
    disorder_labels = {}
    function_labels_map = {}
    binding_masks_map = {}
    linker_masks_map = {}
    binding_type_masks_map = {}  # per-type (Prot/Nucl/Ion/Lipid) validity masks
    protein_sources = {}  # Track protein source (DisProt/PUNCH2)
    
    current_id = None
    current_lines = []

    def process_block(pid, lines):
        if not lines:
            return
            
        # Standard Format:
        # Line 0: Sequence
        # Line 1: Disorder (IDR)
        # Line 2-7: Function Labels (6 lines)
        
        sequence = ""
        label_lines = []
        
        # Heuristic: Check first line
        first_line = lines[0]
        numeric_chars = set('012-')
        if not set(first_line).issubset(numeric_chars):
            sequence = first_line
            label_lines = lines[1:]
        else:
            # No sequence line found
            label_lines = lines

        if sequence:
            sequences[pid] = sequence
            
        if not label_lines:
            return

        seq_len = len(sequence) if sequence else len(label_lines[0])
        
        # --- 1. Disorder Labels (First Label Line) ---
        idr_line = label_lines[0]
        if len(idr_line) == seq_len:
            l_arr = []
            m_arr = []
            for c in idr_line:
                if c == '1':
                    l_arr.append(1.0)
                    m_arr.append(1.0)
                else: # '0' or others
                    l_arr.append(0.0)
                    if c == '0':
                         m_arr.append(1.0)
                    else:
                         m_arr.append(0.0)
            
            disorder_labels[pid] = (
                np.array(l_arr, dtype=np.float32),
                np.array(m_arr, dtype=np.float32)
            )
        
        # --- 2. Function Labels ---
        # Lines 1-7 are Function        
        func_input_lines = []
        
        if len(label_lines) > 1:
            func_input_lines = label_lines[1:]

        if func_input_lines:
            matrix = np.zeros((seq_len, 6), dtype=np.float32)
            binding_mask = np.zeros(seq_len, dtype=np.float32)
            linker_mask = np.zeros(seq_len, dtype=np.float32)
            
            # Collect masks for each binding type
            binding_type_masks = []
            
            # Index 0 (Prot) -> Matrix 0
            if len(func_input_lines) > 0 and len(func_input_lines[0]) == seq_len:
                mask = _fill_matrix_col(matrix, 0, func_input_lines[0])
                binding_type_masks.append(mask)
                
            # Index 1 (Nucl) -> Matrix 1
            if len(func_input_lines) > 1 and len(func_input_lines[1]) == seq_len:
                mask = _fill_matrix_col(matrix, 1, func_input_lines[1])
                binding_type_masks.append(mask)
            
            # Index 2 (Ion) -> Matrix 2
            if len(func_input_lines) > 2 and len(func_input_lines[2]) == seq_len:
                mask = _fill_matrix_col(matrix, 2, func_input_lines[2])
                binding_type_masks.append(mask)

            # Index 3 (Lipid) -> Matrix 3
            if len(func_input_lines) > 3 and len(func_input_lines[3]) == seq_len:
                mask = _fill_matrix_col(matrix, 3, func_input_lines[3])
                binding_type_masks.append(mask)
                
            # Index 4 (Combined) -> Matrix 4
            if len(func_input_lines) > 4 and len(func_input_lines[4]) == seq_len:
                mask = _fill_matrix_col(matrix, 4, func_input_lines[4])
                # Binding mask: position is annotated if ANY binding type is annotated
                if binding_type_masks:
                    binding_mask = np.maximum.reduce(binding_type_masks)
                else:
                    binding_mask = mask
                
            # Index 5 (Linker) -> Matrix 5
            if len(func_input_lines) > 5 and len(func_input_lines[5]) == seq_len:
                linker_mask = _fill_matrix_col(matrix, 5, func_input_lines[5])
                
            function_labels_map[pid] = matrix
            binding_masks_map[pid] = binding_mask
            linker_masks_map[pid] = linker_mask

            # Stack per-type masks into (L, 4); pad with zeros if a type was missing
            type_mask_stack = np.zeros((seq_len, 4), dtype=np.float32)
            for ti, m in enumerate(binding_type_masks[:4]):
                type_mask_stack[:, ti] = m
            binding_type_masks_map[pid] = type_mask_stack

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('>'):
                if current_id:
                    process_block(current_id, current_lines)
                # Parse protein ID and source
                header = line[1:].strip()
                if '|' in header:
                    current_id, source = header.rsplit('|', 1)
                    protein_sources[current_id] = source
                else:
                    current_id = header
                current_lines = []
            elif current_id:
                current_lines.append(line)
                
        if current_id and current_lines:
            process_block(current_id, current_lines)
            
    logger.info(f"Loaded {len(sequences)} sequences, {len(disorder_labels)} disorder labels, {len(function_labels_map)} function labels")
    if protein_sources:
        logger.info(f"Found {len(protein_sources)} proteins with source annotations")
    return sequences, disorder_labels, (function_labels_map, binding_masks_map, linker_masks_map, binding_type_masks_map), protein_sources

def _fill_matrix_col(matrix, col_idx, line_str):
    """Fill matrix column and return mask (1=annotated, 0=unknown)."""
    mask = np.ones(len(line_str), dtype=np.float32)
    for i, char in enumerate(line_str):
        if char == '1':
            matrix[i, col_idx] = 1.0
        elif char == '-':
            matrix[i, col_idx] = 0.0
            mask[i] = 0.0  # Mark as unknown
        else:  # '0'
            matrix[i, col_idx] = 0.0
    return mask
