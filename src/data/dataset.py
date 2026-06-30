from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from .parsers import parse_dataset_file

try:
    import h5py
except ImportError:
    h5py = None

logger = logging.getLogger(__name__)

_EMBEDDING_SUFFIXES = (".pt", ".npy", ".h5", ".hdf5")

def _parse_fasta_sequences(file_path: Path) -> Dict[str, str]:
    sequences: Dict[str, str] = {}
    current_id: Optional[str] = None
    current_seq: List[str] = []
    annotation_chars = {"0", "1", "2", "-"}

    with open(file_path, "r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                # CAID evaluator inputs may interleave label rows in files that still
                # use a .fasta suffix; skip those annotation-only lines.
                if set(line).issubset(annotation_chars):
                    continue
                current_seq.append(line)

    if current_id is not None:
        sequences[current_id] = "".join(current_seq)

    return sequences

def _coerce_embeddings_tensor(embeddings: Any, protein_id: str) -> torch.Tensor:
    if isinstance(embeddings, dict):
        if "embeddings" not in embeddings:
            raise ValueError(f"{protein_id}: Expected key 'embeddings' in embedding payload")
        embeddings = embeddings["embeddings"]

    if isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(embeddings)
    elif not torch.is_tensor(embeddings):
        embeddings = torch.as_tensor(embeddings)

    embeddings = embeddings.detach().cpu().float()

    if embeddings.dim() == 3:
        if embeddings.shape[0] == 1:
            embeddings = embeddings.squeeze(0)
        else:
            raise ValueError(f"{protein_id}: Unexpected batch size {embeddings.shape[0]}, expected 1")
    elif embeddings.dim() != 2:
        raise ValueError(
            f"{protein_id}: Unexpected embedding dimensions {embeddings.dim()}, shape {tuple(embeddings.shape)}"
        )

    return embeddings

def _strip_special_tokens(embeddings: torch.Tensor, expected_len: int, protein_id: str) -> torch.Tensor:
    seq_len = embeddings.shape[0]

    if seq_len == expected_len + 2:
        return embeddings[1:-1]
    if seq_len != expected_len:
        raise ValueError(
            f"{protein_id}: Embedding length {seq_len} and expected length {expected_len} mismatch."
        )
    return embeddings

def _list_hdf5_embedding_ids(h5_path: Path) -> set[str]:
    if h5py is None:
        raise ImportError("h5py is required to read .h5 embeddings")

    ids: set[str] = set()
    with h5py.File(h5_path, "r") as handle:
        if "embeddings" in handle and isinstance(handle["embeddings"], h5py.Group):
            ids.update(handle["embeddings"].keys())

        for key in handle.keys():
            obj = handle[key]
            if isinstance(obj, h5py.Dataset):
                ids.add(key)
            elif isinstance(obj, h5py.Group) and any(name in obj for name in ("embeddings", "embedding")):
                ids.add(key)
    return ids

def _load_hdf5_embedding(h5_path: Path, protein_id: str) -> Any:
    if h5py is None:
        raise ImportError("h5py is required to read .h5 embeddings")

    with h5py.File(h5_path, "r") as handle:
        if protein_id in handle:
            node = handle[protein_id]
        elif "embeddings" in handle and protein_id in handle["embeddings"]:
            node = handle["embeddings"][protein_id]
        else:
            raise FileNotFoundError(f"{protein_id}: embedding not found in HDF5 container {h5_path}")

        if isinstance(node, h5py.Dataset):
            return node[()]
        for key in ("embeddings", "embedding"):
            if key in node:
                return node[key][()]

    raise ValueError(f"{protein_id}: Could not locate embedding dataset inside {h5_path}")

def _list_available_embedding_ids(embedding_source: Path) -> set[str]:
    if embedding_source.is_dir():
        ids: set[str] = set()
        for suffix in _EMBEDDING_SUFFIXES:
            ids.update(path.stem for path in embedding_source.glob(f"*{suffix}"))
        return ids

    if embedding_source.is_file() and embedding_source.suffix.lower() in {".h5", ".hdf5"}:
        return _list_hdf5_embedding_ids(embedding_source)

    raise FileNotFoundError(
        f"Embedding source must be a directory of per-protein files or a single .h5/.hdf5 file: {embedding_source}"
    )

def _load_embedding_by_id(embedding_source: Path, protein_id: str) -> torch.Tensor:
    if embedding_source.is_dir():
        for suffix in (".pt", ".npy", ".h5", ".hdf5"):
            emb_path = embedding_source / f"{protein_id}{suffix}"
            if not emb_path.exists():
                continue
            if suffix == ".pt":
                payload = torch.load(emb_path, map_location="cpu", weights_only=False)
                return _coerce_embeddings_tensor(payload, protein_id)
            if suffix == ".npy":
                return _coerce_embeddings_tensor(np.load(emb_path, allow_pickle=False), protein_id)
            return _coerce_embeddings_tensor(_load_hdf5_embedding(emb_path, protein_id), protein_id)

        raise FileNotFoundError(f"{protein_id}: no embedding file found under {embedding_source}")

    if embedding_source.is_file() and embedding_source.suffix.lower() in {".h5", ".hdf5"}:
        return _coerce_embeddings_tensor(_load_hdf5_embedding(embedding_source, protein_id), protein_id)

    raise FileNotFoundError(f"Unsupported embedding source for {protein_id}: {embedding_source}")

def _empty_function_labels(seq_len: int) -> Dict[str, np.ndarray]:
    return {
        "function_labels": np.zeros((seq_len, 6), dtype=np.float32),
        "binding_mask": np.zeros(seq_len, dtype=np.float32),
        "linker_mask": np.zeros(seq_len, dtype=np.float32),
        "binding_type_masks": np.zeros((seq_len, 4), dtype=np.float32),
    }

class DisorderFunctionDataset(Dataset):
    """
    Dataset for cascaded disorder and function prediction.
    
    Loads pre-computed embeddings and labels from dataset files.
    
    Args:
        embedding_dir: Directory of per-protein embeddings or a single .h5/.hdf5 file
        disorder_file: Path to dataset file (sequence + labels)
        protein_ids: Optional list of protein IDs to load (if None, loads all)
    """
    
    def __init__(
        self,
        embedding_dir: str,
        disorder_file: str,
        protein_ids: Optional[List[str]] = None,
        pdb_loss_weight: float = 1.0  # Weight for PDB_missing samples
    ):
        self.embedding_dir = Path(embedding_dir)
        self.disorder_file = Path(disorder_file)
        self.pdb_loss_weight = pdb_loss_weight
        
        # Load labels using shared parser
        self.sequences, self.disorder_labels, (self.function_labels, self.binding_masks, self.linker_masks, self.binding_type_masks), self.protein_sources = parse_dataset_file(self.disorder_file)
        
        # Identify valid label IDs (must have disorder)
        valid_label_ids = set(self.disorder_labels.keys())

        # Fill missing function data (e.g. for CAID3 single-track files)
        for pid in valid_label_ids:
            if pid not in self.function_labels:
                seq_len = len(self.disorder_labels[pid][0])
                self.function_labels[pid] = np.zeros((seq_len, 6), dtype=np.float32)
                self.binding_masks[pid] = np.zeros(seq_len, dtype=np.float32)
                self.linker_masks[pid] = np.zeros(seq_len, dtype=np.float32)
                self.binding_type_masks[pid] = np.zeros((seq_len, 4), dtype=np.float32)

        # Filter by protein_ids if provided
        if protein_ids is not None:
            self.protein_ids = protein_ids
            logger.info(f"Using {len(protein_ids)} specified protein IDs")
        else:
            # Use intersection of proteins with embeddings and labels
            available_embeddings = _list_available_embedding_ids(self.embedding_dir)
            self.protein_ids = sorted(list(valid_label_ids & available_embeddings))
            logger.info(f"Found {len(self.protein_ids)} proteins with embeddings and labels")
        
        logger.info(f"Dataset initialized with {len(self.protein_ids)} proteins")
    
    def __len__(self) -> int:
        return len(self.protein_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary containing:
                - protein_id: str
                - embeddings: (seq_len, hidden_dim)
                - disorder_labels: (seq_len,)
                - function_labels: (seq_len, 7)
                - mask: (seq_len,) - 1 for valid positions, 0 for padding
        """
        protein_id = self.protein_ids[idx]
        embeddings = _load_embedding_by_id(self.embedding_dir, protein_id)
        
        # Load labels (disorder returns tuple of (labels, mask))
        disorder_labels, disorder_mask = self.disorder_labels[protein_id]
        disorder = torch.from_numpy(disorder_labels)
        disorder_mask_tensor = torch.from_numpy(disorder_mask)
        
        function = torch.from_numpy(self.function_labels[protein_id])
        binding_mask = torch.from_numpy(self.binding_masks[protein_id])
        linker_mask = torch.from_numpy(self.linker_masks[protein_id])
        binding_mask_indiv = torch.from_numpy(self.binding_type_masks[protein_id])
        
        label_len = len(disorder)
        embeddings = _strip_special_tokens(embeddings, label_len, protein_id)
        
        # Create combined mask: disorder_mask AND sequence mask
        mask = disorder_mask_tensor * torch.ones(embeddings.shape[0], dtype=torch.float32)

        # Determine sample loss weight based on source
        loss_weight = 1.0
        source = self.protein_sources.get(protein_id, 'DisProt')
        if 'PDB' in source:
             loss_weight = self.pdb_loss_weight
        
        item = {
            'protein_id': protein_id,
            'embeddings': embeddings,
            'disorder_labels': disorder,
            'function_labels': function,
            'binding_mask': binding_mask,
            'linker_mask': linker_mask,
            'binding_mask_indiv': binding_mask_indiv,
            'mask': mask,
            'loss_weight': torch.tensor(loss_weight, dtype=torch.float32)
        }

        sequence = self.sequences.get(protein_id)
        if sequence is not None:
            item['sequence'] = sequence

        return item


class SubmissionEmbeddingDataset(Dataset):
    """Prediction-only dataset for FASTA inputs with precomputed embeddings.

    The embedding source can be a directory of per-protein files or a single
    .h5/.hdf5 container keyed by protein ID.
    """

    def __init__(
        self,
        embedding_dir: str,
        fasta_file: str,
        protein_ids: Optional[List[str]] = None,
    ):
        self.embedding_dir = Path(embedding_dir)
        self.fasta_file = Path(fasta_file)
        self.sequences = _parse_fasta_sequences(self.fasta_file)

        if not self.sequences:
            raise ValueError(f"No sequences found in FASTA file: {self.fasta_file}")

        sequence_ids = set(self.sequences.keys())
        available_embeddings = _list_available_embedding_ids(self.embedding_dir)
        self.sequence_ids = sequence_ids
        self.available_embedding_ids = available_embeddings

        if protein_ids is not None:
            missing_sequences = sorted(set(protein_ids) - sequence_ids)
            if missing_sequences:
                raise KeyError(f"Protein IDs missing from FASTA: {missing_sequences[:5]}")
            self.protein_ids = [pid for pid in protein_ids if pid in available_embeddings]
        else:
            self.protein_ids = sorted(sequence_ids & available_embeddings)

        logger.info(
            "Submission dataset initialized with %d proteins (%d FASTA, %d embeddings)",
            len(self.protein_ids),
            len(sequence_ids),
            len(available_embeddings),
        )

    def __len__(self) -> int:
        return len(self.protein_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        protein_id = self.protein_ids[idx]
        sequence = self.sequences[protein_id]
        embeddings = _load_embedding_by_id(self.embedding_dir, protein_id)
        embeddings = _strip_special_tokens(embeddings, len(sequence), protein_id)

        empty_labels = _empty_function_labels(len(sequence))

        return {
            'protein_id': protein_id,
            'sequence': sequence,
            'embeddings': embeddings,
            'disorder_labels': torch.zeros(len(sequence), dtype=torch.float32),
            'function_labels': torch.from_numpy(empty_labels['function_labels']),
            'binding_mask': torch.from_numpy(empty_labels['binding_mask']),
            'linker_mask': torch.from_numpy(empty_labels['linker_mask']),
            'binding_mask_indiv': torch.from_numpy(empty_labels['binding_type_masks']),
            'mask': torch.ones(len(sequence), dtype=torch.float32),
            'loss_weight': torch.tensor(1.0, dtype=torch.float32),
        }


class OnTheFlyDisorderFunctionDataset(Dataset):
    """
    Dataset that generates embeddings on-the-fly during training.
    """
    
    def __init__(
        self,
        disorder_file: str,
        embedding_model,
        device: str = 'cuda',
        pdb_loss_weight: float = 1.0 
    ):
        self.disorder_file = Path(disorder_file)
        self.embedding_model = embedding_model
        self.device = device
        self.pdb_loss_weight = pdb_loss_weight      
        
        # Load sequences and labels using shared parser
        self.sequences, self.disorder_labels, (self.function_labels, self.binding_masks, self.linker_masks, self.binding_type_masks), self.protein_sources = parse_dataset_file(self.disorder_file)
        
        # Identify valid IDs (must have sequence and disorder)
        valid_ids = set(self.sequences.keys()) & set(self.disorder_labels.keys())
        
        # Fill missing function data (e.g. for CAID3 single-track files)
        for pid in valid_ids:
            if pid not in self.function_labels:
                seq_len = len(self.sequences[pid])
                self.function_labels[pid] = np.zeros((seq_len, 6), dtype=np.float32)
                self.binding_masks[pid] = np.zeros(seq_len, dtype=np.float32)
                self.linker_masks[pid] = np.zeros(seq_len, dtype=np.float32)
                self.binding_type_masks[pid] = np.zeros((seq_len, 4), dtype=np.float32)
        
        self.protein_ids = sorted(list(valid_ids))
        
        logger.info(f"OnTheFly Dataset initialized with {len(self.protein_ids)} proteins")
    
    def __len__(self) -> int:
        return len(self.protein_ids)
    

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Generate data for on-the-fly training.
        """
        protein_id = self.protein_ids[idx]
        sequence = self.sequences[protein_id]
        
        # Load labels
        disorder_labels, disorder_mask = self.disorder_labels[protein_id]
        disorder = torch.from_numpy(disorder_labels)
        disorder_mask_tensor = torch.from_numpy(disorder_mask)
        
        function = torch.from_numpy(self.function_labels[protein_id])
        binding_mask = torch.from_numpy(self.binding_masks[protein_id])
        linker_mask = torch.from_numpy(self.linker_masks[protein_id])
        binding_mask_indiv = torch.from_numpy(self.binding_type_masks[protein_id])
        
        # Create combined mask (disorder mask)
        mask = disorder_mask_tensor
        
        # Calculate loss weight
        loss_weight = 1.0
        if self.protein_sources:
             source = self.protein_sources.get(protein_id, 'DisProt')
             if 'PDB' in source:
                 loss_weight = self.pdb_loss_weight

        return {
            'protein_id': protein_id,
            'sequence': sequence,
            'embeddings': torch.zeros(len(disorder), 1), # Dummy embedding
            'disorder_labels': disorder,
            'function_labels': function,
            'binding_mask': binding_mask,
            'linker_mask': linker_mask,
            'binding_mask_indiv': binding_mask_indiv,
            'mask': mask,
            'loss_weight': loss_weight
        }

def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Collate function for DataLoader.
    
    Pads sequences to the same length within a batch.
    """
    # Find max sequence length in batch
    max_len = max(item['embeddings'].shape[0] for item in batch)
    hidden_dim = batch[0]['embeddings'].shape[1]
    num_functions = batch[0]['function_labels'].shape[1]
    batch_size = len(batch)
    
    # Check for sequences
    sequences = [item.get('sequence') for item in batch]
    has_sequences = all(s is not None for s in sequences)
    
    # Initialize padded tensors
    embeddings = torch.zeros(batch_size, max_len, hidden_dim)
    disorder_labels = torch.zeros(batch_size, max_len)
    function_labels = torch.zeros(batch_size, max_len, num_functions)
    binding_masks = torch.zeros(batch_size, max_len)
    linker_masks = torch.zeros(batch_size, max_len)
    binding_masks_indiv = torch.zeros(batch_size, max_len, 4)
    masks = torch.zeros(batch_size, max_len)
    protein_ids = []
    
    # NEW: Initialize loss weights tensor
    batch_loss_weights = torch.ones(batch_size, dtype=torch.float32)
    
    # Fill tensors
    for i, item in enumerate(batch):
        seq_len = item['embeddings'].shape[0]
        embeddings[i, :seq_len] = item['embeddings']
        disorder_labels[i, :seq_len] = item['disorder_labels']
        function_labels[i, :seq_len] = item['function_labels']
        binding_masks[i, :seq_len] = item['binding_mask']
        linker_masks[i, :seq_len] = item['linker_mask']
        if 'binding_mask_indiv' in item:
            binding_masks_indiv[i, :seq_len] = item['binding_mask_indiv']
        masks[i, :seq_len] = item['mask']
        # New: Collect weights
        if 'loss_weight' in item:
            batch_loss_weights[i] = item['loss_weight']
            
        protein_ids.append(item['protein_id'])
    
    result = {
        'protein_ids': protein_ids,
        'embeddings': embeddings,
        'disorder_labels': disorder_labels,
        'function_labels': function_labels,
        'binding_mask': binding_masks,
        'linker_mask': linker_masks,
        'binding_mask_indiv': binding_masks_indiv,
        'mask': masks,
        'loss_weight': batch_loss_weights # Return weights
    }
    
    if has_sequences:
        result['sequences'] = sequences
        
    return result
