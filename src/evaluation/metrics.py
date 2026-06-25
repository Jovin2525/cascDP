import torch
import numpy as np
import logging
import math
from sklearn.metrics import (
    roc_auc_score, average_precision_score, matthews_corrcoef,
    precision_recall_curve, balanced_accuracy_score
)
from typing import Dict

logger = logging.getLogger(__name__)

class MetricsCalculator:
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.disorder_preds = []
        self.disorder_labels = []
        self.disorder_probs = []
        
        self.binding_preds = []
        self.binding_labels = []
        self.binding_probs = []
        
        # Individual binding types (Protein, Nucleic, Ion, Lipid)
        self.binding_probs_indiv = [[] for _ in range(4)]
        self.binding_labels_indiv = [[] for _ in range(4)]
        
        self.linker_preds = []
        self.linker_labels = []
        self.linker_probs = []

        # Per-protein storage for macro (per-target) averaging.
        # Each maps pid -> {'probs': list, 'labels': list}.
        self.per_protein_disorder = {}
        self.per_protein_binding = {}
        self.per_protein_linker = {}
        self.per_protein_binding_indiv = [{} for _ in range(4)]
    
    def update(
        self,
        disorder_logits: torch.Tensor,  # (batch, seq_len, 1)
        binding_logits: torch.Tensor,   # (batch, seq_len, 1) - Combined (can be None in Phase 1)
        linker_logits: torch.Tensor,    # (batch, seq_len, 1) (can be None in Phase 1)
        disorder_labels: torch.Tensor,  # (batch, seq_len)
        binding_labels: torch.Tensor,   # (batch, seq_len) - Combined
        linker_labels: torch.Tensor,    # (batch, seq_len)
        disorder_mask: torch.Tensor,    # (batch, seq_len)
        binding_mask: torch.Tensor = None,     # (batch, seq_len) - separate mask for binding
        linker_mask: torch.Tensor = None,      # (batch, seq_len) - separate mask for linker
        binding_logits_indiv: torch.Tensor = None, # (batch, seq_len, 5) - Individual
        binding_labels_indiv: torch.Tensor = None, # (batch, seq_len, 5) - Individual
        binding_mask_indiv: torch.Tensor = None,   # (batch, seq_len, 4) - per-type masks
        protein_ids = None,                        # list of pids; enables per-target macro
    ):
        """
        Accumulate predictions for batch.
        
        Applies sigmoid to logits and filters by task-specific masks.
        """
        # Convert to numpy, apply sigmoid
        if disorder_logits is not None:
            disorder_probs = torch.sigmoid(disorder_logits.squeeze(-1)).cpu().numpy()
        else:
            disorder_probs = None
        
        # Handle None logits (Phase 1)
        if binding_logits is not None:
            binding_probs = torch.sigmoid(binding_logits.squeeze(-1)).cpu().numpy()
        else:
            binding_probs = None
        
        if linker_logits is not None:
            linker_probs = torch.sigmoid(linker_logits.squeeze(-1)).cpu().numpy()
        else:
            linker_probs = None
        
        disorder_labels_np = disorder_labels.cpu().numpy() if disorder_labels is not None else None
        binding_labels_np = binding_labels.cpu().numpy() if binding_labels is not None else None
        linker_labels_np = linker_labels.cpu().numpy() if linker_labels is not None else None
        disorder_mask_np = disorder_mask.cpu().numpy() if disorder_mask is not None else None
        binding_mask_np = binding_mask.cpu().numpy() if binding_mask is not None else None
        linker_mask_np = linker_mask.cpu().numpy() if linker_mask is not None else None
        
        # Individual binding types
        binding_probs_indiv_np = None
        binding_labels_indiv_np = None
        if binding_logits_indiv is not None and binding_labels_indiv is not None:
             binding_probs_indiv_np = torch.sigmoid(binding_logits_indiv).detach().cpu().numpy() # (N, L, C)
             binding_labels_indiv_np = binding_labels_indiv.cpu().numpy() # (N, L, C)

        binding_mask_indiv_np = (
            binding_mask_indiv.cpu().numpy() if binding_mask_indiv is not None else None
        )

        batch_size = None
        for candidate in (disorder_labels_np, binding_labels_np, linker_labels_np):
            if candidate is not None:
                batch_size = candidate.shape[0]
                break
        if batch_size is None:
            return

        # Flatten and filter by task-specific masks
        for i in range(batch_size):
            pid = protein_ids[i] if protein_ids is not None else None

            # Disorder - use disorder mask
            if disorder_probs is not None and disorder_labels_np is not None and disorder_mask_np is not None:
                disorder_valid = disorder_mask_np[i] == 1
                d_probs_i = disorder_probs[i][disorder_valid]
                d_labels_i = disorder_labels_np[i][disorder_valid]
                self.disorder_probs.extend(d_probs_i)
                self.disorder_labels.extend(d_labels_i)
                if pid is not None and len(d_probs_i) > 0:
                    bucket = self.per_protein_disorder.setdefault(pid, {'probs': [], 'labels': []})
                    bucket['probs'].extend(d_probs_i)
                    bucket['labels'].extend(d_labels_i)

            # Binding - use binding mask (skip if None)
            # CAID3 convention: only include proteins with >=1 positive binding
            # residue in the masked region. All-zero proteins (confirmed non-binding)
            # are excluded so they don't inflate AUC/APS with easy true negatives.
            if binding_probs is not None and binding_labels_np is not None and binding_mask_np is not None:
                binding_valid = binding_mask_np[i] == 1
                b_probs_i = binding_probs[i][binding_valid]
                b_labels_i = binding_labels_np[i][binding_valid]
                has_positive_binding = b_labels_i.sum() > 0
                if has_positive_binding:
                    self.binding_probs.extend(b_probs_i)
                    self.binding_labels.extend(b_labels_i)
                    if pid is not None and len(b_probs_i) > 0:
                        bucket = self.per_protein_binding.setdefault(pid, {'probs': [], 'labels': []})
                        bucket['probs'].extend(b_probs_i)
                        bucket['labels'].extend(b_labels_i)

                # Individual Binding Types -- use per-type mask when available
                # Apply same positive-only filter per type.
                if binding_probs_indiv_np is not None:
                    num_types = binding_probs_indiv_np.shape[-1]
                    if binding_mask_indiv_np is None:
                        continue
                    for type_idx in range(num_types):
                        if type_idx >= binding_mask_indiv_np.shape[-1]:
                            continue
                        type_valid = binding_mask_indiv_np[i, :, type_idx] == 1
                        t_probs_i = binding_probs_indiv_np[i, type_valid, type_idx]
                        t_labels_i = binding_labels_indiv_np[i, type_valid, type_idx]
                        if t_labels_i.sum() > 0:
                            self.binding_probs_indiv[type_idx].extend(t_probs_i)
                            self.binding_labels_indiv[type_idx].extend(t_labels_i)
                            if pid is not None and len(t_probs_i) > 0:
                                bucket = self.per_protein_binding_indiv[type_idx].setdefault(pid, {'probs': [], 'labels': []})
                                bucket['probs'].extend(t_probs_i)
                                bucket['labels'].extend(t_labels_i)

            # Linker - use linker mask (skip if None)
            # CAID3 convention: only include proteins with >=1 positive linker residue.
            if linker_probs is not None and linker_labels_np is not None and linker_mask_np is not None:
                linker_valid = linker_mask_np[i] == 1
                l_probs_i = linker_probs[i][linker_valid]
                l_labels_i = linker_labels_np[i][linker_valid]
                has_positive_linker = l_labels_i.sum() > 0
                if has_positive_linker:
                    self.linker_probs.extend(l_probs_i)
                    self.linker_labels.extend(l_labels_i)
                    if pid is not None and len(l_probs_i) > 0:
                        bucket = self.per_protein_linker.setdefault(pid, {'probs': [], 'labels': []})
                        bucket['probs'].extend(l_probs_i)
                        bucket['labels'].extend(l_labels_i)
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute all metrics using accumulated predictions.
        
        Returns:
            Dictionary with metrics for each task
        """
        metrics = {}
        
        # Convert to numpy arrays
        disorder_probs = np.array(self.disorder_probs)
        disorder_labels = np.array(self.disorder_labels)
        
        binding_probs = np.array(self.binding_probs)
        binding_labels = np.array(self.binding_labels)
        
        linker_probs = np.array(self.linker_probs)
        linker_labels = np.array(self.linker_labels)
        
        # Compute disorder metrics
        disorder_metrics = self._compute_binary_metrics(
            disorder_probs, disorder_labels, prefix="disorder"
        )
        metrics.update(disorder_metrics)
        
        # Compute binding metrics
        binding_metrics = self._compute_binary_metrics(
            binding_probs, binding_labels, prefix="binding"
        )
        metrics.update(binding_metrics)

        # Compute individual binding metrics
        binding_types = ['protein', 'nucleic_acid', 'ion', 'lipid']
        for i, b_type in enumerate(binding_types):
             probs = np.array(self.binding_probs_indiv[i])
             labels = np.array(self.binding_labels_indiv[i])
             
             # Prefix: binding_{b_type} (e.g. binding_protein)
             type_metrics = self._compute_binary_metrics(
                 probs, labels, prefix=f"binding_{b_type}"
             )
             metrics.update(type_metrics)
        
        # Compute linker metrics
        linker_metrics = self._compute_binary_metrics(
            linker_probs, linker_labels, prefix="linker"
        )
        metrics.update(linker_metrics)

        # ---- Per-target (macro) averaging — CAID3 style ----
        # CAID3 applies a single global optimal threshold (F_max threshold from the
        # dataset-level metrics), binarises each protein's predictions at that
        # threshold, then computes F1 / Precision / Recall / MCC / BAC per protein
        # and takes the mean across proteins.  Single-class proteins produce NaN for
        # some metrics (e.g. MCC/BAC when TN=0); NaN values are skipped by nanmean,
        # matching CAID3's implicit behaviour (ffill/bfill on a per-threshold matrix
        # effectively propagates valid values rather than including undefined ones).
        # Proteins with <2 valid residues are still skipped.
        binding_types = ['protein', 'nucleic_acid', 'ion', 'lipid']
        per_target_specs = [
            ('disorder', self.per_protein_disorder),
            ('binding',  self.per_protein_binding),
            ('linker',   self.per_protein_linker),
        ]
        for ti, b_type in enumerate(binding_types):
            per_target_specs.append((f'binding_{b_type}', self.per_protein_binding_indiv[ti]))

        keys_pt = ['f1', 'precision', 'recall', 'mcc', 'bac']
        for prefix, per_protein in per_target_specs:
            if not per_protein:
                continue
            # Get the global optimal threshold for this task.
            threshold = metrics.get(f'{prefix}_optimal_threshold', 0.5)
            if threshold is None or (isinstance(threshold, float) and math.isnan(threshold)):
                threshold = 0.5

            agg = {k: [] for k in keys_pt}
            n_proteins = 0
            for pid, bucket in per_protein.items():
                p_arr = np.asarray(bucket['probs'], dtype=np.float32)
                l_arr = np.asarray(bucket['labels'], dtype=np.float32)
                if len(l_arr) < 2:
                    continue
                n_proteins += 1
                preds = (p_arr >= threshold).astype(np.float32)
                tp = float(((preds == 1) & (l_arr == 1)).sum())
                fp = float(((preds == 1) & (l_arr == 0)).sum())
                fn = float(((preds == 0) & (l_arr == 1)).sum())
                tn = float(((preds == 0) & (l_arr == 0)).sum())

                # F1
                denom_f1 = 2 * tp + fp + fn
                f1 = (2 * tp / denom_f1) if denom_f1 > 0 else float('nan')
                agg['f1'].append(f1)

                # Precision
                denom_p = tp + fp
                agg['precision'].append((tp / denom_p) if denom_p > 0 else float('nan'))

                # Recall
                denom_r = tp + fn
                agg['recall'].append((tp / denom_r) if denom_r > 0 else float('nan'))

                # MCC — NaN when TN=FN=0 (all-positive) or TP=FP=0 (all-negative)
                denom_mcc = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                agg['mcc'].append(
                    ((tp * tn - fp * fn) / denom_mcc) if denom_mcc > 0 else float('nan')
                )

                # BAC — NaN when a class is absent
                tpr = (tp / (tp + fn)) if (tp + fn) > 0 else float('nan')
                tnr = (tn / (tn + fp)) if (tn + fp) > 0 else float('nan')
                bac = (
                    (tpr + tnr) / 2
                    if not (math.isnan(tpr) or math.isnan(tnr))
                    else float('nan')
                )
                agg['bac'].append(bac)

            if n_proteins > 0:
                for k in keys_pt:
                    vals = [v for v in agg[k] if not (isinstance(v, float) and math.isnan(v))]
                    metrics[f'{prefix}_{k}_per_target'] = (
                        float(np.mean(vals)) if vals else float('nan')
                    )
                metrics[f'{prefix}_n_targets'] = float(n_proteins)

        return metrics
    
    def _compute_binary_metrics(
        self, 
        probs: np.ndarray, 
        labels: np.ndarray, 
        prefix: str
    ) -> Dict[str, float]:
        """
        Compute metrics for a single binary task.
        
        Args:
            probs: Predicted probabilities (N,)
            labels: Ground truth labels (N,)
            prefix: Metric name prefix (e.g., 'disorder', 'binding')
        
        Returns:
            Dictionary with metrics
        """
        metrics = {}

        # Default NaN keys (returned when sample-set is empty or single-class).
        # CAID3 reports: AUC, APS, F_max + optimal threshold, precision/recall
        # at F_max, MCC at F_max, BAC at F_max.
        nan_keys = [
            f'{prefix}_auc', f'{prefix}_aps', f'{prefix}_f_max',
            f'{prefix}_optimal_threshold', f'{prefix}_precision',
            f'{prefix}_recall', f'{prefix}_mcc', f'{prefix}_bac',
        ]

        # Check if we have any samples at all (handle zero-mask case)
        if len(labels) == 0:
            return {k: float('nan') for k in nan_keys}

        # Check if we have any positive samples
        num_pos = np.sum(labels == 1)
        num_neg = np.sum(labels == 0)

        if num_pos == 0 or num_neg == 0:
            logger.warning(f"{prefix}: Only one class present, metrics may be undefined")
            zero = {k: 0.0 for k in nan_keys}
            zero[f'{prefix}_optimal_threshold'] = 0.5
            return zero

        # AUC and Average Precision Score
        try:
            metrics[f'{prefix}_auc'] = roc_auc_score(labels, probs)
            metrics[f'{prefix}_aps'] = average_precision_score(labels, probs)
        except Exception as e:
            logger.warning(f"{prefix} AUC/APS computation failed: {e}")
            metrics[f'{prefix}_auc'] = 0.0
            metrics[f'{prefix}_aps'] = 0.0

        # F_max: Find optimal threshold that maximizes F1
        precisions, recalls, thresholds = precision_recall_curve(labels, probs)

        # Compute F1 for each threshold
        f1_scores = []
        for p, r in zip(precisions[:-1], recalls[:-1]):
            if p + r > 0:
                f1 = 2 * (p * r) / (p + r)
            else:
                f1 = 0.0
            f1_scores.append(f1)

        # Find optimal threshold
        if len(f1_scores) > 0:
            best_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[best_idx]
            f_max = f1_scores[best_idx]
            optimal_precision = precisions[best_idx]
            optimal_recall = recalls[best_idx]
        else:
            optimal_threshold = 0.5
            f_max = 0.0
            optimal_precision = 0.0
            optimal_recall = 0.0

        metrics[f'{prefix}_f_max'] = f_max
        metrics[f'{prefix}_optimal_threshold'] = optimal_threshold
        metrics[f'{prefix}_precision'] = optimal_precision
        metrics[f'{prefix}_recall'] = optimal_recall

        # Threshold-dependent metrics at the F-max threshold (CAID3 convention).
        # Binarization uses '>=' to match sklearn semantics.
        preds = (probs >= optimal_threshold).astype(int)
        try:
            metrics[f'{prefix}_mcc'] = matthews_corrcoef(labels, preds)
        except Exception:
            metrics[f'{prefix}_mcc'] = 0.0
        try:
            metrics[f'{prefix}_bac'] = balanced_accuracy_score(labels, preds)
        except Exception:
            metrics[f'{prefix}_bac'] = 0.0

        return metrics
    
    def format_metrics(self, metrics: Dict[str, float]) -> str:
        """
        Format metrics dictionary into readable string.
        
        Args:
            metrics: Dictionary from compute_metrics()
        
        Returns:
            Formatted string
        """
        import math
        
        lines = []
        lines.append("=" * 60)
        lines.append("2-HEAD MODEL METRICS")
        lines.append("Binding: Combined from 4 types (Protein/Nucleic/Ion/Lipid)")
        lines.append("Linker: Flexible linker regions")
        lines.append("Pool: dataset-level (micro). _per_target = mean over proteins.")
        lines.append("=" * 60)

        # Helper to check if task has valid metrics
        def has_valid_metrics(prefix):
            f_max = metrics.get(f'{prefix}_f_max', float('nan'))
            return not math.isnan(f_max)

        def fmt(v, places=4):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return "  n/a "
            return f"{v:.{places}f}"

        def append_block(prefix, full=True):
            """Append a metric block for `prefix`. full=False prints a compact subset."""
            lines.append(f"  ROC-AUC:   {fmt(metrics.get(f'{prefix}_auc'))}")
            lines.append(f"  PR-AUC:    {fmt(metrics.get(f'{prefix}_aps'))}")
            lines.append(f"  F_max:     {fmt(metrics.get(f'{prefix}_f_max'))}")
            if full:
                lines.append(f"  Threshold: {fmt(metrics.get(f'{prefix}_optimal_threshold'))}")
                lines.append(f"  Precision: {fmt(metrics.get(f'{prefix}_precision'))}")
                lines.append(f"  Recall:    {fmt(metrics.get(f'{prefix}_recall'))}")
            lines.append(f"  MCC:       {fmt(metrics.get(f'{prefix}_mcc'))}")
            lines.append(f"  BAC:       {fmt(metrics.get(f'{prefix}_bac'))}")
            # Per-target (macro) averaged metrics — CAID3 style:
            # binary stats at global optimal threshold, nanmean across proteins.
            if not math.isnan(metrics.get(f'{prefix}_f1_per_target', float('nan'))):
                lines.append(f"  -- per-target avg ({metrics.get(f'{prefix}_n_targets', 0):.0f} proteins, thr={fmt(metrics.get(f'{prefix}_optimal_threshold'), 3)}) --")
                lines.append(f"  F1:        {fmt(metrics.get(f'{prefix}_f1_per_target'))}")
                lines.append(f"  Precision: {fmt(metrics.get(f'{prefix}_precision_per_target'))}")
                lines.append(f"  Recall:    {fmt(metrics.get(f'{prefix}_recall_per_target'))}")
                lines.append(f"  MCC:       {fmt(metrics.get(f'{prefix}_mcc_per_target'))}")
                lines.append(f"  BAC:       {fmt(metrics.get(f'{prefix}_bac_per_target'))}")

        # Disorder
        if has_valid_metrics('disorder'):
            lines.append("\nDISORDER PREDICTION:")
            lines.append("-" * 60)
            append_block('disorder')

            # Per-source breakdown (DisProt / PDB_missing)
            for sub_prefix, sub_label in [('disorder_disprot', 'DisProt'), ('disorder_pdb', 'PDB_missing')]:
                if has_valid_metrics(sub_prefix):
                    lines.append(f"\n  -- {sub_label} subset --")
                    append_block(sub_prefix, full=False)

        # Binding
        if has_valid_metrics('binding'):
            lines.append("\nBINDING PREDICTION (Combined: Protein/Nucleic/Ion/Lipid):")
            lines.append("-" * 60)
            append_block('binding')

            # Individual Binding Types
            binding_types = ['protein', 'nucleic_acid', 'ion', 'lipid']
            for b_type in binding_types:
                prefix = f"binding_{b_type}"
                if has_valid_metrics(prefix):
                    lines.append(f"\n  -- {b_type.upper()} BINDING --")
                    append_block(prefix, full=False)

        # Linker
        if has_valid_metrics('linker'):
            lines.append("\nLINKER PREDICTION:")
            lines.append("-" * 60)
            append_block('linker')

        lines.append("=" * 60)

        return "\n".join(lines)