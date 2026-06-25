import argparse
import yaml
from pathlib import Path
import torch
import numpy as np
import logging
from typing import Dict, List, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from src.models.backbone import create_backbone
from src.models.cascDP_phase1 import cascDP_Phase1
from src.models.cascDP_phase2 import cascDP_Phase2
from esm.sdk.api import ESMProtein, LogitsConfig

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_fasta(fasta_path: str) -> Dict[str, str]:
    sequences = {}
    current_id = None
    current_seq = []
    
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith('>'):
                if current_id is not None:
                    sequences[current_id] = ''.join(current_seq)
                current_id = line[1:].split()[0]  # Take first word after >
                current_seq = []
            else:
                current_seq.append(line)
        
        if current_id is not None:
            sequences[current_id] = ''.join(current_seq)
    
    return sequences

def load_model(checkpoint_path: str, device: str = 'cuda'):
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_config' not in checkpoint:
        raise ValueError("Checkpoint does not contain model_config")
    logging.info("Loading model config from checkpoint")
    model_cfg = checkpoint['model_config']

    backbone_type = model_cfg.get('backbone_type', 'esmc')
    if 'backbone_name' not in model_cfg:
        raise ValueError("Checkpoint model_config must contain 'backbone_name'")
    model_name = model_cfg['backbone_name']
    
    phase1_context_type = model_cfg.get('phase1_context_type', model_cfg.get('context_type'))
    if phase1_context_type is None:
        raise ValueError("Checkpoint model_config must contain 'phase1_context_type' or Phase 1 'context_type'")
    logging.info(f"Model config: Phase 1 context_type={phase1_context_type}")
    
    # Create model
    # Auto-detect LoRA from checkpoint keys
    checkpoint_keys = list(checkpoint.get('model_state_dict', {}).keys())
    has_lora_keys = any('lora_' in key or 'base_model' in key for key in checkpoint_keys)
    
    if has_lora_keys:
        # Try to get LoRA config from model_cfg
        lora_cfg = model_cfg.get('lora', {})
        lora_r = lora_cfg.get('r', 32)
        lora_alpha = lora_cfg.get('lora_alpha', 64)
        target_modules = lora_cfg.get('target_modules', ["attn.out_proj", "ffn.1", "ffn.3"])
        layers_to_transform = lora_cfg.get('layers_to_transform', None)
        
        logging.info(f"Detected LoRA in checkpoint - loading with rank={lora_r}, alpha={lora_alpha}, "
                     f"layers_to_transform={layers_to_transform}")
        use_lora = True
        lora_dropout = 0.0  # No dropout for eval
    else:
        logging.info("No LoRA detected in checkpoint - loading without LoRA")
        use_lora = False
        lora_r = None
        lora_alpha = None
        lora_dropout = None
        target_modules = None
        layers_to_transform = None
    
    use_binding_head = model_cfg.get('use_binding_head', True)
    logging.info(f"Using binding head: {use_binding_head}")

    backbone = create_backbone(
        backbone_type=backbone_type,
        model_name=model_name,
        device=device,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        layers_to_transform=layers_to_transform
    )
    
    # Detect model phase from checkpoint keys
    is_phase2 = any('binding_' in key or 'linker_' in key for key in checkpoint_keys)
    
    # Determine if CRF was used based on checkpoint keys
    if is_phase2:
        # For Phase 2 model, Phase 1 CRF keys would be under 'phase1.crf'
        use_crf = any('phase1.crf.transitions' in key for key in checkpoint_keys)
    else:
        # For Phase 1 model, keys are at root
        use_crf = any('crf.transitions' in key for key in checkpoint_keys)

    if use_crf:
        logging.info("Detected CRF in checkpoint (Phase 1)")

    if is_phase2:
        logging.info("Detected Phase 2 model (function heads)")
        use_crf_linker = model_cfg.get('use_crf_linker', False)
        if use_crf_linker:
            logging.info("Using Linker CRF from checkpoint config")
        
        # Create Phase 1 model first
        phase1_model = cascDP_Phase1(
            backbone=backbone,
            device=device,
            context_type=phase1_context_type,
            use_crf=use_crf,
        )
        
        if 'use_linker_head' not in model_cfg:
            raise ValueError("Phase 2 checkpoint model_config must contain 'use_linker_head'")
        use_linker_head = model_cfg['use_linker_head']

        logging.info(f"Using linker head: {use_linker_head}")

        # Create Phase 2 model
        phase2_context_type, binding_context_type, linker_context_type = (
            cascDP_Phase2.resolve_context_types(model_cfg)
        )
        
        model = cascDP_Phase2(
            phase1_model=phase1_model,
            device=device,
            context_type=phase2_context_type,
            binding_context_type=binding_context_type,
            linker_context_type=linker_context_type,
            use_binding_head=use_binding_head,
            use_linker_head=use_linker_head,
            use_crf_linker=use_crf_linker,
            binding_combined=model_cfg.get('binding_combined', False),
            binding_head_type=model_cfg.get('binding_head_type', 'cnn'),
        )
    else:
        logging.info("Detected Phase 1 model (disorder only)")
        model = cascDP_Phase1(
            backbone=backbone,
            device=device,
            context_type=phase1_context_type,
            use_crf=use_crf,
        )
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    
    def safe_get_bias(layer):
        if not hasattr(layer, 'bias') or layer.bias is None: return None
        if layer.bias.numel() == 1: return layer.bias.item()
        return layer.bias.detach().cpu().numpy()

    def format_bias_val(val):
        if val is None: return "None"
        if isinstance(val, (float, int)): return f"{val:.4f}"
        if isinstance(val, np.ndarray):
             return "[" + ", ".join([f"{v:.4f}" for v in val]) + "]"
        return str(val)

    if hasattr(model, 'phase1'):
        disorder_bias = safe_get_bias(model.phase1.disorder_initial)
        binding_bias = model.binding_output_layer.bias.detach().cpu().numpy() if getattr(model, 'binding_output_layer', None) is not None and model.binding_output_layer.bias is not None else None
        linker_bias = None
        if hasattr(model, 'linker_head') and hasattr(model.linker_head, 'final'):
             linker_bias = safe_get_bias(model.linker_head.final)
    else:
        disorder_bias = safe_get_bias(model.disorder_initial)
        binding_bias = None
        linker_bias = None

    disorder_str = format_bias_val(disorder_bias)
    linker_str = format_bias_val(linker_bias)

    if binding_bias is not None:
        binding_str = format_bias_val(binding_bias)
        logging.info(f"Learned biases - Disorder: {disorder_str}, Binding: {binding_str}, Linker: {linker_str}")
    else:
        logging.info(f"Learned biases - Disorder: {disorder_str}, Linker: {linker_str}")
    
    epoch = checkpoint.get('epoch', 'N/A')
    logging.info(f"Loaded model from epoch {epoch}")
    if 'best_metric' in checkpoint:
        logging.info(f"Best metric used during training: {checkpoint['best_metric']}")
    
    # Load saved thresholds from checkpoint (set at best-val epoch, fixed across runs)
    saved_thresholds = {
        'disorder': checkpoint.get('best_threshold', None),
        'binding': checkpoint.get('best_binding_threshold', None),
        'linker': checkpoint.get('best_linker_threshold', None),
    }
    
    # Drop any that were not saved (None means fall back to default)
    optimal_thresholds = {k: v for k, v in saved_thresholds.items() if v is not None}
    if optimal_thresholds:
        logging.info(f"Loaded optimal thresholds from checkpoint: {optimal_thresholds}")
    else:
        logging.warning("Optimal thresholds not found in checkpoint. Falling back to default thresholds (0.5).")
        optimal_thresholds = {'disorder': 0.5, 'binding': 0.5, 'linker': 0.5}
    
    return model, optimal_thresholds


@torch.no_grad()
def predict_sequence(model: cascDP_Phase2, sequence: str, device: str, optimal_thresholds: dict = None):
    # Predict disorder, binding, and linker for a single sequence.
    if optimal_thresholds is None:
        optimal_thresholds = {'disorder': 0.5, 'binding': 0.5, 'linker': 0.5}

    sequences = [sequence]
    disorder_logits, binding_logits, linker_logits = model(sequences=sequences)
    
    # Disorder
    # Output shape: (batch, seq_len, 1) or (batch, seq_len) or (batch, seq_len, 2)
    
    # Handle CRF logits - convert (B, L, 2) to relative logits (B, L, 1)
    if disorder_logits.shape[-1] == 2:
         disorder_logits = disorder_logits[..., 1:2] - disorder_logits[..., 0:1]

    if disorder_logits.dim() == 3:
        disorder_logits = disorder_logits.squeeze(-1)
    
    disorder_probs = torch.sigmoid(disorder_logits)[0].cpu().numpy()
    
    # Binding
    binding_probs = None
    binding_pred = None
    if binding_logits is not None:
        binding_probs = torch.sigmoid(binding_logits)[0].cpu().numpy()  # (L,) or (L,4)
        bind_thresh = optimal_thresholds.get('binding', 0.5)
        if binding_probs.ndim == 1 or (binding_probs.ndim == 2 and binding_probs.shape[-1] == 1):
            # Combined single-output head
            bp = binding_probs.squeeze()
            binding_pred = (bp > float(bind_thresh)).astype(int).reshape(-1, 1)
        else:
            # Multi-label 4-output head
            thresh_arr = np.array(bind_thresh if isinstance(bind_thresh, list) else [float(bind_thresh)] * binding_probs.shape[-1])
            binding_pred = (binding_probs > thresh_arr).astype(int)
        
    # Linker
    linker_probs = None
    linker_pred = None
    
    if linker_logits is not None:
        if model.use_crf_linker:
            emissions = linker_logits
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=device)
            best_paths = model.linker_crf.decode(emissions, mask=mask)
            linker_pred = np.array(best_paths[0])
            linker_probs = torch.softmax(linker_logits, dim=-1)[0, :, 1].cpu().numpy()
        else:
             # linker_logits: (batch, seq_len, 1) or scalar
             if linker_logits.dim() == 3:
                 linker_logits = linker_logits.squeeze(-1)
             linker_probs = torch.sigmoid(linker_logits)[0].cpu().numpy()
             # Use optimal threshold for linker prediction
             linker_pred = (linker_probs > optimal_thresholds.get('linker', 0.5)).astype(int)

    return disorder_probs, binding_probs, linker_probs, linker_pred, binding_pred

def plot_single_protein(
    protein_id: str,
    sequence: str,
    disorder_probs: np.ndarray,
    output_path: Path,
    binding_probs: Optional[np.ndarray] = None,
    linker_probs: Optional[np.ndarray] = None,
    binding_pred: Optional[np.ndarray] = None,
    linker_pred: Optional[np.ndarray] = None,
    binding_types: List[str] = None,
    optimal_thresholds: dict = None
):
    if binding_types is None:
        binding_types = ['Protein', 'RNA', 'DNA', 'Ion']
    
    if optimal_thresholds is None:
        optimal_thresholds = {'disorder': 0.5, 'binding': 0.5, 'linker': 0.5}

    seq_len = len(sequence)
    x = np.arange(seq_len)
    
    tracks = [] # List of (Label, BinaryData, Color)
    
    # Disorder (Binary)
    disorder_thresh = optimal_thresholds.get('disorder', 0.5)
    disorder_binary = (disorder_probs > disorder_thresh).astype(int)
    tracks.append(("Disorder Prediction", disorder_binary, "black"))

    # Binding Types
    if binding_pred is not None:
        # Define colors for known types
        type_colors = {
            'protein': 'tab:blue',
            'dna': 'tab:green',
            'rna': 'tab:orange',
            'ion': 'tab:purple',
            'lipid': 'tab:red'
        }
        
        num_types = binding_pred.shape[1]
        for i in range(num_types):
            label = binding_types[i] if binding_types and i < len(binding_types) else f"Binding_{i}"
            # Normalize label for color lookup
            color = 'gray'
            for key, val in type_colors.items():
                if key in label.lower():
                    color = val
                    break
            
            tracks.append((f"{label} Prediction", binding_pred[:, i], color))

    # Linker
    if linker_pred is not None:
        tracks.append(("Linker Prediction", linker_pred, "tab:cyan"))

    n_tracks = len(tracks)
    bottom_height = max(2, n_tracks * 0.5)
    fig_height = 3 + bottom_height
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, fig_height), 
                                   gridspec_kw={'height_ratios': [3, bottom_height]}, sharex=True)
    
    # Subplot 1: Disorder Probability
    ax1.plot(x, disorder_probs, label='Disorder Probability', color='black', linewidth=1.5)
    ax1.fill_between(x, 0, disorder_probs, color='black', alpha=0.1)
    ax1.axhline(y=disorder_thresh, color='red', linestyle='--', linewidth=1, 
                label=f"Threshold: {disorder_thresh:.3f}")
    ax1.set_ylabel("Probability")
    ax1.set_title(f"Disorder Probability")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right')

    # Subplot 2: Binary Tracks
    yticks = []
    yticklabels = []
    
    # Iterate in reverse so tracks[0] is plotted at top (highest y)
    for i, (label, data, color) in enumerate(reversed(tracks)):
        # i moves from 0 (bottom) to n-1 (top)
        y_center = i
        
        # Calculate segments for broken_barh
        if data is not None and np.any(data):
            # Compute run-length encoding
             d = np.diff(np.concatenate(([0], data, [0])))
             starts = np.where(d == 1)[0]
             ends = np.where(d == -1)[0]
             xranges = [(s, e-s) for s, e in zip(starts, ends)]
             
             if xranges:
                 ax2.broken_barh(xranges, (y_center - 0.4, 0.8), facecolors=color)
        
        yticks.append(y_center)
        yticklabels.append(label)

    ax2.set_yticks(yticks)
    ax2.set_yticklabels(yticklabels)
    ax2.set_ylim(-0.5, n_tracks - 0.5)
    ax2.set_xlim(0, seq_len)
    ax2.set_xlabel("Residue Index")
    
    # Add grid lines between tracks
    ax2.set_yticks(np.arange(n_tracks) - 0.5, minor=True)
    ax2.grid(True, axis='y', which='minor', linestyle='-', alpha=0.5, color='gray')
    ax2.grid(True, axis='x', linestyle=':', alpha=0.6)
    ax2.tick_params(axis='y', which='minor', length=0) # Hide minor ticks

    fig.suptitle(f"Protein: {protein_id} (Length: {seq_len})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

def save_predictions(predictions: Dict, output_dir: Path, plot: bool = False, optimal_thresholds: dict = None):
    output_dir.mkdir(parents=True, exist_ok=True)

    if plot:
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        logging.info(f"Saving plots to {plots_dir}")

        binding_types = [t.replace('_binding', '').replace('_', ' ').title() for t in cascDP_Phase2.BINDING_TYPES]

        for pid, pred in predictions.items():
            plot_path = plots_dir / f"{pid}.png"
            plot_single_protein(
                protein_id=pid,
                sequence=pred['sequence'],
                disorder_probs=np.array(pred['disorder_probs']),
                output_path=plot_path,
                binding_probs=None, # Not used in plot_single_protein anymore given we pass preds
                linker_probs=None,  # Not used in plot_single_protein anymore given we pass preds
                binding_pred=np.array(pred['binding_pred']) if 'binding_pred' in pred else None,
                linker_pred=np.array(pred['linker_pred']) if 'linker_pred' in pred else None,
                binding_types=binding_types,
                optimal_thresholds=optimal_thresholds
            )

    import json
    output_file = output_dir / 'predictions.json'

    # Convert all numpy arrays to lists for JSON
    serializable_preds = {}
    for pid, pred in predictions.items():
        serializable_preds[pid] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                   for k, v in pred.items() if v is not None}

    with open(output_file, 'w') as f:
        json.dump(serializable_preds, f, indent=2)
    logging.info(f"Saved predictions to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Predict with cascDP Phase 2 model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to training configuration YAML file')
    parser.add_argument('--fasta', type=str, required=True,
                       help='Input FASTA file')
    parser.add_argument('--output-dir', type=str, default='results/predictions',
                       help='Output directory for predictions')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots for each prediction')
    parser.add_argument('--gaussian-sigma', type=float, default=0.0,
                       help='Gaussian smoothing sigma for disorder probabilities (post-prediction). '
                            '0 = disabled. ESMDisPred uses ~0.2.')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    args = parser.parse_args()
    
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    logging.info(f"Loading model from {args.checkpoint}...")
    model, optimal_thresholds = load_model(args.checkpoint, device)
    
    logger.info(f"Loading sequences from {args.fasta}")
    sequences = parse_fasta(args.fasta)
    logger.info(f"Found {len(sequences)} sequences")

    predictions = {}
    for pid, seq in sequences.items():
        logger.info(f"Predicting {pid} (length: {len(seq)})")
        
        disorder_probs, binding_probs, linker_probs, linker_pred, binding_pred = predict_sequence(model, seq, device, optimal_thresholds=optimal_thresholds)
        
        # Post-prediction Gaussian smoothing on disorder probabilities
        if args.gaussian_sigma > 0:
            disorder_probs = np.clip(gaussian_filter1d(disorder_probs, sigma=args.gaussian_sigma), 0.0, 1.0)
        
        # Use optimal threshold for disorder prediction binary mask
        disorder_threshold = optimal_thresholds.get('disorder', 0.5)
        
        pred_entry = {
            'sequence': seq,
            'disorder_probs': disorder_probs,
            'disorder_pred': (disorder_probs > disorder_threshold).astype(int),
        }
        
        if binding_probs is not None:
            pred_entry['binding_probs'] = binding_probs
            pred_entry['binding_pred'] = binding_pred
            
        if linker_probs is not None:
            pred_entry['linker_probs'] = linker_probs
            pred_entry['linker_pred'] = linker_pred
            logger.info(f"  - Max Linker Prob: {np.max(linker_probs):.4f} (Active: {np.sum(linker_pred)} residues)")
            
        predictions[pid] = pred_entry
    
    output_dir = Path(args.output_dir)
    save_predictions(predictions, output_dir, plot=args.plot, optimal_thresholds=optimal_thresholds)   
    logger.info("Prediction completed!")

if __name__ == '__main__':
    main()
