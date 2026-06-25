"""
Merge two separately-trained Phase 2 checkpoints into a single unified model.

Phase 2 trains binding and linker heads independently on top of a frozen Phase 1
disorder model.  Each head gets its own checkpoint.  This script:

  1. Rebuilds the Phase 1 architecture (backbone + disorder head) from the config.
  2. Creates a unified Phase 2 model with *both* heads enabled.
  3. Copies shared weights (phase1.*, disorder_proj*) from the binding checkpoint.
  4. Copies binding-specific weights (binding_*) from the binding checkpoint.
  5. Copies linker-specific weights (linker_*) from the linker checkpoint.
  6. Saves a single checkpoint containing the merged state dict, model config,
     and optimal thresholds from each head's training.

The merged checkpoint can then be loaded for inference (src.cli.predict) or
optional joint fine-tuning (configs/phase2/joint.yaml).

Train the two checkpoints first:
    python -m src.cli.train --config configs/phase2/binding.yaml
    python -m src.cli.train --config configs/phase2/linker.yaml

Then merge:
    python -m src.cli.merge_phase2 \\
        --binding_ckpt checkpoints/phase2/binding/best_model.pt \\
        --linker_ckpt  checkpoints/phase2/linker/best_model.pt  \\
        --output       checkpoints/phase2/unified.pt
"""

import argparse
import logging
import torch
from pathlib import Path
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
from src.models.backbone import create_backbone
from src.models import cascDP_Phase1, cascDP_Phase2

def build_phase1(model_config: dict, device: str) -> cascDP_Phase1:
    lora_config = model_config.get('lora', {})
    if 'backbone_name' not in model_config:
        raise ValueError("Checkpoint model_config must contain 'backbone_name'")

    backbone = create_backbone(
        backbone_type=model_config.get('backbone_type', 'esmc'),
        model_name=model_config['backbone_name'],
        use_lora=model_config.get('use_lora', False),
        lora_r=lora_config.get('r', 16),
        lora_alpha=lora_config.get('lora_alpha', 32),
        lora_dropout=lora_config.get('lora_dropout', 0.05),
        layers_to_transform=lora_config.get('layers_to_transform', None),
        target_modules=lora_config.get('target_modules', None),
        device=device,
    )

    phase1 = cascDP_Phase1(
        backbone=backbone,
        device=device,
        context_type=model_config.get('phase1_context_type', 'bigru'),
        dropout=model_config.get('dropout', 0.5),
        use_crf=model_config.get('use_crf', False),
    )
    return phase1

def main():
    parser = argparse.ArgumentParser(
        description="Merge Phase 2 binding-only + linker-only checkpoints into a unified model"
    )
    parser.add_argument("--binding_ckpt", required=True,
                        help="Path to binding-only Phase 2 checkpoint")
    parser.add_argument("--linker_ckpt",  required=True,
                        help="Path to linker-only Phase 2 checkpoint")
    parser.add_argument("--output",       required=True,
                        help="Output path for the merged checkpoint (.pt)")
    parser.add_argument("--device",       default="cpu",
                        help="Device to use for merging (default: cpu)")
    args = parser.parse_args()

    device = args.device

    # Load individual checkpoints to extract model config and thresholds
    binding_ckpt = torch.load(args.binding_ckpt, map_location=device, weights_only=False)
    linker_ckpt  = torch.load(args.linker_ckpt,  map_location=device, weights_only=False)

    # Extract model configs from checkpoints (saved during training)
    # Note: two heads may use different Phase 2 context modules
    binding_config = binding_ckpt['model_config']
    linker_config = linker_ckpt['model_config']
    logger.info("Model configs loaded from binding and linker checkpoints")

    _, binding_context_type, _ = cascDP_Phase2.resolve_context_types(binding_config)
    _, _, linker_context_type = cascDP_Phase2.resolve_context_types(linker_config)
    phase2_context_type = binding_context_type
    logger.info(
        "Merged Phase 2 contexts — binding: %s, linker: %s",
        binding_context_type,
        linker_context_type,
    )

    logger.info("Building Phase 1 model architecture…")
    phase1 = build_phase1(binding_config, device)

    logger.info("Merging binding + linker checkpoints…")
    model = cascDP_Phase2.from_separate_checkpoints(
        binding_ckpt_path=args.binding_ckpt,
        linker_ckpt_path=args.linker_ckpt,
        phase1_model=phase1,
        device=device,
        context_type=phase2_context_type,
        binding_context_type=binding_context_type,
        linker_context_type=linker_context_type,
        use_crf_linker=linker_config.get('use_crf_linker', False),
        binding_combined=binding_config.get('binding_combined', False),
        binding_head_type=binding_config.get('binding_head_type', 'cnn'),
    )

    # Build unified model config
    unified_config = binding_config.copy()
    unified_config['use_binding_head'] = True
    unified_config['use_linker_head'] = True
    unified_config['phase2_context_type'] = phase2_context_type
    unified_config['binding_context_type'] = binding_context_type
    unified_config['linker_context_type'] = linker_context_type
    unified_config['use_crf_linker'] = linker_config.get('use_crf_linker', False)
    unified_config['binding_combined'] = binding_config.get('binding_combined', False)
    unified_config['binding_head_type'] = binding_config.get('binding_head_type', 'cnn')
    unified_config['architecture_version'] = '2.3'

    # Extract optimal thresholds from each head's training
    best_threshold = binding_ckpt.get('best_threshold', 0.5)
    best_binding_threshold = binding_ckpt.get('best_binding_threshold', 0.5)
    best_linker_threshold = linker_ckpt.get('best_linker_threshold', 0.5)

    logger.info(f"Thresholds — disorder: {best_threshold}, binding: {best_binding_threshold}, linker: {best_linker_threshold}")

    merged_checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': unified_config,
        'best_threshold': best_threshold,
        'best_binding_threshold': best_binding_threshold,
        'best_linker_threshold': best_linker_threshold,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged_checkpoint, output_path)
    logger.info(f"Saved merged checkpoint -> {output_path}")

if __name__ == "__main__":
    main()