"""
Print the training config embedded in a checkpoint.

Usage:
    python -m src.cli.print_config checkpoints/phase2/binding/best_model.pt
    python -m src.cli.print_config checkpoints/phase1/best_model_final_pdb1.pt
"""

import pprint
import argparse
import torch
import yaml

def main():
    parser = argparse.ArgumentParser(description="Print config saved inside a checkpoint")
    parser.add_argument("checkpoint", help="Path to .pt checkpoint file")
    parser.add_argument(
        "--section",
        choices=["training_config", "model_config", "both"],
        default="training_config",
        help="Which config section to print (default: training_config)",
    )
    parser.add_argument(
        "--format",
        choices=["yaml", "pretty"],
        default="yaml",
        help="Output format (default: yaml)",
    )
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    sections = []
    if args.section in ("training_config", "both"):
        sections.append(("training_config", ckpt.get("training_config")))
    if args.section in ("model_config", "both"):
        sections.append(("model_config", ckpt.get("model_config")))

    for name, data in sections:
        if data is None:
            print(f"# [{name}] not found in checkpoint (saved before this feature was added)")
            continue
        if args.section == "both":
            print(f"# ── {name} ──")
        if args.format == "yaml":
            print(yaml.dump(data, default_flow_style=False, sort_keys=False), end="")
        else:
            pprint.pprint(data)
        if args.section == "both":
            print()

if __name__ == "__main__":
    main()