"""
ESM-C model loaders using the biohub/ESMC-* HuggingFace collection repos.

The collection repos (biohub/ESMC-300M, biohub/ESMC-600M, biohub/ESMC-6B)
store weights as safetensors in the Biohub transformers-fork checkpoint format,
which uses different sublayer key names than the esm package's ESMC class:

  Transformers-fork                    esm-package ESMC
  ─────────────────────────────────    ─────────────────────────────────
  attn.layernorm_qkv.layer_norm_weight attn.layernorm_qkv.0.weight
  attn.layernorm_qkv.layer_norm_bias   attn.layernorm_qkv.0.bias
  attn.layernorm_qkv.weight            attn.layernorm_qkv.1.weight
  ffn.layer_norm_weight                ffn.0.weight
  ffn.layer_norm_bias                  ffn.0.bias
  ffn.fc1_weight                       ffn.1.weight
  ffn.fc2_weight                       ffn.3.weight

The tensor shapes are identical — only key names differ.  This module downloads
the safetensors, remaps keys, and loads them into the esm package's ESMC class
(which provides the encode()/logits() SDK interface the rest of the codebase uses).

Importing this module registers all three builders into
esm.pretrained.LOCAL_MODEL_REGISTRY.
"""
import json
import re
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from esm.models.esmc import ESMC
from esm.tokenization import get_esmc_model_tokenizers
from esm.pretrained import register_local_model


_MODELS = {
    "esmc_300m": {"repo_id": "biohub/ESMC-300M", "d_model": 960, "n_heads": 15, "n_layers": 30},
    "esmc_600m": {"repo_id": "biohub/ESMC-600M", "d_model": 1152, "n_heads": 18, "n_layers": 36},
    "esmc_6b": {"repo_id": "biohub/ESMC-6B", "d_model": 2560, "n_heads": 40, "n_layers": 80},
}


def _remap_key(key: str) -> str | None:
    key = key.removeprefix("esmc.")
    if "_extra_state" in key or key.startswith("lm_head."):
        return None
    key = re.sub(r"attn\.layernorm_qkv\.layer_norm_weight$", "attn.layernorm_qkv.0.weight", key)
    key = re.sub(r"attn\.layernorm_qkv\.layer_norm_bias$", "attn.layernorm_qkv.0.bias", key)
    key = re.sub(r"attn\.layernorm_qkv\.weight$", "attn.layernorm_qkv.1.weight", key)
    key = re.sub(r"ffn\.layer_norm_weight$", "ffn.0.weight", key)
    key = re.sub(r"ffn\.layer_norm_bias$", "ffn.0.bias", key)
    key = re.sub(r"ffn\.fc1_weight$", "ffn.1.weight", key)
    key = re.sub(r"ffn\.fc2_weight$", "ffn.3.weight", key)
    return key


def _load_remapped_state_dict(snapshot_dir: Path) -> dict[str, torch.Tensor]:
    index_path = snapshot_dir / "model.safetensors.index.json"
    if index_path.is_file():
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
    else:
        shard_files = [p.name for p in snapshot_dir.glob("*.safetensors")]

    state_dict: dict[str, torch.Tensor] = {}
    for shard_file in shard_files:
        shard = load_file(str(snapshot_dir / shard_file))
        for k, v in shard.items():
            new_key = _remap_key(k)
            if new_key is not None and new_key not in state_dict:
                state_dict[new_key] = v
    return state_dict


def _make_builder(repo_id, d_model, n_heads, n_layers):
    def builder(device=torch.device("cpu"), use_flash_attn=True):
        model = ESMC(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            tokenizer=get_esmc_model_tokenizers(),
            use_flash_attn=use_flash_attn,
        ).eval()

        snapshot_dir = Path(snapshot_download(repo_id=repo_id))
        state_dict = _load_remapped_state_dict(snapshot_dir)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        expected_missing = {k for k in missing if k.startswith("sequence_head.")}
        if set(missing) - expected_missing:
            raise RuntimeError(
                f"Unexpected missing keys loading {repo_id}: "
                f"{set(missing) - expected_missing}"
            )

        model = model.to(device)
        return model
    return builder


for _name, _cfg in _MODELS.items():
    register_local_model(
        _name,
        _make_builder(_cfg["repo_id"], _cfg["d_model"], _cfg["n_heads"], _cfg["n_layers"]),
    )
