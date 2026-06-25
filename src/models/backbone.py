import torch
import torch.nn as nn
from typing import List, Optional, Dict
from abc import ABC, abstractmethod
try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

def _infer_hidden_dim_from_model_name(model_name: str) -> int:
    lowered = model_name.lower()
    if "300m" in lowered:
        return 960
    if "600m" in lowered:
        return 1152
    if "6b" in lowered:
        return 2560
    raise ValueError(
        f"Cannot infer hidden dimension for model {model_name}. "
        "Please check model configuration."
    )

class ProteinBackbone(ABC):

    @abstractmethod
    def get_model(self) -> nn.Module:
        pass
    
    @abstractmethod
    def get_hidden_dim(self) -> int:
        # Return embedding dimension
        pass
    
    @abstractmethod
    def apply_lora(self, r: int, alpha: int, dropout: float) -> bool:
        """
        Apply LoRA adapters to model
        
        Returns:
            True if LoRA was successfully applied, False otherwise.
        """
        pass

class ESMCBackbone(ProteinBackbone):
    
    def __init__(self, model_name: str = "esmc_600m", device: str = "cuda"):
        import src.models.esmc_compat  # noqa: F401 — registers esmc_300m/600m/6b loaders
        from esm.models.esmc import ESMC
        
        self.model_name = model_name
        self.device = device
        
        print(f"Loading ESM-C backbone: {model_name}...")
        self.model = ESMC.from_pretrained(model_name)
        self.model.to(device)
        self.hidden_dim = self._infer_hidden_dim()
        print(f"Hidden dimension: {self.hidden_dim}")
    
    def _infer_hidden_dim(self) -> int:
        if hasattr(self.model, 'd_model'):
            return self.model.d_model
        elif hasattr(self.model, 'config') and hasattr(self.model.config, 'hidden_size'):
            return self.model.config.hidden_size
        else:
            return _infer_hidden_dim_from_model_name(self.model_name)
    
    def get_model(self) -> nn.Module:
        return self.model
    
    def get_hidden_dim(self) -> int:
        # Return embedding dimension
        return self.hidden_dim
    
    def apply_lora(self, r: int = 16, alpha: int = 32, dropout: float = 0.05, layers_to_transform: Optional[list] = None, target_modules: Optional[list] = None) -> bool:
        import logging
        _log = logging.getLogger(__name__)

        if not PEFT_AVAILABLE:
            _log.error("peft library not available. Cannot apply LoRA.")
            return False

        _log.info(f"Configuring LoRA (r={r}, alpha={alpha}, dropout={dropout})...")

        # Use config's target_modules if provided, else detect
        if target_modules is None:
            target_modules = self._find_lora_targets()
        if not target_modules:
            _log.error("No suitable linear layers found for LoRA.")
            return False
        _log.info(f"LoRA target modules: {target_modules}")

        # Configure LoRA
        peft_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias="none",
            task_type=None,
            layers_to_transform=layers_to_transform
        )

        # Apply LoRA — wraps Linear layers and marks only LoRA params as trainable
        try:
            self.model = get_peft_model(self.model, peft_config)
        except Exception as e:
            _log.error(f"get_peft_model failed: {e}")
            return False

        # Explicitly freeze base model params
        for name, param in self.model.named_parameters():
            if 'lora_' not in name:
                param.requires_grad = False

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        _log.info(f"LoRA applied: {trainable:,} / {total:,} backbone params trainable ({100*trainable/total:.2f}%)")
        
        self.lora_r = r
        self.lora_alpha = alpha
        self.lora_dropout = dropout
        self.lora_target_modules = target_modules
        
        return True
    
    def _find_lora_targets(self) -> List[str]:
        # Identify LoRA target modules for ESM-C
        linear_layer_names = set()
        full_layer_paths = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                full_layer_paths.append(name)
                layer_name = name.split('.')[-1]
                linear_layer_names.add(layer_name)
        
        # Check for ESM-C architecture
        has_transformer_blocks = any('transformer.blocks' in path for path in full_layer_paths)
        
        if not has_transformer_blocks:
            raise ValueError(
                f"Could not detect ESM-C architecture in model {self.model_name}. "
                f"Found layer suffixes: {sorted(linear_layer_names)}"
            )

        targets = []
        
        # Attention output projection
        if 'out_proj' in linear_layer_names:
            targets.append('attn.out_proj')
        
        ffn_layers = [l for l in full_layer_paths if 'ffn' in l]
        if any('ffn.1' in l for l in ffn_layers):
            targets.append('ffn.1')
        if any('ffn.3' in l for l in ffn_layers):
            targets.append('ffn.3')
        
        if targets:
            print(f"Detected ESM-C architecture. Using Option 2 (Attention + FFN).")
            print(f"Target modules: {targets}")
            return targets
        
        raise ValueError("Could not identify suitable LoRA targets in ESM-C model.")

class PrecomputedEmbeddingBackbone(ProteinBackbone):
    def __init__(self, model_name: str = "esmc_600m", hidden_dim: Optional[int] = None):
        self.model_name = model_name
        self.hidden_dim = hidden_dim or _infer_hidden_dim_from_model_name(model_name)
        self.model = nn.Identity()

    def get_model(self) -> nn.Module:
        return self.model

    def get_hidden_dim(self) -> int:
        return self.hidden_dim

    def apply_lora(self, r: int, alpha: int, dropout: float) -> bool:
        return False

def create_precomputed_backbone(
    model_name: str = "esmc_600m",
    hidden_dim: Optional[int] = None,
) -> ProteinBackbone:
    return PrecomputedEmbeddingBackbone(model_name=model_name, hidden_dim=hidden_dim)

def create_backbone(
    backbone_type: str = "esmc",
    model_name: str = "esmc_600m",
    device: str = "cuda",
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    layers_to_transform: Optional[list] = None,
    target_modules: Optional[list] = None
) -> ProteinBackbone:

    if backbone_type.lower() == "esmc":
        backbone = ESMCBackbone(model_name=model_name, device=device)
        if use_lora:
            success = backbone.apply_lora(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, layers_to_transform=layers_to_transform, target_modules=target_modules)
            if not success:
                print("Proceeding without LoRA.")
        return backbone
    else:
        raise NotImplementedError(
            f"Backbone type '{backbone_type}' not implemented. "
            f"Currently supported: 'esmc'"
        )