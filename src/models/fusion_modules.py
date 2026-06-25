import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention mechanism for disorder-to-sequence fusion.
    
    Uses disorder features as Key/Value and sequence embeddings as Query
    This allows the model to selectively attend to relevant disorder patterns for binding and linker prediction at each residue position
    """
    def __init__(self, sequence_dim: int, disorder_dim: int, num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = sequence_dim // num_heads
        assert self.head_dim * num_heads == sequence_dim, "sequence_dim must be divisible by num_heads"
        
        # Project disorder features to K, V
        self.disorder_to_kv = nn.Linear(disorder_dim, sequence_dim * 2)
        
        # Project sequence embeddings to Q
        self.sequence_to_q = nn.Linear(sequence_dim, sequence_dim)
        
        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(sequence_dim, sequence_dim),
            nn.LayerNorm(sequence_dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, sequence_embeddings: torch.Tensor, disorder_features: torch.Tensor, need_weights: bool = False):
        """
        Apply cross-attention from sequence to disorder features.

        Returns:
            If need_weights=False: Disorder-aware features (Batch, Seq_Len, sequence_dim)
            If need_weights=True: (Disorder-aware features, Attention weights (Batch, Heads, Seq_Len, Seq_Len))
        """
        batch_size, seq_len, _ = sequence_embeddings.shape
        
        # Generate Q from sequence embeddings
        Q = self.sequence_to_q(sequence_embeddings)  # (Batch, Seq_Len, sequence_dim)
        
        # Generate K, V from disorder features
        kv = self.disorder_to_kv(disorder_features)  # (Batch, Seq_Len, sequence_dim * 2)
        K, V = kv.chunk(2, dim=-1)  # Each: (Batch, Seq_Len, sequence_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (Batch, Heads, Seq_Len, Head_Dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        dropout_p = self.dropout.p if self.training else 0.0
        
        if need_weights:
            # Scaled dot-product attention to extract weights
            attn_weight = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
            attn_weight = torch.nn.functional.softmax(attn_weight, dim=-1)
            
            if dropout_p > 0.0:
                attn_weight = torch.nn.functional.dropout(attn_weight, p=dropout_p)
                
            attn_output = torch.matmul(attn_weight, V)
        else:
            # Use PyTorch's memory-efficient scaled_dot_product_attention (Flash Attention 2 backend if available)
            # Avoids materializing full attention matrix and saves massive memory
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=False
            )  # (Batch, Heads, Seq_Len, Head_Dim)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)  # (Batch, Seq_Len, sequence_dim)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        if need_weights:
            return output, attn_weight
            
        return output