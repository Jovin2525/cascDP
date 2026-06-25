import torch
import torch.nn as nn

class BiGRUContext(nn.Module):
    def __init__(self, in_channels, hidden_channels=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_channels,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.norm = nn.LayerNorm(hidden_channels * 2)
        self.proj = nn.Linear(hidden_channels * 2, in_channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # GRU output: (Batch, Seq_Len, Hidden_Dim * 2)
        out, _ = self.gru(x)
        out = self.norm(out)
        out = self.proj(out)
        return self.dropout(out)


class ASPPBlock(nn.Module):
    # Captures both local details and regional context using dilated convolutions.
    def __init__(self, in_dim, out_dim, dropout=0.1, dilations=(2, 4, 8)):
        super().__init__()

        # Parallel branches with different receptive fields
        self.branch1 = nn.Conv1d(in_dim, out_dim, 1)                         # Local (1x1)
        self.branch2 = nn.Conv1d(in_dim, out_dim, 3, padding=dilations[0], dilation=dilations[0])
        self.branch3 = nn.Conv1d(in_dim, out_dim, 3, padding=dilations[1], dilation=dilations[1])
        self.branch4 = nn.Conv1d(in_dim, out_dim, 3, padding=dilations[2], dilation=dilations[2])

        self.fusion = nn.Conv1d(out_dim * 4, out_dim, 1)              # Fusion: out*4 -> out
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
            
        x_t = x.transpose(1, 2)
        
        b1 = self.branch1(x_t)
        b2 = self.branch2(x_t)
        b3 = self.branch3(x_t)
        b4 = self.branch4(x_t)
        
        # Concatenate and fuse
        out = self.fusion(torch.cat([b1, b2, b3, b4], dim=1))
        
        # Back to (Batch, Seq_Len, Channels)
        out = out.transpose(1, 2)
        
        out = self.norm(out)
        out = self.act(out)
        return self.dropout(out)

class BiLSTMContext(nn.Module):
    def __init__(self, in_channels, hidden_channels=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_channels,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.norm = nn.LayerNorm(hidden_channels * 2)
        # Project back to input dimension to match residual connection
        self.proj = nn.Linear(hidden_channels * 2, in_channels)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x):
        # LSTM output: (Batch, Seq_Len, Hidden_Dim * 2)
        out, _ = self.lstm(x)
        out = self.norm(out)
        out = self.proj(out)
        return self.dropout_layer(out)

class CNNHead(nn.Module):
    # Simple CNN-based prediction head.
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.5, dilation=1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.output_dim = output_dim
        
        current_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            # Use dilation only for the last conv layer
            is_last = (i == len(hidden_dims) - 1)
            dil = dilation if is_last else 1
            pad = dil  # padding = dilation for kernel_size=3 to maintain size
            self.layers.append(nn.Conv1d(current_dim, h_dim, kernel_size=3, padding=pad, dilation=dil))
            self.norms.append(nn.LayerNorm(h_dim))
            self.dropouts.append(nn.Dropout(dropout))
            current_dim = h_dim
        
        self.final = nn.Linear(current_dim, output_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        # x: (Batch, Seq_Len, Dim)
        for conv, norm, drop in zip(self.layers, self.norms, self.dropouts):
            x_conv = x.transpose(1, 2)
            x_conv = conv(x_conv)
            x = x_conv.transpose(1, 2)
            x = norm(x)
            x = self.activation(x)
            x = drop(x)
        
        # Return raw logits (no activation) for BCEWithLogitsLoss
        x = self.final(x)
        return x.squeeze(-1) if self.output_dim == 1 else x