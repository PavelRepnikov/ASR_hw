import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention



class FeedForwardModule(nn.Module):
    def __init__(self, d_model, expansion_factor, dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model * expansion_factor)
        self.linear2 = nn.Linear(d_model * expansion_factor, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.norm(x)
        return x



class ConvolutionModule(nn.Module):
    def __init__(self, d_model, kernel_size, conv_expansion_factor, dropout):
        super(ConvolutionModule, self).__init__()
        self.conv = nn.Conv1d(d_model, d_model * conv_expansion_factor, kernel_size, padding=kernel_size//2)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        x = self.norm(x)
        return x


class ConformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_expansion_factor, conv_expansion_factor, kernel_size, dropout):
        super(ConformerBlock, self).__init__()
        self.attn = MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.ffn = FeedForwardModule(d_model, ff_expansion_factor, dropout)
        self.conv = ConvolutionModule(d_model, kernel_size, conv_expansion_factor, dropout)

    def forward(self, x):
        # Multihead attention
        x_attn, _ = self.attn(x, x, x)
        x = x + x_attn  # Skip connection

        # Feed-forward module
        x_ffn = self.ffn(x)
        x = x + x_ffn  # Skip connection

        # Convolution module
        x_conv = self.conv(x)
        x = x + x_conv  # Skip connection

        return x


class Conformer(nn.Module):
    def __init__(self, num_blocks, d_model, n_heads, kernel_size, ff_expansion_factor,
                 conv_expansion_factor, dropout, num_classes, n_tokens=None, max_length=256):
        super().__init__()

        # Spectrogram input to embedding via convolution
        self.embedding = nn.Conv2d(1, d_model, kernel_size=(3, 3), padding=1)  # assuming spectrogram input

        # Stack Conformer blocks
        self.blocks = nn.ModuleList(
            [ConformerBlock(d_model, n_heads, ff_expansion_factor, conv_expansion_factor, kernel_size, dropout)
             for _ in range(num_blocks)]
        )
        
        # Final classification layer
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, **batch):
        # Extract spectrogram from batch
        x = batch["spectrogram"]
        print(f"Initial shape of x (spectrogram): {x.shape}")
        
        # Convert spectrogram [B, 1, T, F] to [B, 1, F, T] for convolution
        x = x.permute(0, 1, 3, 2)  # Shape [B, 1, F, T]
        print(f"Shape after permuting (B, 1, F, T): {x.shape}")

        # Apply embedding layer (convolution)
        x = self.embedding(x)  # This will output [B, C, F', T']
        print(f"Shape after embedding (convolution): {x.shape}")

        # Instead of reshaping into 4D, permute to [B, T', F', C] 
        x = x.permute(0, 3, 2, 1)  # [B, T', F', C] => (B, T', C, F')
        print(f"Shape after permuting for attention: {x.shape}")

        # Reshape to [B, T', F' * C] before passing to attention
        x = x.view(x.shape[0], x.shape[1], -1)  # Shape: [B, T', F' * C]
        print(f"Shape after reshaping for attention: {x.shape}")

        # Now, pass the reshaped tensor to the attention layer (3D tensor expected)
        x_attn, _ = self.attn(x, x, x)  # MHA expects [B, T', C]
        print(f"Shape after attention: {x_attn.shape}")

        # Apply Conformer blocks
        for block in self.blocks:
            x_attn = block(x_attn)
            print(f"Shape after Conformer block: {x_attn.shape}")

        # Global Average Pooling (optional)
        x_attn = x_attn.mean(dim=-1)  # Mean across the feature axis
        print(f"Shape after Global Average Pooling: {x_attn.shape}")

        # Final output layer
        x_out = self.fc(x_attn)
        print(f"Shape after final fc layer: {x_out.shape}")
        
        return x_out






    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
