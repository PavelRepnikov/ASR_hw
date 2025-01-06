import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention




class FeedForwardModule(nn.Module):
    def __init__(self, d_model, expansion_factor, dropout):
        super().__init__()
        self.d_model = d_model
        self.expansion_factor = expansion_factor
        self.dropout = dropout

        self.layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * expansion_factor)
        self.fc2 = nn.Linear(d_model * expansion_factor, d_model)

    def forward(self, x):
        # Print the shape of x before applying LayerNorm
        # print(f"Input shape before LayerNorm: {x.shape}")

        # Ensure the last dimension is equal to d_model
        if x.shape[-1] != self.d_model:
            # print(f"Warning: Expected input with last dimension {self.d_model}, but got input with last dimension {x.shape[-1]}")

            # Apply padding or reshaping to fix the mismatch (if necessary)
            if x.shape[-1] < self.d_model:
                padding_size = self.d_model - x.shape[-1]
                x = F.pad(x, (0, padding_size))  # Pad the sequence dimension
            elif x.shape[-1] > self.d_model:
                # If the sequence is too long, truncate (or apply some other logic)
                x = x[:, :, :self.d_model]

            # print(f"Shape after reshaping/padding: {x.shape}")

        # Apply LayerNorm
        x = self.layer_norm(x)

        # Print the shape of x after applying LayerNorm
        # print(f"Shape after LayerNorm: {x.shape}")

        # Apply the first Linear layer (expanding the model size)
        x = self.fc1(x)

        # Print the shape after the first Linear layer
        # print(f"Shape after first Linear (fc1): {x.shape}")

        # Apply ReLU activation and Dropout
        x = nn.ReLU()(x)
        x = nn.Dropout(self.dropout)(x)

        # Print the shape after ReLU and Dropout
        # print(f"Shape after ReLU and Dropout: {x.shape}")

        # Apply the second Linear layer (reducing the model size back)
        x = self.fc2(x)

        # Apply Dropout again
        x = nn.Dropout(self.dropout)(x)


        return x










class ConvolutionModule(nn.Module):
    def __init__(self, d_model, kernel_size, conv_expansion_factor, dropout):
        super(ConvolutionModule, self).__init__()

        self.padding = (kernel_size - 1) // 2  # This ensures output size is close to input size
        
        # Depthwise convolution
        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,  # Input channels (d_model)
            out_channels=d_model,  # Output channels (d_model)
            kernel_size=kernel_size,
            groups=d_model,  # Depthwise convolution
            padding=self.padding
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Gate convolution (expanding channels)
        self.gate_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model * conv_expansion_factor,  # Expanded channels for gating mechanism
            kernel_size=kernel_size,
            groups=d_model,
            padding=self.padding
        )
        
        # 1x1 convolution to adjust the number of channels in gate to match x
        self.adjust_channels = nn.Conv1d(
            in_channels=d_model * conv_expansion_factor,
            out_channels=d_model,  # Reduce to match the number of channels in input (d_model)
            kernel_size=1
        )

    def forward(self, x):
        # print("input to forward module:", x.shape)
        # Ensure input has 3 dimensions [B, C, T] for Conv1d
        if x.dim() == 3:  # Already in the correct format [B, C, T]
            x = x.permute(0, 2, 1)
        else:
            raise ValueError(f"Input tensor must have 2 or 3 dimensions, got {x.dim()}")
        # Apply depthwise convolution
        x = self.depthwise_conv(x)
        x = self.dropout(x)
        
        # Apply the gating mechanism (with expanded channels)
        gate = self.gate_conv(x)
        
        # Adjust the channels of gate to match x
        gate = self.adjust_channels(gate)

        # Ensure the sizes match before gating
        if x.size(2) != gate.size(2):
            # If they don't match, trim the larger tensor
            min_size = min(x.size(2), gate.size(2))
            x = x[:, :, :min_size]
            gate = gate[:, :, :min_size]

        # Apply gating mechanism
        return x * torch.sigmoid(gate)


class ConformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_expansion_factor, conv_expansion_factor, kernel_size, dropout):
        super(ConformerBlock, self).__init__()
        
        self.ff_module_1 = FeedForwardModule(d_model, ff_expansion_factor, dropout)
        self.attention = nn.Sequential(
            nn.LayerNorm(d_model),
            MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True),
        )
        self.conv_module = ConvolutionModule(d_model, kernel_size, conv_expansion_factor, dropout)
        self.ff_module_2 = FeedForwardModule(d_model, ff_expansion_factor, dropout)
        self.norm = nn.LayerNorm(d_model)

        # Define Conv1d layer with the proper input/output channels
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, groups=d_model)

        # Ensure weights of Conv1d are moved to 'cuda:1'
        self.depthwise_conv = self.depthwise_conv.to('cuda:1')

        # Move all layers and parameters to 'cuda:1' during initialization
        self.to('cuda:1')

    def forward(self, x, mask=None):
        # Ensure the input tensor is moved to 'cuda:1'
        x = x.to('cuda:1')

        # Apply feed-forward module 1
        x = self.ff_module_1(x)

        # Apply multi-head attention
        attn_output, attn_weights = self.attention[1](self.attention[0](x), x, x, key_padding_mask=mask)

        # Apply convolution module
        x = self.conv_module(x)

        # Apply depthwise convolution
        x = self.depthwise_conv(x)
        x = x.permute(0, 2, 1)  # [batch, channels, seq_len] --> [batch, seq_len, channels]

        # Apply feed-forward module 2
        x = self.ff_module_2(x)

        # Apply layer normalization
        x = self.norm(x)

        # Return both the output tensor and attention weights
        return x, attn_weights






# class ConformerBlock(nn.Module):
#     def __init__(self, d_model, n_heads, ff_expansion_factor, conv_expansion_factor, kernel_size, dropout):
#         super(ConformerBlock, self).__init__()

#         # Feed-Forward Modules
#         self.ff_module_1 = FeedForwardModule(d_model, ff_expansion_factor, dropout)
#         self.ff_module_2 = FeedForwardModule(d_model, ff_expansion_factor, dropout)

#         # Multihead Attention with LayerNorm
#         self.attn_norm = nn.LayerNorm(d_model)
#         self.attn = MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)

#         # Convolution Module
#         self.conv_module = ConvolutionModule(d_model, kernel_size, conv_expansion_factor, dropout)

#         # Depthwise Convolution
#         self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, groups=d_model)
#         self.depthwise_conv = self.depthwise_conv.to('cuda:1')

#         # Final LayerNorm
#         self.norm = nn.LayerNorm(d_model)

#         # Move all layers to the correct device
#         self.to('cuda:1')

#     def forward(self, x, mask=None):
#         # Initial Input Debugging
#         print(f"Input Shape: {x.shape}, Mean: {x.mean().item():.6f}, Std: {x.std().item():.6f}")

#         # Feed-Forward Module 1
#         x = self.ff_module_1(x)
#         print(f"After FF Module 1 Shape: {x.shape}, Mean: {x.mean().item():.6f}, Std: {x.std().item():.6f}")

#         # Multihead Attention
#         x = self.attn_norm(x)
#         attn_output, attn_weights = self.attn(x, x, x, key_padding_mask=mask)
#         x = attn_output
#         print(f"After Attention Shape: {x.shape}, Mean: {x.mean().item():.6f}, Std: {x.std().item():.6f}")
#         print(f"Attention Weights Mean: {attn_weights.mean().item():.6f}, Std: {attn_weights.std().item():.6f}")

#         # Convolution Module
#         x = self.conv_module(x)
#         print(f"After Conv Module Shape: {x.shape}, Mean: {x.mean().item():.6f}, Std: {x.std().item():.6f}")

#         # Depthwise Convolution
#         x = x.permute(0, 2, 1)  # [batch, seq_len, features] -> [batch, features, seq_len]
#         x = self.depthwise_conv(x)
#         x = x.permute(0, 2, 1)  # Back to [batch, seq_len, features]
#         print(f"After Depthwise Conv Shape: {x.shape}, Mean: {x.mean().item():.6f}, Std: {x.std().item():.6f}")

#         # Feed-Forward Module 2
#         x = self.ff_module_2(x)
#         print(f"After FF Module 2 Shape: {x.shape}, Mean: {x.mean().item():.6f}, Std: {x.std().item():.6f}")

#         # Final LayerNorm
#         x = self.norm(x)
#         print(f"After Final Norm Shape: {x.shape}, Mean: {x.mean().item():.6f}, Std: {x.std().item():.6f}")

#         # Gradient Debugging
#         if self.training:
#             for name, param in self.named_parameters():
#                 if param.grad is None:
#                     print(f"Gradient missing for {name}")

#         # Return both the output tensor and attention weights
#         return x, attn_weights


















class Conformer(nn.Module):
    def __init__(self, num_blocks, d_model, n_heads, kernel_size, ff_expansion_factor,
                 conv_expansion_factor, dropout, num_classes, n_tokens=None, max_length=256):
        super().__init__()
        self.d_model = d_model
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # Set device
        
        # Ensure input layer outputs 64 channels
        self.input_layer = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),  # Output 64 channels
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
        ).to(self.device)

        # Downsampling layer (additional)
        self.downsampling_layer = nn.Conv2d(
            d_model, d_model, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
        ).to(self.device)

        # MaxPooling layer to ensure the sequence length doesn't exceed max_length (256)
        self.pooling_layer = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=(0, 0)).to(self.device)

        self.blocks = nn.ModuleList(
            [
                ConformerBlock(
                    d_model,
                    n_heads,
                    ff_expansion_factor,
                    conv_expansion_factor,
                    kernel_size,
                    dropout,
                )
                for _ in range(num_blocks)
            ]
        ).to(self.device)

        self.num_classes = n_tokens if n_tokens is not None else num_classes
        self.output_layer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.num_classes),
        ).to(self.device)

        self.max_length = max_length



    # def forward(self, **batch):
    #     x = batch["spectrogram"].to(self.device)
    #     if x.dim() != 4:
    #         raise ValueError(f"Expected input with 4 dimensions (B, C, F, T), got {x.dim()}")

    #     # Move input to the correct device
    #     x = x.to(self.device)

    #     # Pass through the input layer (conv2d -> batchnorm -> relu)
    #     # Input shape: [B, 1, 128, 918] -> after conv2d: [B, d_model, 128, 459] (downsampled along the time axis)
    #     x = self.input_layer(x)
    #     x = self.downsampling_layer(x)
    #     x = self.pooling_layer(x)

    #     # Flattening the output of convolutional layers
    #     x = x.view(x.size(0), -1, x.size(-1))  # Reshape to [batch, time, features]

    #     # Passing through the Conformer blocks
    #     attention_weights = []
    #     for block in self.blocks:
    #         x, attn = block(x)
    #         attention_weights.append(attn)

    #     # Output layer (classification)
    #     logits = self.output_layer(x)  # [B, T, num_classes]
        
    #     # Convert logits to log probabilities
    #     log_probs = F.log_softmax(logits, dim=-1)

    #     # Calculate the length of the predicted sequences (batch_size,)
    #     log_probs_length = torch.full((x.size(0),), x.size(1), dtype=torch.long).to(self.device)

    #     # Return both log_probs and attention weights
    #     return {
    #         "log_probs": log_probs,
    #         "attention_weights": attention_weights,
    #         "log_probs_length": log_probs_length
    #     }
    def forward(self, **batch):
        # Extract and validate the spectrogram input
        x = batch["spectrogram"].to(self.device)
        if x.dim() != 4:
            raise ValueError(f"Expected input with 4 dimensions (B, C, F, T), got {x.dim()}")

        # Debug: Log the shape of the input
        if x.isnan().any():
            raise ValueError("Input contains NaN values.")
        # print(f"Input Shape: {x.shape}")

        # Input Layer: conv2d -> batchnorm -> relu
        try:
            x = self.input_layer(x)
        except Exception as e:
            raise RuntimeError(f"Error in input_layer: {e}")
        # print(f"After Input Layer Shape: {x.shape}")

        # Downsampling
        try:
            x = self.downsampling_layer(x)
        except Exception as e:
            raise RuntimeError(f"Error in downsampling_layer: {e}")
        # print(f"After Downsampling Layer Shape: {x.shape}")

        # Pooling
        try:
            x = self.pooling_layer(x)
        except Exception as e:
            raise RuntimeError(f"Error in pooling_layer: {e}")
        # print(f"After Pooling Layer Shape: {x.shape}")

        # Reshape for Conformer blocks
        try:
            x = x.view(x.size(0), -1, x.size(-1))  # Reshape to [B, T, F]
        except Exception as e:
            raise RuntimeError(f"Error in reshaping: {e}")
        # print(f"After Reshaping Shape: {x.shape}")

        # Conformer Blocks
        attention_weights = []
        for i, block in enumerate(self.blocks):
            try:
                x, attn = block(x)
            except Exception as e:
                raise RuntimeError(f"Error in Conformer block {i}: {e}")
            if x.isnan().any():
                raise ValueError(f"NaN detected in Conformer block {i} output.")
            # print(f"Block {i} Output Shape: {x.shape}")
            # print(f"Block {i} Attention Mean: {attn.mean().item()}, Std: {attn.std().item()}")
            attention_weights.append(attn)

        # Output Layer
        try:
            logits = self.output_layer(x)  # [B, T, num_classes]
        except Exception as e:
            raise RuntimeError(f"Error in output_layer: {e}")
        # print(f"Logits Shape: {logits.shape}")
        # print(f"Logits Mean: {logits.mean().item()}, Std: {logits.std().item()}")

        # Convert logits to log probabilities
        try:
            log_probs = F.log_softmax(logits, dim=-1)
        except Exception as e:
            raise RuntimeError(f"Error in log_softmax: {e}")
        # print(f"Log Probs Shape: {log_probs.shape}")
        # print(f"Log Probs Mean: {log_probs.mean().item()}, Std: {log_probs.std().item()}")

        # Calculate the length of the predicted sequences (batch_size,)
        log_probs_length = torch.full((x.size(0),), x.size(1), dtype=torch.long).to(self.device)

        # Final Output
        return {
            "log_probs": log_probs,
            "attention_weights": attention_weights,
            "log_probs_length": log_probs_length,
        }





    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().str()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info