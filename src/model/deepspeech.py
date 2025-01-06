import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSpeech2(nn.Module):
    def __init__(self, num_classes, input_dim=128, hidden_dim=1024, num_gru_layers=5, dropout=0.1, n_tokens=None):
        super(DeepSpeech2, self).__init__()

        # Define the layers of the model
        self.rnn = nn.ModuleList()
        for _ in range(num_gru_layers):
            input_size = input_dim if _ == 0 else hidden_dim * 2  # First GRU layer takes input_dim (128), subsequent layers take hidden_dim * 2
            self.rnn.append(nn.GRU(input_size, hidden_dim, batch_first=True, dropout=dropout, bidirectional=True))

        # Fully connected layer to transform hidden states to output classes
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

        # Batch normalization (for feature dimension)
        self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)

    def forward(self, **batch):
        # Extract spectrogram input and batch-related data from keyword arguments
        x = batch["spectrogram"]  # Shape: [batch_size, channels, freq_bins, time_steps]

        # Permute to match expected shape: [batch_size, time_steps, freq_bins]
        x = x.permute(0, 3, 2, 1).contiguous()  # Shape: [batch_size, time_steps, freq_bins, channels]
        x = x.squeeze(-1)  # Remove the channels dimension, shape: [batch_size, time_steps, freq_bins]

        # Pass through each GRU layer
        for gru_layer in self.rnn:
            x, _ = gru_layer(x)

        # Apply batch normalization over the feature dimension (hidden_dim * 2)
        x = x.reshape(-1, x.size(2))  # Flatten to shape [batch_size * time_steps, hidden_dim * 2]
        x = self.batch_norm(x)  # Apply batch normalization

        # Reshape back to [batch_size, time_steps, hidden_dim * 2]
        x = x.reshape(batch["spectrogram"].size(0), -1, x.size(1))

        # Pass through fully connected layer
        x = self.fc(x)

        # Calculate log probabilities (for training with CTC or other loss functions)
        log_probs = F.log_softmax(x, dim=-1)

        # Calculate input lengths (assuming no padding in the time dimension)
        input_lengths = torch.full((x.size(0),), x.size(1), dtype=torch.int32)
        # print("input_lengths:", input_lengths)
        # Return output as a dictionary with log_probs and input_lengths
        output = {
            "log_probs": log_probs,
            "log_probs_length": input_lengths
        }

        return output



    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum([p.numel() for p in self.parameters() if p.requires_grad])

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
