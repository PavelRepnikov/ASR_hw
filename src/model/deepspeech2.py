import torch
from torch import nn
from torch.nn import Sequential


class NormGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.1):
        """
        Initializes a GRU layer with batch normalization.
        
        Args:
            input_dim (int): Dimensionality of input features.
            hidden_dim (int): Dimensionality of hidden states.
            dropout_rate (float): Dropout rate for GRU layer.
        """
        super().__init__()
        self.gru_layer = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            dropout=dropout_rate,
            batch_first=False,
            bidirectional=True,
        )
        self.normalizer = nn.BatchNorm1d(hidden_dim)

    def forward(self, inputs, hidden_state=None):
        """
        Executes the forward pass through GRU and normalization layers.

        Args:
            inputs (Tensor): Input sequence tensor.
            hidden_state (Tensor, optional): Initial hidden state.

        Returns:
            Tensor: Output features from the GRU.
            Tensor: Updated hidden state.
        """
        output, hidden_state = self.gru_layer(inputs, hidden_state)
        output = output.view(output.size(0), output.size(1), 2, -1).sum(dim=2)

        batch_size, time_steps = output.size(1), output.size(0)
        normalized = output.view(batch_size * time_steps, -1)
        normalized = self.normalizer(normalized)
        output = normalized.view(time_steps, batch_size, -1).contiguous()

        return output, hidden_state


class DeepSpeech2(nn.Module):
    def __init__(self, n_feats, n_tokens, num_rnn_layers, hidden_size, rnn_dropout):
        """
        Initializes the DeepSpeech2 model.

        Args:
            n_feats (int): Number of input features.
            n_tokens (int): Number of output tokens.
            num_rnn_layers (int): Number of RNN layers.
            hidden_size (int): Size of RNN hidden state.
            rnn_dropout (float): Dropout rate for RNN layers.
        """
        super().__init__()
        self.conv_params = {
            "conv1": {"padding": (20, 5), "kernel_size": (41, 11), "stride": (2, 2)},
            "conv2": {"padding": (10, 5), "kernel_size": (21, 11), "stride": (2, 2)},
            "conv3": {"padding": (10, 5), "kernel_size": (21, 11), "stride": (2, 1)},
        }

        self.conv_layers = Sequential(
            nn.Conv2d(1, 32, **self.conv_params["conv1"]),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, **self.conv_params["conv2"]),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 96, **self.conv_params["conv3"]),
            nn.BatchNorm2d(96),
            nn.ReLU(),
        )

        rnn_input_dim = self.calculate_rnn_input_size(n_feats) * 96
        self.recurrent_layers = Sequential(
            *[
                NormGRU(
                    input_dim=(hidden_size if i > 0 else rnn_input_dim),
                    hidden_dim=hidden_size,
                    dropout_rate=rnn_dropout,
                )
                for i in range(num_rnn_layers)
            ]
        )

        self.output_layer = nn.Linear(hidden_size, n_tokens)

    def calculate_rnn_input_size(self, n_feats):
        """
        Computes the input size for RNNs after convolution layers.

        Args:
            n_feats (int): Number of input features.
        Returns:
            int: Transformed size after convolution layers.
        """
        size = n_feats
        for params in self.conv_params.values():
            size = (
                size + 2 * params["padding"][0] - params["kernel_size"][0]
            ) // params["stride"][0] + 1
        return size

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Forward pass of the model.

        Args:
            spectrogram (Tensor): Input spectrogram tensor.
            spectrogram_length (Tensor): Length of input spectrograms.

        Returns:
            dict: Dictionary containing logits, log probabilities, and lengths.
        """
        features = self.conv_layers(spectrogram.unsqueeze(1))
        features = features.view(
            features.size(0), features.size(1) * features.size(2), features.size(3)
        )
        features = features.transpose(1, 2).transpose(0, 1).contiguous()

        hidden_state = None
        for rnn_layer in self.recurrent_layers:
            features, hidden_state = rnn_layer(features, hidden_state)

        time_steps, batch_size = features.size(0), features.size(1)
        features = features.view(time_steps * batch_size, -1)
        logits = self.output_layer(features)
        logits = logits.view(time_steps, batch_size, -1).transpose(0, 1)

        log_probs = nn.functional.log_softmax(logits, dim=-1)
        return {
            "logits": logits,
            "log_probs": log_probs,
            "log_probs_length": self.transform_temporal_lengths(spectrogram_length),
        }

    def transform_temporal_lengths(self, input_lengths):
        """
        Computes the temporal lengths after convolution compression.

        Args:
            input_lengths (Tensor): Original input lengths.

        Returns:
            Tensor: Adjusted lengths after convolutions.
        """
        for params in self.conv_params.values():
            input_lengths = (
                input_lengths + 2 * params["padding"][1] - params["kernel_size"][1]
            ) // params["stride"][1] + 1
        return input_lengths

    def __str__(self):
        """
        Provides model summary with parameter counts.

        Returns:
            str: Model details.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        details = super().__str__()
        details += f"\nTotal parameters: {total_params}"
        details += f"\nTrainable parameters: {trainable_params}"
        return details
