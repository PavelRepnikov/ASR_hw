import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSpeech(nn.Module):
    def __init__(self, num_classes, input_dim=13, hidden_dim=1024, num_gru_layers=5, dropout=0.1, n_tokens=None):
        super(DeepSpeech, self).__init__()

        self.n_tokens = n_tokens
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_gru_layers = num_gru_layers
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        # self.gru = nn.GRU(input_size=128, hidden_size=hidden_dim, num_layers=num_gru_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.gru = nn.GRU(input_size=128, hidden_size=hidden_dim, num_layers=num_gru_layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, num_classes)  # Bidirectional GRU outputs twice the hidden size

        # Dropout layer
        self.dropout = nn.Dropout(dropout)
    

    def forward(self, audio=None, spectrogram=None, input_lengths=None, **kwargs):

        device = next(self.parameters()).device

        if audio is not None:
            if audio.dim() == 2:
                audio = audio.unsqueeze(1)
            elif audio.dim() == 3:
                audio = audio.unsqueeze(1)
            x = audio
        elif spectrogram is not None:
            if spectrogram.dim() == 3:
                spectrogram = spectrogram.unsqueeze(1)
            x = spectrogram
        else:
            raise ValueError("Either 'audio' or 'spectrogram' must be provided.")
        
        x = x.to(device)

        x = F.relu(self.conv1(x))
        # print(f"Shape after conv1: {x.shape}")

        x = F.relu(self.conv2(x))
        # print(f"Shape after conv2: {x.shape}")

        x = F.relu(self.conv3(x))
        
        x = x.squeeze(1)
        # print(f"Shape after squeeze: {x.shape}")

        if x.dim() == 4:
            x = x.squeeze(2)
        if x.dim() == 3:
            x = x.permute(0, 2, 1)
        # print(f"Shape of tensor before GRU: {x.shape}")

        if not x.is_contiguous():
            # print("Tensor is not contiguous before GRU. Forcing contiguity.")
            x = x.contiguous()

        # Reset hidden state for GRU at the start of each forward pass
        h0 = torch.zeros(self.num_gru_layers, x.size(0), self.hidden_dim).to(x.device)  # Move h0 to the correct device

        x, _ = self.gru(x, h0)

        x = self.dropout(x)

        x = self.fc(x)

        log_probs = F.log_softmax(x, dim=-1)
        
        input_lengths = self.calculate_sequence_lengths(x)

        output = {
            "log_probs": log_probs,
            "log_probs_length": input_lengths
        }

        return output

    def calculate_sequence_lengths(self, x):
        """
        Compute the length of each sequence in the batch.
        This is done by counting non-padding elements along the time dimension.
        """
        sequence_lengths = (x != 0).sum(dim=1)
        
        return sequence_lengths

    # In your loss computation function
    def compute_ctc_loss(self, log_probs, targets, input_lengths, target_lengths):
        # print("Log probs shape:", log_probs.shape)
        # print("Targets shape:", targets.shape)
        # print("Input lengths shape:", input_lengths.shape)
        # print("Target lengths shape:", target_lengths.shape)
        
        assert input_lengths.dim() == 1 and input_lengths.size(0) == log_probs.size(0), \
            f"Expected input_lengths of shape [batch_size], but got {input_lengths.shape} for batch size {log_probs.size(0)}"

        log_probs = log_probs.permute(1, 0, 2)
        # print("Permuted log_probs shape:", log_probs.shape)

        ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
        
        # print("CTC Loss:", loss.item())
        
        return loss

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
