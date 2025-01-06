import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.datasets import LIBRISPEECH
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from jiwer import wer, cer
import string
import random
import numpy as np
from itertools import islice

wandb.init(
    project="asr-LAST",
    config={
        "batch_size": 8,
        "learning_rate": 3e-5,
        "epochs": 10000,
        "dataset": "LibriSpeech train-clean-360",
    }
)
config = wandb.config

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = config.batch_size
LEARNING_RATE = config.learning_rate
EPOCHS = config.epochs
epoch_length = 100
characters = string.ascii_lowercase + " '"
blank_label = len(characters)

char_to_idx = {char: idx for idx, char in enumerate(characters)}
char_to_idx['<blank>'] = blank_label
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

def text_to_indices(text):
    return [char_to_idx.get(char, blank_label) for char in text.lower() if char in char_to_idx]

def indices_to_text(indices):
    chars = []
    for idx in indices:
        if idx == blank_label:
            continue
        chars.append(idx_to_char.get(idx, ''))
    return ''.join(chars)

class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        waveform, sample_rate, transcript, _, _, _ = self.dataset[idx]
        return waveform.squeeze(0), transcript.lower()

def collate_fn(batch):
    waveforms, transcripts = zip(*batch)
    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80)
    spectrograms = [mel_transform(waveform) for waveform in waveforms]

    max_length = max(s.size(1) for s in spectrograms)

    spectrograms = [
        torch.nn.functional.pad(s, (0, max_length - s.size(1))) if s.size(1) < max_length else s
        for s in spectrograms
    ]
    spectrograms = torch.stack(spectrograms)

    spectrograms = spectrograms.permute(2, 0, 1)

    input_lengths = torch.full((spectrograms.size(1),), max_length, dtype=torch.long)

    if wandb.run is not None:
        spect = spectrograms[:, 0, :].cpu().numpy()
        wandb.log({"spectrogram": wandb.Image(spect, caption="Input Spectrogram")})

    targets = [text_to_indices(t) for t in transcripts]
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets = torch.cat([torch.tensor(t, dtype=torch.long) for t in targets])

    return spectrograms, targets, input_lengths, target_lengths, transcripts

val_dataset = SpeechDataset(LIBRISPEECH("./data", url="test-clean", download=False))
test_clean_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)


train_dataset = SpeechDataset(LIBRISPEECH("./data", url="train-clean-360", download=False))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

class DeepSpeech2(nn.Module):
    def __init__(self, num_classes, input_dim=80, hidden_dim=512, num_gru_layers=5, dropout=0.1, n_tokens=None):
        super(DeepSpeech2, self).__init__()
        self.rnn = nn.ModuleList()
        for _ in range(num_gru_layers):
            input_size = input_dim if _ == 0 else hidden_dim * 2  
            self.rnn.append(nn.GRU(input_size, hidden_dim, batch_first=True, dropout=dropout, bidirectional=True))
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

        self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)

    def forward(self, **batch):
        x = batch["spectrogram"]
        x = x.permute(1, 0, 2).contiguous()
        x = x.squeeze(-1)
        for gru_layer in self.rnn:
            x, _ = gru_layer(x)

        x = x.contiguous().view(-1, x.size(2))
        x = self.batch_norm(x)

        x = x.view(batch["spectrogram"].size(0), -1, x.size(1))

        x = self.fc(x)

        log_probs = F.log_softmax(x, dim=-1)

        input_lengths = torch.full((x.size(0),), x.size(1), dtype=torch.int32).to(x.device)

        output = {
            "log_probs": log_probs,
            "log_probs_length": input_lengths
        }

        return output

    def __str__(self):
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum([p.numel() for p in self.parameters() if p.requires_grad])

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info

model = DeepSpeech2(num_classes=len(char_to_idx)).to(DEVICE)

criterion = nn.CTCLoss(blank=blank_label, zero_infinity=True).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def greedy_decoder(output, output_lengths):
    output = torch.softmax(output, dim=2)
    _, max_indices = torch.max(output, dim=2)
    max_indices = max_indices.transpose(0, 1).cpu().numpy()
    
    decoded = []
    for indices in max_indices:
        text = []
        previous = blank_label
        for idx in indices:
            if idx != previous and idx != blank_label:
                text.append(idx_to_char.get(idx, ''))
            previous = idx
        decoded.append(''.join(text))
    return decoded



best_cer = float('inf')


for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    # Shuffle the dataset manually to ensure continuity across epochs
    train_indices = torch.randperm(len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    
    start_idx = (epoch * epoch_length * BATCH_SIZE) % len(train_dataset)
    end_idx = start_idx + (epoch_length * BATCH_SIZE)
    selected_indices = train_indices[start_idx:end_idx]
    epoch_subset = torch.utils.data.Subset(train_dataset, selected_indices)

    epoch_loader = DataLoader(epoch_subset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    progress_bar = tqdm(epoch_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch", total=epoch_length)
    
    for batch_idx, (spectrograms, targets, input_lengths, target_lengths, transcripts) in enumerate(progress_bar):
        spectrograms = spectrograms.to(DEVICE)
        targets = targets.to(DEVICE)
        input_lengths = input_lengths.to(DEVICE)
        target_lengths = target_lengths.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(spectrogram=spectrograms)
        loss = criterion(outputs["log_probs"], targets, input_lengths, target_lengths)

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
    
    epoch_loss = running_loss / epoch_length
    
    model.eval()
    with torch.no_grad():
        val_spectrograms, val_targets, val_input_lengths, val_target_lengths, val_transcripts = next(iter(test_clean_loader))
        val_spectrograms = val_spectrograms.to(DEVICE)
        val_outputs = model(spectrogram=val_spectrograms)
        predictions = greedy_decoder(val_outputs['log_probs'], val_input_lengths)

        targets_text = []
        start = 0
        for length in val_target_lengths:
            t = val_targets[start:start + length]
            start += length
            targets_text.append(indices_to_text(t.cpu().numpy()))
        
        total_cer = 0
        total_wer = 0
        for pred, target in zip(predictions, targets_text):
            total_cer += cer(target, pred)
            total_wer += wer(target, pred)
        avg_cer = total_cer / len(predictions)
        avg_wer = total_wer / len(predictions)
        
        for i in range(min(5, len(predictions))):
            print(f"Prediction {i}: {predictions[i]}")
            print(f"Target {i}: {targets_text[i]}")
        
        wandb.log({
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "CER": avg_cer,
            "WER": avg_wer,
            "sample_input_spectrogram": wandb.Image(val_spectrograms[0].cpu().numpy(), caption="Sample Spectrogram"),
            "sample_prediction": predictions[0],
            "sample_target": targets_text[0]
        })
        
        for i in range(min(5, len(predictions))):
            wandb.log({
                f"prediction_{i}": predictions[i],
                f"target_{i}": targets_text[i]
            })
        
        if avg_cer < best_cer:
            best_cer = avg_cer
            torch.save(model.state_dict(), f"best_model_epoch_{epoch+1}_cer_{avg_cer:.4f}.pth")
    
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.4f}, CER: {avg_cer:.4f}, WER: {avg_wer:.4f}")

torch.save(model.state_dict(), "deepspeech2_final.pth")

