import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict

def collate_fn(dataset_items: List[Dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (List[Dict]): List of objects from dataset.__getitem__.

    Returns:
        result_batch (Dict[Tensor]): Dict containing batch-version of the tensors.
    """
    # print(f"Dataset items (first item): {dataset_items[0]}")  # Print the first item in the list

    audio = [item['audio'] for item in dataset_items]
    spectrogram = [item['spectrogram'] for item in dataset_items]
    text_encoded = [item['text_encoded'] for item in dataset_items]
    
    for idx, text in enumerate(text_encoded):
        if text.dim() != 1:
            text_encoded[idx] = text.squeeze(0)
            if text_encoded[idx].dim() != 1:
                raise ValueError(f"Text at index {idx} is still not a 1D tensor after squeezing: {text.size()}")
    
    text_strings = [item['text'] for item in dataset_items]
    
    text_encoded_padded = pad_sequence(text_encoded, batch_first=True, padding_value=0)

    max_audio_length = max([item.size(1) for item in audio])
    padded_audio = torch.stack([
        torch.cat([item, torch.zeros(item.size(0), max_audio_length - item.size(1))], dim=1) 
        if item.size(1) < max_audio_length else item
        for item in audio
    ])

    max_spectrogram_length = max([item.size(2) for item in spectrogram])
    padded_spectrogram = torch.stack([
        torch.cat([item, torch.zeros(item.size(0), item.size(1), max_spectrogram_length - item.size(2))], dim=2) 
        if item.size(2) < max_spectrogram_length else item
        for item in spectrogram
    ])

    for idx, spec in enumerate(spectrogram):
        if spec.dim() == 3 and spec.size(0) == 1:
            spectrogram[idx] = spec.squeeze(0)
        if spectrogram[idx].dim() == 3:
            spectrogram[idx] = spectrogram[idx].mean(dim=0)

    # print(f"Padded audio shape: {padded_audio.shape}")
    # print(f"Padded spectrogram shape: {padded_spectrogram.shape}")

    spectrogram_lengths = torch.tensor([item.size(-1) for item in spectrogram], dtype=torch.long)
    # padded_spectrogram = padded_spectrogram[:, :, :128, 0]
    padded_spectrogram = padded_spectrogram[:, :, :128, :]


    result_batch = {
        'audio': padded_audio,
        'spectrogram': padded_spectrogram,
        'text_encoded': text_encoded_padded,
        'text': text_strings,
        'text_encoded_length': torch.tensor([len(item['text_encoded']) for item in dataset_items], dtype=torch.long),
        'audio_path': [item['audio_path'] for item in dataset_items],
        'spectrogram_length': spectrogram_lengths
    }

    return result_batch
