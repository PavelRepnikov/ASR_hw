import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    batch_data = {
        "text_encoded_lengths": [],
        "spectrogram_lengths": [],
        "spectrograms": [],
        "text_encodeds": [],
        "texts": [],
        "audios": [],
        "audio_paths": [],
    }

    for item in dataset_items:
        batch_data["text_encoded_lengths"].append(item["text_encoded"].shape[1])
        batch_data["spectrogram_lengths"].append(item["spectrogram"].shape[2])

        batch_data["spectrograms"].append(
            item["spectrogram"].squeeze(0).transpose(0, -1)
        )
        batch_data["text_encodeds"].append(
            item["text_encoded"].squeeze(0).transpose(0, -1)
        )
        batch_data["texts"].append(item["text"])
        batch_data["audios"].append(item["audio"].squeeze(0))
        batch_data["audio_paths"].append(item["audio_path"])

    batch_data["text_encoded_lengths"] = torch.tensor(
        batch_data["text_encoded_lengths"], dtype=torch.long
    )
    batch_data["spectrogram_lengths"] = torch.tensor(
        batch_data["spectrogram_lengths"], dtype=torch.long
    )

    batch_data["spectrograms"] = pad_sequence(
        batch_data["spectrograms"], batch_first=True, padding_value=0.0
    ).transpose(1, -1)
    batch_data["text_encodeds"] = pad_sequence(
        batch_data["text_encodeds"], batch_first=True, padding_value=0
    ).transpose(1, -1)
    batch_data["audios"] = pad_sequence(
        batch_data["audios"], batch_first=True, padding_value=0.0
    ).transpose(1, -1)

    return {
        "text_encoded_length": batch_data["text_encoded_lengths"],
        "spectrogram_length": batch_data["spectrogram_lengths"],
        "spectrogram": batch_data["spectrograms"],
        "text_encoded": batch_data["text_encodeds"],
        "text": batch_data["texts"],
        "audio": batch_data["audios"],
        "audio_path": batch_data["audio_paths"],
    }