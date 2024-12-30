import torch
import torchaudio
from torch import nn
from src.model.deepspeech import DeepSpeech
from torchaudio.transforms import Resample
import heapq
from tqdm import tqdm


def from_pretrained(model, pretrained_path, device):
    """
    Initialize model with weights from pretrained pth file.
    """
    pretrained_path = str(pretrained_path)
    if hasattr(model, "logger"):
        model.logger.info(f"Loading model weights from: {pretrained_path} ...")
    else:
        print(f"Loading model weights from: {pretrained_path} ...")
    checkpoint = torch.load(pretrained_path, map_location=device)

    if checkpoint.get("state_dict") is not None:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint)

def preprocess_audio(audio_path, device, sample_rate=16000):
    """
    Preprocesses audio input by loading and converting it to the desired format.
    Args:
        audio_path (str): Path to the audio file.
        device (torch.device): Device for computations.
        sample_rate (int): Target sample rate for the audio.
    Returns:
        torch.Tensor: Preprocessed audio tensor.
    """
    waveform, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)
    return waveform.to(device)

def infer(model, audio_path, device, beam_width=5):
    
    labels = [chr(i) for i in range(28)] 
    blank_idx = 0

    waveform, sample_rate = torchaudio.load(audio_path)

    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        resampler = Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    waveform = waveform.unsqueeze(0)

    waveform = waveform.to(device)

    batch = {"audio": waveform}

    model.eval()
    with torch.no_grad():
        output = model(**batch)

    log_probs = output["log_probs"]
    input_lengths = output["log_probs_length"]

    log_probs = log_probs[0]
    input_lengths = input_lengths[0]

    beams = [([], 0.0)]

    for t in tqdm(range(input_lengths[0]), desc="Decoding", unit="step"):
        new_beams = []
        probs = log_probs[t].cpu().numpy()

        for seq, score in beams:
            for idx in range(len(probs)):
                new_seq = seq + [idx]
                new_score = score + probs[idx]
                new_beams.append((new_seq, new_score))

        beams = heapq.nlargest(beam_width, new_beams, key=lambda x: x[1])

    best_seq, _ = beams[0]

    transcription = []
    prev_char = None
    for idx in best_seq:
        if idx != blank_idx and (prev_char != idx):
            transcription.append(labels[idx])
        prev_char = idx

    print("Transcription:", "".join(transcription))

    return "".join(transcription)




if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    # device = "cuda"
    model = DeepSpeech(
        input_dim=80,
        hidden_dim=256,
        num_gru_layers=3,
        num_classes=29,
        dropout=0.1
    )

    pretrained_path = "saved/model_best.pth"
    from_pretrained(model, pretrained_path, device)

    audio_path = "ex.flac"  # Your specific audio file path

    output = infer(model, audio_path, device)
    # print(f"Transcription: {output}")