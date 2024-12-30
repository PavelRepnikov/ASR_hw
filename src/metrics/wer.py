from typing import List
import torch
from torch import Tensor
import numpy as np

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_wer

class ArgmaxWERMetric(BaseMetric):
    """
    Computes the Word Error Rate (WER) using argmax decoding.

    Args:
        text_encoder: An object with `normalize_text` and `ctc_decode` methods 
                      for processing and decoding text.
    """
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ) -> float:
        """
        Args:
            log_probs: Log probabilities from the model (Tensor of shape [batch_size, seq_length, vocab_size]).
            log_probs_length: Lengths of the log probability sequences (Tensor of shape [batch_size]).
            text: List of ground-truth target texts.

        Returns:
            Average WER for the batch.
        """
        wers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().cpu().numpy()
        
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            # Normalize the target text
            target_text = self.text_encoder.normalize_text(target_text)
            
            # Ensure length is a scalar integer, selecting the first element if it's an array
            if isinstance(length, np.ndarray):
                length = int(length[0])  # Handle as needed: select the first element or adjust
                
            # Decode the prediction using CTC decoding
            pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            
            # print(f"Target text(WER): {target_text}")
            # print(f"Predicted text(WER): {pred_text}")               
            
            # Calculate WER and append
            wers.append(calc_wer(target_text, pred_text))
        
        # Handle empty WER list gracefully
        return sum(wers) / len(wers) if wers else 0.0
