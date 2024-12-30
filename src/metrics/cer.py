from typing import List

import torch
from torch import Tensor
import numpy as np

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_cer

# TODO add beam search/lm versions
# Note: they can be written in a pretty way
# Note 2: overall metric design can be significantly improved


class ArgmaxCERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        cers = []
        
        # Get predictions as NumPy array
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        
        # Ensure lengths are NumPy arrays and detaching them from the graph
        lengths = log_probs_length.detach().cpu().numpy()
        
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            # Print the type and shape of length to debug
            # print(f"length: {length}, type: {type(length)}, shape: {getattr(length, 'shape', 'N/A')}")
            
            # Normalize the target text
            target_text = self.text_encoder.normalize_text(target_text)
            
            # If length is an array, choose the first element or handle as needed
            if isinstance(length, np.ndarray):
                length = int(length[0])  # Select the first element, or adjust as needed
            else:
                length = int(length)
            
            # Slice log_prob_vec based on length
            log_prob_vec = log_prob_vec[:length]
            
            # Decode the prediction text
            pred_text = self.text_encoder.ctc_decode(log_prob_vec)
            
            # Print the target text and prediction text for debugging
            # print(f"Target text(CER): {target_text}")
            # print(f"Predicted text(CER): {pred_text}")            
            
            # Calculate CER for this prediction
            cers.append(calc_cer(target_text, pred_text))
        
        # Return the average CER
        return sum(cers) / len(cers)
