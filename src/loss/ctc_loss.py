import torch
from torch import Tensor
from torch.nn import CTCLoss


class CTCLossWrapper(CTCLoss):
    def forward(
        self, log_probs, log_probs_length, text_encoded, text_encoded_length, **batch
    ) -> Tensor:
        # Transpose log_probs to match the shape expected by CTC Loss
        log_probs_t = torch.transpose(log_probs, 0, 1)

        # Print the batch_size and input_lengths to debug
        # print("Batch size:", log_probs.size(0))
        # print("log_probs_length shape:", log_probs_length.shape)
        # print("log_probs_length:", log_probs_length)
        # print("text_encoded_length shape:", text_encoded_length.shape)
        # print("text_encoded_length:", text_encoded_length)
        
        log_probs_length = log_probs_length.max(dim=1)[0]  

        # Ensure text_encoded_length is also 1D
        assert text_encoded_length.dim() == 1, "text_encoded_length must be a 1D tensor"


        # print("log_probs_length:", log_probs_length)
        loss = super().forward(
            log_probs=log_probs_t,
            targets=text_encoded,
            input_lengths=log_probs_length,
            target_lengths=text_encoded_length,
        )

        return {"loss": loss}
