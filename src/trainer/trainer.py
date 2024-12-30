from pathlib import Path

import pandas as pd
import numpy as np

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)

        if not isinstance(outputs, dict):
            raise ValueError(f"Expected model output to be a dictionary, but got {type(outputs)}")

        # Further check if 'log_probs' is in the outputs (adjust based on your model's expected output)
        if "log_probs" not in outputs:
            raise KeyError("Expected 'log_probs' key in model output, but it is missing")


        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_spectrogram(**batch)
        else:
            # Log Stuff
            self.log_spectrogram(**batch)
            self.log_predictions(**batch)

    def log_spectrogram(self, spectrogram, **batch):
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("spectrogram", image)
    


    def log_predictions(
        self, text, log_probs, log_probs_length, audio_path, examples_to_log=10, **batch
    ):
        # Perform argmax to get predicted indices
        argmax_inds = log_probs.cpu().argmax(-1).numpy()

        # Ensure log_probs_length is correctly handled as a 1D array of lengths
        log_probs_length = log_probs_length.cpu().numpy()  # Convert to numpy if it's a tensor

        # Safely slice the predictions based on log_probs_length
        for inds, ind_len in zip(argmax_inds, log_probs_length):
            # print(f"ind_len type: {type(ind_len)}, value: {ind_len}")  # Debug print
            # print(f"inds type: {type(inds)}, value: {inds}")  # Debug print
            
            # Check if ind_len is an array with multiple elements
            if isinstance(ind_len, np.ndarray):
                # If ind_len is an array, process each element individually
                for length in ind_len:
                    # Slice each prediction based on its corresponding length
                    inds = inds[:int(length)]  # Ensure ind_len is used correctly as an integer
                    # print(f"Sliced inds: {inds}")  # Debug print
            else:
                # If it's a scalar, handle it as a single value
                inds = inds[:int(ind_len)]  # Ensure ind_len is used correctly as an integer
                # print(f"Sliced inds: {inds}")  # Debug print

        # Decode the predictions
        argmax_texts_raw = [self.text_encoder.decode(inds) for inds in argmax_inds]
        argmax_texts = [self.text_encoder.ctc_decode(inds) for inds in argmax_inds]
        
        # Create tuples of predictions, targets, raw predictions, and audio paths
        tuples = list(zip(argmax_texts, text, argmax_texts_raw, audio_path))

        rows = {}
        # Log results for the first few examples
        for pred, target, raw_pred, audio_path in tuples[:examples_to_log]:
            target = self.text_encoder.normalize_text(target)
            wer = calc_wer(target, pred) * 100
            cer = calc_cer(target, pred) * 100

            # Collect results into a dictionary
            rows[Path(audio_path).name] = {
                "target": target,
                "raw prediction": raw_pred,
                "predictions": pred,
                "wer": wer,
                "cer": cer,
            }
        
        # Log the table using the writer
        self.writer.add_table(
            "predictions", pd.DataFrame.from_dict(rows, orient="index")
        )
