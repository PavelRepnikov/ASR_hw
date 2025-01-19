import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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
        # spectrogram_for_plot = torch.log(spectrogram_for_plot + 1e-5) # for beautiful pictures when logging only
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("spectrogram", image)

    def log_predictions(
        self,
        text,
        logits: torch.tensor,
        log_probs: torch.tensor,
        log_probs_length: torch.tensor,
        audio_path,
        audio: torch.tensor,
        examples_to_log=5,
        **batch
    ):
        # TODO add beam search
        # Note: by improving text encoder and metrics design
        # this logging can also be improved significantly

        # сперва будем выбирать случайные examples_to_log объектов, а потом всё считать
        indices = random.sample(range(len(text)), min(examples_to_log, len(text)))

        texts = [text[i] for i in indices]
        logits = logits[indices].detach().cpu().numpy()
        log_probas = log_probs[indices].detach().cpu().numpy()
        log_probs_lengths = log_probs_length[indices].detach().cpu().numpy()
        audio_paths = [audio_path[i] for i in indices]
        audios = audio[indices].squeeze().numpy()

        argmax_inds = log_probas.argmax(-1)
        argmax_inds = [
            inds[: int(ind_len)]
            for inds, ind_len in zip(argmax_inds, log_probs_lengths)
        ]
        argmax_texts_raw = [self.text_encoder.decode(inds) for inds in argmax_inds]
        argmax_texts = [self.text_encoder.ctc_decode(inds) for inds in argmax_inds]

        # CTC bs
        # bs_texts = [
        #     self.text_encoder.ctc_beam_search(proba[:length], 3)
        #     for proba, length in zip(log_probas, log_probs_lengths)
        # ]
        # lm_texts = [
        #     self.text_encoder.lm_ctc_beam_search(logits_vec[:length], 25)
        #     for logits_vec, length in zip(logits, log_probs_lengths)
        # ]

        tuples = list(
            zip(
                argmax_texts,
                # bs_texts,
                # lm_texts,
                texts,
                argmax_texts_raw,
                audio_paths,
                audios,
            )
        )

        rows = {}
        for (
            pred_argmax,
            # pred_bs,
            # pred_lm,
            target,
            raw_pred,
            audio_path,
            audio_aug,
        ) in tuples:
            target = self.text_encoder.normalize_text(target)
            wer = calc_wer(target, pred_argmax) * 100
            cer = calc_cer(target, pred_argmax) * 100

            # beam_wer = calc_wer(target, pred_bs) * 100
            # beam_cer = calc_cer(target, pred_bs) * 100

            # lm_wer = calc_wer(target, pred_lm) * 100
            # lm_cer = calc_cer(target, pred_lm) * 100

            rows[Path(audio_path).name] = {
                "audio_augmented": self.writer.wandb.Audio(audio_aug, sample_rate=16000),
                "target": target,
                "raw prediction": raw_pred,
                "predictions": pred_argmax,
                # "beam_predictions": pred_bs,
                # "lm_predictions": pred_lm,
                "wer_argmax": wer,
                "cer_argmax": cer,
                # "wer_beam": beam_wer,
                # "cer_beam": beam_cer,
                # "wer_lm": lm_wer,
                # "cer_lm": lm_cer,
            }

        self.writer.add_table(
            "predictions", pd.DataFrame.from_dict(rows, orient="index")
        )
