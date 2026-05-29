from __future__ import annotations

import torch


class AudioCollator:
    """
    Pads variable-length audio feature sequences and label sequences to the
    longest example in the batch. Called automatically by DataLoader when
    returned from AudioDataset.collate_fn.

    Output keys match the ConformerModel.forward() batch contract:
        features         (B, T_max, n_mels)  zero-padded
        feature_lengths  (B,)                real frame count per utterance
        labels           (B, L_max)          zero-padded token ids
        label_lengths    (B,)                real label length per utterance
    """

    def __call__(self, batch: list[dict]) -> dict:
        feature_lengths = torch.tensor(
            [x["feature_length"] for x in batch], dtype=torch.long
        )
        label_lengths = torch.tensor(
            [x["label_length"] for x in batch], dtype=torch.long
        )

        T_max = int(feature_lengths.max())
        L_max = int(label_lengths.max())
        n_mels = batch[0]["features"].size(1)
        B = len(batch)

        padded_features = torch.zeros(B, T_max, n_mels)
        padded_labels = torch.zeros(B, L_max, dtype=torch.long)

        for i, sample in enumerate(batch):
            f = sample["features"]
            l = sample["labels"]
            padded_features[i, : f.size(0)] = f
            padded_labels[i, : l.size(0)] = l

        return {
            "features":        padded_features,
            "feature_lengths": feature_lengths,
            "labels":          padded_labels,
            "label_lengths":   label_lengths,
        }
