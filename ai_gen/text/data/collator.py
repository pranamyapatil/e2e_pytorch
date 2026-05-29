from __future__ import annotations

import torch


class Seq2SeqCollator:
    """
    Pads a batch of variable-length source/target pairs to the longest sequence
    in each batch. Called automatically by DataLoader when returned from
    Seq2SeqDataset.collate_fn.

    Output keys match the Seq2SeqTransformer.forward() batch contract:
        input_ids              (B, S)  source ids, right-padded with pad_id
        attention_mask         (B, S)  1 = real token, 0 = pad
        decoder_input_ids      (B, T)  BOS + target[:-1]
        decoder_attention_mask (B, T)
        labels                 (B, T)  target[1:] + EOS, padded with -100
                                       (-100 is ignored by F.cross_entropy)
    """

    def __init__(self, pad_id: int = 0):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        return {
            "input_ids":               self._pad([x["input_ids"]         for x in batch], self.pad_id),
            "attention_mask":          self._mask([x["input_ids"]        for x in batch]),
            "decoder_input_ids":       self._pad([x["decoder_input_ids"] for x in batch], self.pad_id),
            "decoder_attention_mask":  self._mask([x["decoder_input_ids"] for x in batch]),
            "labels":                  self._pad([x["labels"]            for x in batch], fill=-100),
        }

    @staticmethod
    def _pad(seqs: list[torch.Tensor], fill: int) -> torch.Tensor:
        max_len = max(s.size(0) for s in seqs)
        out = torch.full((len(seqs), max_len), fill, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, : s.size(0)] = s
        return out

    @staticmethod
    def _mask(seqs: list[torch.Tensor]) -> torch.Tensor:
        max_len = max(s.size(0) for s in seqs)
        mask = torch.zeros(len(seqs), max_len, dtype=torch.long)
        for i, s in enumerate(seqs):
            mask[i, : s.size(0)] = 1
        return mask
