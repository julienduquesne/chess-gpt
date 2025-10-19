import torch
from chessgpt.tokenizers import Tokenizer
from chessgpt.datasets import GamesDataset


class GamesDataLoader:
    def __init__(
        self,
        dataset: GamesDataset,
        tokenizer: Tokenizer,
        batch_size: int,
        shuffle: bool = True,
    ):
        self._dataset = dataset
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._num_samples = len(dataset)

    def __iter__(self):
        indices = list(range(self._num_samples))
        if self._shuffle:
            import random

            random.shuffle(indices)

        for start_idx in range(0, self._num_samples, self._batch_size):
            batch_indices = indices[start_idx : start_idx + self._batch_size]
            batch = self._collate(
                [self._tokenizer.encode(self._dataset[i]) for i in batch_indices]
            )
            yield batch

    def __len__(self):
        return (self._num_samples + self._batch_size - 1) // self._batch_size

    def _collate(
        self, batch: list[list[int]], max_length: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Collate function to pad sequences in a batch to the same length.
        """
        max_length = (
            max_length if max_length is not None else max(len(seq) for seq in batch)
        )
        padded_batch = []
        attention_masks = []

        for seq in batch:
            padding_length = max_length - len(seq)
            padded_seq = seq + [self._tokenizer.pad_token_id] * padding_length
            attention_mask = [1] * len(seq) + [0] * padding_length

            padded_batch.append(padded_seq)
            attention_masks.append(attention_mask)

        return torch.tensor(padded_batch), torch.tensor(attention_masks)
