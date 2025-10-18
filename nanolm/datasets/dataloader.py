from typing import Callable
from nanolm.tokenizers.tokenizer import Tokenizer
from .dataset import TextDataset


class DataLoader:
    def __init__(
        self,
        dataset: TextDataset,
        tokenizer: Tokenizer,
        batch_size: int,
        collate_fn: Callable[[list[list[int]]], tuple],
        shuffle: bool = True,
    ):
        self._dataset = dataset
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._num_samples = len(dataset)
        self._collate_fn = collate_fn

    def __iter__(self):
        indices = list(range(self._num_samples))
        if self._shuffle:
            import random

            random.shuffle(indices)

        for start_idx in range(0, self._num_samples, self._batch_size):
            batch_indices = indices[start_idx : start_idx + self._batch_size]
            batch = self._collate_fn(
                [self._tokenizer.encode(self._dataset[i]) for i in batch_indices]
            )
            yield batch

    def __len__(self):
        return (self._num_samples + self._batch_size - 1) // self._batch_size
