from pathlib import Path


class TextDataset:
    """
    A simple text dataset class that loads data from a given file path.
    """

    def __init__(self, path: Path):
        self.data: list[str] = _load_data(path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> str:
        return self.data[idx]


def _load_data(path: Path) -> list[str]:
    with open(path, "r") as f:
        return f.readlines()


def collate_fn(
    batch: list[list[int]], pad_token_id: int, max_length: int | None = None
) -> tuple:
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
        padded_seq = seq + [pad_token_id] * padding_length
        attention_mask = [1] * len(seq) + [0] * padding_length

        padded_batch.append(padded_seq)
        attention_masks.append(attention_mask)

    return padded_batch, attention_masks
