from pathlib import Path


class GamesDataset:
    """
    A simple dataset class to store games in pgn format that loads data from a given file path.
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
