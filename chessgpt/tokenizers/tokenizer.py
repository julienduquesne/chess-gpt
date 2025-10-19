import logging
from chessgpt.datasets import GamesDataset
from .constants import PAD_TOKEN, UNK_TOKEN, START_TOKEN


class Tokenizer:
    """A simple whitespace tokenizer with vocabulary support."""

    def __init__(self):
        self.vocab = {}
        self._inv_vocab = {}

    def fit(self, dst: GamesDataset):
        unique_tokens = set()
        for sen in dst:
            tokens = sen.strip().split()
            unique_tokens.update(tokens)

        self.vocab = {
            token: idx for idx, token in enumerate(sorted(unique_tokens), start=0)
        }
        self.vocab[UNK_TOKEN] = len(self.vocab)
        self.vocab[PAD_TOKEN] = len(self.vocab)
        self.vocab[START_TOKEN] = len(self.vocab)
        self._inv_vocab = {idx: token for token, idx in self.vocab.items()}
        logging.info(f"Vocabulary size: {len(self.vocab)}")

    @property
    def pad_token_id(self) -> int:
        return self.vocab[PAD_TOKEN]

    @property
    def unk_token_id(self) -> int:
        return self.vocab[UNK_TOKEN]

    def load_vocab(self, vocab_file: str) -> dict:
        vocab = {}
        with open(vocab_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                token = line.strip()
                vocab[token] = idx
        vocab[UNK_TOKEN] = len(vocab)
        vocab[PAD_TOKEN] = len(vocab)
        return vocab

    def tokenize(self, text: str) -> list[int]:
        tokens = text.split()
        return [token if token in self.vocab else UNK_TOKEN for token in tokens]

    def encode(self, text: str) -> list[int]:
        tokens = self.tokenize(text)
        return [self.vocab[token] for token in tokens]

    def decode(self, token_ids: list[int]) -> str:
        tokens = [self._inv_vocab.get(token_id, UNK_TOKEN) for token_id in token_ids]
        return " ".join(tokens)
