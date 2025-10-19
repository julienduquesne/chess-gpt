import logging
import torch
import torch.nn.functional as F
from tqdm import tqdm

from chessgpt.constants import DATA_DIR
from chessgpt.dataloaders import GamesDataLoader, GamesBatch
from chessgpt.datasets import GamesDataset
from chessgpt.eval import count_legal_moves
from chessgpt.model import ChessGPT
from chessgpt.tokenizers import Tokenizer


def train(
    model: ChessGPT,
    train_dataloader: GamesDataLoader,
    val_dataloader: GamesDataLoader,
    num_epochs: int,
    lr: float,
):
    """
    Train the ChessGPT model.

    Parameters
    ----------
    model : ChessGPT
        The ChessGPT model to be trained.
    train_dataloader : GamesDataLoader
        DataLoader for the training dataset.
    val_dataloader : GamesDataLoader
        DataLoader for the validation dataset.
    num_epochs : int
        Number of training epochs.
    lr : float
        Learning rate for the optimizer.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(num_epochs):
        run_epoch(model, train_dataloader, train=True, optimizer=optimizer)
        run_epoch(model, val_dataloader, train=False)


def run_epoch(
    model: ChessGPT,
    dataloader: GamesDataLoader,
    train: bool,
    optimizer: torch.optim.Optimizer | None = None,
):
    """
    Run a single training epoch.

    Parameters
    ----------
    model : ChessGPT
        The ChessGPT model to be trained.
    dataloader : GamesDataLoader
        DataLoader for the training dataset.
    train : bool
        If True, run training; otherwise, run validation.
    optimizer : torch.optim.Optimizer | None
        Optimizer for updating model parameters.
    """
    if train:
        if optimizer is None:
            raise ValueError("Optimizer must be provided for training.")
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    total_tokens = 0
    pbar = tqdm(dataloader, desc="train" if train else "val", leave=False)
    legal_moves = 0
    for games_batch in pbar:
        inputs, attn_masks, targets = _prepare_batch(games_batch)
        if train:
            optimizer.zero_grad()  # type: ignore
        logits, _ = model(inputs, attention_mask=attn_masks[:, :-1])
        pred_tokens = logits.argmax(dim=-1)
        B, L, V = logits.shape
        pad_id = tokenizer.pad_token_id
        loss = F.cross_entropy(
            logits.view(B * L, V),
            targets.reshape(B * L),
            ignore_index=pad_id,
        )
        if train:
            loss.backward()
            optimizer.step()  # type: ignore

        batch_tokens = B * L
        total_loss += loss.item()
        total_tokens += batch_tokens
        pbar.set_postfix(loss=f"{(loss.item() / batch_tokens):.4f}")
        moves = tokenizer.decode_batch(inputs.tolist())
        pred_moves = tokenizer.decode_batch(pred_tokens.tolist())
        legal_moves += count_legal_moves(moves, pred_moves)
    logging.info(f"Train loss: {total_loss / total_tokens}")
    logging.info(
        f"Percentage of legal moves: {(legal_moves / total_tokens) * 100:.4f}%"
    )


def _prepare_batch(
    batch: GamesBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare a batch of data for training or evaluation.

    Parameters
    ----------
    batch : list[list[int]]
        A batch of tokenized game sequences.
    tokenizer : Tokenizer
        The tokenizer used to process the sequences.

    Returns
    -------
    tokens : torch.Tensor
        Tensor of input tokens.
    attn_masks : torch.Tensor
        Tensor of attention masks.
    targets : torch.Tensor
        Tensor of target tokens.
    """
    attn_masks = batch.attention_masks
    tokens = batch.tokens
    attn_masks = torch.tril(
        attn_masks.view(-1, 1, attn_masks.size(-1)).repeat(1, attn_masks.size(-1), 1)
    ).bool()
    targets = tokens[:, 1:]
    inputs = tokens[:, :-1]
    return inputs, attn_masks, targets


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_dataset = GamesDataset(DATA_DIR / "games.train.txt")
    val_dataset = GamesDataset(DATA_DIR / "games.val.txt")
    test_dataset = GamesDataset(DATA_DIR / "games.test.txt")

    tokenizer = Tokenizer()
    tokenizer.fit(train_dataset)
    train_dataloader = GamesDataLoader(
        dataset=train_dataset,
        tokenizer=tokenizer,
        batch_size=16,
        shuffle=True,
    )
    val_dataloader = GamesDataLoader(
        dataset=val_dataset,
        tokenizer=tokenizer,
        batch_size=16,
        shuffle=False,
    )
    model = ChessGPT(
        vocab_size=len(tokenizer.vocab),
        max_seq_len=512,
        embed_size=256,
        num_layers=1,
        num_heads=8,
        hidden_dim=512,
    )
    train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=10,
        lr=1e-4,
    )
    test_dataloader = GamesDataLoader(
        dataset=test_dataset,
        tokenizer=tokenizer,
        batch_size=16,
        shuffle=False,
    )
    run_epoch(model, test_dataloader, train=False)
