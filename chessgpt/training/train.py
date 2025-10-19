import logging
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from chessgpt.constants import DATA_DIR
from chessgpt.dataloaders import GamesDataLoader
from chessgpt.datasets.dataset import GamesDataset
from chessgpt.model.chessgpt import ChessGPT
from chessgpt.tokenizers.tokenizer import Tokenizer


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
        run_train_epoch(model, train_dataloader, optimizer)
        run_val_epoch(model, val_dataloader)


def run_train_epoch(
    model: ChessGPT, dataloader: GamesDataLoader, optimizer: torch.optim.Optimizer
):
    """
    Run a single training epoch.
    Parameters
    ----------
    model : ChessGPT
        The ChessGPT model to be trained.
    dataloader : GamesDataLoader
        DataLoader for the training dataset.
    optimizer : torch.optim.Optimizer
        Optimizer for updating model parameters.
    """
    model.train()
    total_loss = 0.0
    total_tokens = 0
    pbar = tqdm(dataloader, desc="train", leave=False)
    for games_batch in pbar:
        tokens, attn_masks = games_batch
        attn_masks = torch.tril(
            attn_masks.view(-1, 1, attn_masks.size(-1)).repeat(
                1, attn_masks.size(-1), 1
            )
        ).bool()
        start_time = time.time()
        targets = tokens[:, 1:]
        inputs = tokens[:, :-1]
        optimizer.zero_grad()
        logits, _ = model(inputs, attention_mask=attn_masks[:, :-1])
        B, L, V = logits.shape
        loss = F.cross_entropy(logits.view(B * L, V), targets.reshape(B * L))
        loss.backward()
        optimizer.step()
        elapsed = time.time() - start_time
        batch_tokens = B * L
        total_loss += loss.item()
        total_tokens += batch_tokens
        running_loss = loss.item() / batch_tokens
        tok_per_s = batch_tokens / elapsed if elapsed > 0 else 0.0
        pbar.set_postfix(loss=f"{running_loss:.4f}", tok_s=f"{tok_per_s:.0f}")
    logging.info(f"Train loss: {total_loss / total_tokens}")


def run_val_epoch(
    model: ChessGPT,
    dataloader: GamesDataLoader,
):
    """
    Run a single validation epoch.

    Parameters
    ----------
    model : ChessGPT
        The ChessGPT model to be validated.
    dataloader : GamesDataLoader
        DataLoader for the validation dataset.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for games_batch in tqdm(dataloader):
            tokens, attn_masks = games_batch
            attn_masks = torch.tril(
                attn_masks.view(-1, 1, attn_masks.size(-1)).repeat(
                    1, attn_masks.size(-1), 1
                )
            ).bool()
            targets = tokens[:, 1:]
            inputs = tokens[:, :-1]
            logits, _ = model(inputs, attention_mask=attn_masks[:, :-1])
            B, L, V = logits.shape
            loss = F.cross_entropy(
                logits.view(B * L, V),
                targets.reshape(B * L),
            )
            total_loss += loss.item()
            total_tokens += B * L
    logging.info(f"Validation loss: {total_loss / total_tokens}")


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
    run_val_epoch(model, test_dataloader)
