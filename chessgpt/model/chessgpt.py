import torch
import torch.nn as nn

from .transformer import TransformerBlock
from .positional_encoding import positional_encoding


class ChessGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        embed_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_size
        )
        self.pos_embeddings = positional_encoding(max_seq_len, embed_size)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(embed_size, num_heads, hidden_dim)
                for _ in range(num_layers)
            ]
        )
        self.proj = nn.Linear(embed_size, vocab_size)

    def forward(
        self, tokens: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the ChessGPT model.

        Parameters
        ----------
        tokens : torch.Tensor
            Input tensor of shape (batch_size, seq_len) containing token indices.
        attention_mask : torch.Tensor
            Attention mask tensor of shape (seq_len, seq_len).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, embed_size).
        """
        x = self.embeddings(tokens) + self.pos_embeddings[: tokens.size(1), :]
        for transformer in self.transformer_blocks:
            x, _ = transformer(x)
        logits = self.proj(x)
        return logits, x


def print_model_summary(model: torch.nn.Module):
    print(model)
    total = 0
    trainable = 0
    print("\nParameter summary:")
    for name, param in model.named_parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
        print(f"{name:40s} shape={tuple(param.shape)} params={count}")
    print(f"\nTotal params: {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Non-trainable params: {total - trainable:,}")
