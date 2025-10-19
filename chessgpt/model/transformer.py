import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    TODO : Add dropout functionality.
    """

    def __init__(self, embed_size: int, num_heads: int):
        self._embed_size = embed_size
        self._num_heads = num_heads
        self._head_dim = embed_size // num_heads
        assert self._head_dim * num_heads == embed_size, (
            "Embedding size must be divisible by number of heads"
        )
        self.w_q = nn.Linear(self._embed_size, self._embed_size)
        self.w_k = nn.Linear(self._embed_size, self._embed_size)
        self.w_v = nn.Linear(self._embed_size, self._embed_size)
        self.fc_out = nn.Linear(self._head_dim * num_heads, self._embed_size)
        self.scale = math.sqrt(self._head_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for multi-head attention.

        Parameters
        ----------
        query : torch.Tensor
            Query tensor of shape (batch_size, seq_len, embed_size).
        key : torch.Tensor
            Key tensor of shape (batch_size, seq_len, embed_size).
        value : torch.Tensor
            Value tensor of shape (batch_size, seq_len, embed_size).
        mask : torch.Tensor | None
            Optional mask tensor of shape (seq_len, seq_len) to prevent attention to certain positions

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, embed_size).
        torch.Tensor
            Attention weights of shape (batch_size, num_heads, seq_len, seq_len).
        """
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        B, L, _ = Q.shape

        Q = Q.view(B, L, self._num_heads, self._head_dim).transpose(1, 2)
        K = K.view(B, L, self._num_heads, self._head_dim).transpose(1, 2)
        V = V.view(B, L, self._num_heads, self._head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attention = torch.softmax(scores, dim=-1)
        out = torch.matmul(attention, V).transpose(1, 2)
        out = out.contiguous().view(B, L, self._head_dim * self._num_heads)
        out = self.fc_out(out)
        return out, attention


class TransformerBlock(nn.Module):
    """
    A single transformer block consisting of multi-head attention and feed-forward network.
    """

    def __init__(self, embed_size: int, num_heads: int, hidden_dim: int):
        self.attention = MultiHeadAttention(embed_size, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_size),
        )
        self.layernorm1 = nn.LayerNorm(embed_size)
        self.layernorm2 = nn.LayerNorm(embed_size)

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the transformer block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, embed_size).
        attention_mask : torch.Tensor | None
            Optional attention mask tensor of shape (seq_len, seq_len).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, embed_size).
        torch.Tensor
            Attention weights of shape (batch_size, num_heads, seq_len, seq_len).
        """
        attention, _ = self.attention(x, x, x, attention_mask)
        x = self.layernorm1(x + attention)
        ff = self.feed_forward(x)
        x = self.layernorm2(x + ff)
        return x, attention
