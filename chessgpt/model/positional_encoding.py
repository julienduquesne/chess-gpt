import torch


def positional_encoding(max_seq_len: int, embed_size: int) -> torch.Tensor:
    """
    Generate positional encoding for input sequences.

    Parameters
    ----------
    max_seq_len : int
        Maximum length of the input sequences.
    embed_size : int
        Dimension of the embeddings.
    Returns
    -------
    torch.Tensor
        Positional encoding tensor of shape (max_seq_len, embed_size).
    """
    position = torch.arange(0, max_seq_len).unsqueeze(1)
    div_term = torch.pow(
        10000, (2 * (torch.arange(0, embed_size, 2) // 2)) / embed_size
    )
    pe = torch.zeros(max_seq_len, embed_size)
    pe[:, 0::2] = torch.sin(position / div_term)
    pe[:, 1::2] = torch.cos(position / div_term)
    return pe
