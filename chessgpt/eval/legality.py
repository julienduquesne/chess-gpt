import chess

from chessgpt.tokenizers.constants import END_TOKEN, PAD_TOKEN, START_TOKEN, UNK_TOKEN


def count_legal_moves(true_moves: list[str], pred_moves: list[str]) -> int:
    """
    Count the frequency of legal moves in the true and predicted moves.

    Parameters
    ----------
    true_moves : list[str]
        Batch of true moves of the game.
    preds_moves : list[str]
        Batch of predicted moves by the model.

    Returns
    -------
    legal_moves_count : int
        Number of legal moves in the predicted moves.
    """
    legal_moves_count = 0
    for game_moves, game_next_move in zip(true_moves, pred_moves):
        board = chess.Board()
        moves = game_moves.strip().split()
        next_moves = game_next_move.strip().split()
        for move, next_move in zip(moves, next_moves):
            if move == START_TOKEN:
                continue
            if move in [PAD_TOKEN, UNK_TOKEN, END_TOKEN]:
                break
            board.push_san(move)
            try:
                board.parse_san(next_move)
                legal_moves_count += 1
            except (
                chess.IllegalMoveError,
                chess.InvalidMoveError,
                chess.AmbiguousMoveError,
            ):
                continue
    return legal_moves_count
