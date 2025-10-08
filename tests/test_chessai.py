import chess
from src.chessAI import ChessAI, SimpleMLP


def test_simple_forward_and_predict():
    model = SimpleMLP([8, 4])
    import numpy as np
    X = np.zeros((2, model.sizes[0]), dtype=np.float32)
    X[0, -1] = 1.0
    preds = model.predict_batch(X)
    assert preds.shape[0] == 2


def test_choose_move_has_valid_move():
    ai = ChessAI(sequence=[], depth=1)
    board = chess.Board()
    mv = ai.choose_move(board)
    assert mv is None or isinstance(mv, chess.Move)
