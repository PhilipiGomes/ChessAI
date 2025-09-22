import chess
import numpy as np
import random
import os
from typing import List, Tuple, Optional, Dict
from openings import chess_openings

# --- Config ---
PIECE_TYPES = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]
NUM_CHANNELS = 12
SQUARES = 64
INPUT_SIZE = NUM_CHANNELS * SQUARES + 1
MATE_SCORE = 100000.0
MASK64 = (1 << 64) - 1

# --- Zobrist table generation ---
_zobrist_piece = {}
_zobrist_side = 0
_zobrist_castling = {}
_zobrist_ep = {}


def init_zobrist(seed: Optional[int] = 0):
    global _zobrist_piece, _zobrist_side, _zobrist_castling, _zobrist_ep
    rng = random.Random(seed)
    _zobrist_piece = {}
    for pidx in range(6):
        for color in (0, 1):
            for sq in range(64):
                _zobrist_piece[(pidx, color, sq)] = rng.getrandbits(64) & MASK64
    _zobrist_side = rng.getrandbits(64) & MASK64
    _zobrist_castling = {
        "K": rng.getrandbits(64) & MASK64,
        "Q": rng.getrandbits(64) & MASK64,
        "k": rng.getrandbits(64) & MASK64,
        "q": rng.getrandbits(64) & MASK64,
    }
    _zobrist_ep = {f: rng.getrandbits(64) & MASK64 for f in range(8)}


# initialize default zobrist
init_zobrist(seed=0)


def board_zobrist_key(board: chess.Board) -> int:
    """Compute Zobrist hash for entire board (no incremental updates)."""
    h = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            pidx = PIECE_TYPES.index(piece.piece_type)  # 0..5
            color = 0 if piece.color == chess.WHITE else 1
            h ^= _zobrist_piece[(pidx, color, sq)]
    if board.turn == chess.BLACK:
        h ^= _zobrist_side
    # castling rights
    if board.has_kingside_castling_rights(chess.WHITE):
        h ^= _zobrist_castling["K"]
    if board.has_queenside_castling_rights(chess.WHITE):
        h ^= _zobrist_castling["Q"]
    if board.has_kingside_castling_rights(chess.BLACK):
        h ^= _zobrist_castling["k"]
    if board.has_queenside_castling_rights(chess.BLACK):
        h ^= _zobrist_castling["q"]
    # en-passant file
    if board.ep_square is not None:
        file = chess.square_file(board.ep_square)
        h ^= _zobrist_ep[file]
    return h & MASK64


# --- Features: board -> vector ---
def board_to_feature_vector(board: chess.Board) -> np.ndarray:
    vec = np.zeros(INPUT_SIZE, dtype=np.float32)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            pidx = PIECE_TYPES.index(piece.piece_type)
            channel = pidx + (0 if piece.color == chess.WHITE else 6)
            vec[channel * SQUARES + sq] = 1.0
    vec[-1] = 1.0 if board.turn == chess.WHITE else -1.0
    return vec


# --- Simple MLP (corrigido, vetorizado, save/load) ---
import json
import os
from tqdm import tqdm


class SimpleMLP:
    """
    Rede MLP vetorizada. hidden_sizes é uma lista com qualquer número de camadas.
    """

    def __init__(
        self,
        input_size=INPUT_SIZE,
        hidden_sizes: List[int] = [128, 64],
        seed: Optional[int] = 42,
    ):
        rng = np.random.RandomState(seed)
        self.sizes = [int(input_size)] + [int(h) for h in hidden_sizes] + [1]
        self.n_layers = len(self.sizes) - 1
        self.W: List[np.ndarray] = []
        self.b: List[np.ndarray] = []
        for i in range(self.n_layers):
            fan_in = self.sizes[i]
            fan_out = self.sizes[i + 1]
            std = np.sqrt(1.0 / max(1, fan_in))  # Xavier-ish for tanh
            self.W.append(
                rng.normal(0.0, std, size=(fan_out, fan_in)).astype(np.float32)
            )
            self.b.append(np.zeros((fan_out,), dtype=np.float32))

    def forward(self, x: np.ndarray) -> float:
        a = x.astype(np.float32)
        for i in range(self.n_layers - 1):
            z = self.W[i].dot(a) + self.b[i]
            a = np.tanh(z)
        z = self.W[-1].dot(a) + self.b[-1]
        return float(z[0])

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Forward vetorizado: X shape (B, input_size) -> preds shape (B,)"""
        a = X.astype(np.float32)
        for i in range(self.n_layers - 1):
            z = a.dot(self.W[i].T) + self.b[i]  # (B, out)
            a = np.tanh(z)
        z = a.dot(self.W[-1].T) + self.b[-1]  # shape (B,1)
        return z.reshape(-1)

    def train_sgd(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs=10,
        lr=1e-3,
        batch_size=32,
        verbose=False,
        plot=True,
    ):
        """
        Vetorizado por mini-batch. Loss: 0.5 * mean((pred - y)^2).
        Mostra progresso por época usando tqdm quando verbose=True.
        """
        N = X.shape[0]
        if N == 0:
            return
        X = X.astype(np.float32)
        y = y.astype(np.float32).reshape(-1)
        losses = {}

        # barra de progresso nas épocas
        pbar = tqdm(range(epochs), desc="Treinando", unit="ep", disable=not verbose)
        for epoch in pbar:
            perm = np.random.permutation(N)
            epoch_loss_sum = 0.0
            for bstart in range(0, N, batch_size):
                batch_idx = perm[bstart : bstart + batch_size]
                Xb = X[batch_idx]  # (B, in)
                yb = y[batch_idx]  # (B,)
                B = Xb.shape[0]

                # Forward (store activations)
                activations = [Xb]  # a0 = Xb (B, in)
                zs = []
                a = Xb
                for i in range(self.n_layers - 1):
                    z = a.dot(self.W[i].T) + self.b[i]  # (B, out)
                    zs.append(z)
                    a = np.tanh(z)
                    activations.append(a)
                zL = a.dot(self.W[-1].T) + self.b[-1]  # (B, 1)
                zs.append(zL)
                preds = zL.reshape(-1)  # (B,)

                # Loss (mean over batch, with 0.5 factor)
                err = preds - yb
                batch_loss = 0.5 * np.mean(err**2)
                epoch_loss_sum += batch_loss * B

                # Backprop (vectorized)
                # dL/dzL = (pred - y) / B
                delta = (err.reshape(B, 1)) / float(B)  # (B, 1)

                # Prepare grads containers
                dW = [np.zeros_like(w) for w in self.W]
                db = [np.zeros_like(bb) for bb in self.b]

                # a_prev for last layer is the last hidden activation
                a_prev = activations[-1]  # (B, hidden_last)
                dW[-1] = delta.T.dot(a_prev).astype(np.float32)  # (1, hidden_last)
                db[-1] = np.sum(delta, axis=0).astype(np.float32)  # (1,)

                # propagate backwards for hidden layers
                delta_prev = delta  # (B, out_next)
                for l in range(self.n_layers - 2, -1, -1):
                    z_l = zs[l]  # (B, out_l)
                    deriv = 1.0 - np.tanh(z_l) ** 2  # (B, out_l)
                    # delta_l = (delta_prev @ W_{l+1}) * deriv
                    delta_l = delta_prev.dot(self.W[l + 1]) * deriv  # (B, out_l)
                    a_prev = activations[l]  # (B, in_l)
                    dW[l] = delta_l.T.dot(a_prev).astype(np.float32)  # (out_l, in_l)
                    db[l] = np.sum(delta_l, axis=0).astype(np.float32)  # (out_l,)
                    delta_prev = delta_l

                # update params
                for i in range(len(self.W)):
                    self.W[i] -= lr * dW[i]
                    self.b[i] -= lr * db[i]

            epoch_loss = epoch_loss_sum / float(N)
            if verbose:
                # atualiza postfix com loss
                pbar.set_postfix(loss=f"{epoch_loss:.6f}")
            if plot:
                losses[epoch] = epoch_loss
        return losses

    # --- salvar / carregar pesos (compatível com shapes diferentes) ---
    def save(self, prefix: str):
        os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
        np.save(
            prefix + "/chess_mlp_W.npy",
            np.array(self.W, dtype=object),
            allow_pickle=True,
        )
        np.save(
            prefix + "/chess_mlp_b.npy",
            np.array(self.b, dtype=object),
            allow_pickle=True,
        )
        with open(prefix + "/chess_mlp_meta.json", "w") as f:
            json.dump({"sizes": self.sizes}, f)

    def load(self, prefix: str):
        W_path = prefix + "/chess_mlp_W.npy"
        b_path = prefix + "/chess_mlp_b.npy"
        meta_path = prefix + "/chess_mlp_meta.json"
        if not os.path.exists(W_path) or not os.path.exists(b_path):
            raise FileNotFoundError(f"Arquivos do modelo não encontrados: {W_path} or {b_path}")
        W_obj = np.load(W_path, allow_pickle=True)
        b_obj = np.load(b_path, allow_pickle=True)
        self.W = [np.array(w, dtype=np.float32) for w in list(W_obj)]
        self.b = [np.array(bb, dtype=np.float32) for bb in list(b_obj)]
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
                self.sizes = [int(s) for s in meta.get("sizes", self.sizes)]
        else:
            input_size = self.W[0].shape[1]
            hidden_outs = [w.shape[0] for w in self.W[:-1]]
            self.sizes = [input_size] + hidden_outs + [self.W[-1].shape[0]]
        self.n_layers = len(self.sizes) - 1


# --- Transposition Table flags ---
EXACT = 0
LOWERBOUND = 1
UPPERBOUND = 2


# --- Chess AI with Negamax + Alpha-Beta + TT ---
class ChessAI:
    def __init__(
        self,
        model: Optional[SimpleMLP] = None,
        depth: int = 2,
        zobrist_seed: Optional[int] = 0,
        sequence: List[str] = [],
    ):
        self.model = model if model is not None else SimpleMLP()
        self.depth = max(1, int(depth))
        init_zobrist(seed=zobrist_seed)
        self.tt = {}  # zobrist_key -> (depth, flag, value, best_move_uci)
        self.sequence = sequence

    def load_model(self, prefix: str):
        """Carrega pesos salvos para a rede usada pela AI."""
        self.model.load(prefix)

    def evaluate_board(self, board: chess.Board) -> float:
        vec = board_to_feature_vector(board)
        return self.model.forward(vec)

    def _minmax(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
    ) -> float:
        key = board_zobrist_key(board)
        entry = self.tt.get(key)
        if entry is not None:
            entry_depth, entry_flag, entry_value, _ = entry
            if entry_depth >= depth:
                if entry_flag == EXACT:
                    return entry_value
                if entry_flag == LOWERBOUND and entry_value > alpha:
                    alpha = entry_value
                elif entry_flag == UPPERBOUND and entry_value < beta:
                    beta = entry_value
                if alpha >= beta:
                    return entry_value

        if board.is_game_over():
            if board.is_checkmate():
                return MATE_SCORE if board.turn == chess.BLACK else -MATE_SCORE
            else:
                return 0.0

        if depth == 0:
            return self.evaluate_board(board)

        alpha_orig = alpha
        best_value = -np.inf if maximizing else np.inf
        best_move_uci = None

        moves = list(board.legal_moves)

        def move_ordering(m: chess.Move) -> int:
            score = 0
            if board.is_capture(m):
                score += 1
            if board.gives_check(m):
                score += 2
            if m.promotion is not None:
                score += 1.5
            return score

        moves_sorted = (
            sorted(moves, key=move_ordering, reverse=True) if len(moves) > 1 else moves
        )

        for mv in moves_sorted:
            board.push(mv)
            val = self._minmax(board, depth - 1, alpha, beta, not maximizing)
            board.pop()

            if maximizing:
                if val > best_value:
                    best_value = val
                    best_move_uci = mv.uci()
                if val > alpha:
                    alpha = val
                if alpha >= beta:
                    break
            else:
                if val < best_value:
                    best_value = val
                    best_move_uci = mv.uci()
                if val < beta:
                    beta = val
                if alpha >= beta:
                    break

        if maximizing:
            if best_value <= alpha_orig:
                flag = UPPERBOUND
            elif best_value >= beta:
                flag = LOWERBOUND
            else:
                flag = EXACT
        else:
            if best_value >= beta:
                flag = LOWERBOUND
            elif best_value <= alpha_orig:
                flag = UPPERBOUND
            else:
                flag = EXACT

        self.tt[key] = (depth, flag, best_value, best_move_uci)
        return best_value

    def filter_openings(
        self, op: Dict[str, List[str]], sequence: List[str]
    ) -> Dict[str, List[str]]:
        if sequence:
            filtered_openings: Dict[str, List[str]] = {}
            for opening_name, moves in op.items():
                if moves[: len(sequence)] == sequence and (
                    len(moves) > len(sequence) if len(sequence) > 0 else True
                ):
                    filtered_openings[opening_name] = moves
            return filtered_openings
        else:
            return op

    def choose_move(self, board: chess.Board) -> Optional[chess.Move]:
        # Opening logic.
        if self.sequence or board.fen() == chess.STARTING_FEN:
            filtered = self.filter_openings(chess_openings, self.sequence)
            if filtered:
                opening = random.choice(list(filtered.items()))
                san_move = opening[1][len(self.sequence)]
                return chess.Move.from_uci(board.parse_san(san_move).uci())

        best_move: Optional[chess.Move] = None
        best_value = -np.inf
        alpha = -np.inf
        beta = np.inf

        # prepare move ordering using TT entry for root
        moves = list(board.legal_moves)

        def move_ordering(m: chess.Move) -> int:
            score = 0
            if board.is_capture(m):
                score += 1
            if board.gives_check(m):
                score += 2
            if m.promotion is not None:
                score += 1.5
            return score

        moves.sort(key=move_ordering, reverse=True)

        for mv in moves:
            board.push(mv)
            val = self._minmax(board, self.depth - 1, alpha, beta, not board.turn)
            board.pop()
            if val > best_value:
                best_value = val
                best_move = mv
            if val > alpha:
                alpha = val
        return best_move
