import datetime
import json
import os
import threading
import time

import chess
import numpy as np
from flask import Flask, jsonify, render_template, request

from src.chessAI import ChessAI, SimpleMLP
from src.train import build_dataset, load_positions_from_csv

# Project uses `src/templates` and `src/static` after reorganization
app = Flask(__name__, template_folder="src/templates", static_folder="src/static")

# Simple game state kept in memory per server run.
# For a real app you'd use sessions or persistent storage.
GAME = {
    "board": chess.Board(),
    "moves": [],
    "mode": "pve",  # pvp, pve, eve
    "ai_depth": 2,
    "ai1_depth": 2,
    "ai2_depth": 2,
    "human_color": chess.WHITE,
    "game_running": False,
    "ai_delay": 0.4,  # seconds between AI moves
}

# AI-vs-AI runner control
EVE_THREAD = None
EVE_STOP = threading.Event()

# Training background runner state
TRAIN_THREAD = None
TRAIN_STOP = threading.Event()
TRAIN_STATE = {
    "running": False,
    "epoch": 0,
    "epochs": 0,
    "loss": None,
    "best_loss": None,
    "history": [],  # list of {epoch, loss, best_loss}
    "params": {},
}


# cached in-memory model (SimpleMLP instance) to avoid reloading from disk every time
CACHED_MODEL = None
CACHED_MODEL_PREFIX = None

# model directory root (note: models live under src/model)
MODEL_DIR = os.path.join("src", "model")
PREV_MODEL_DIR = os.path.join("src", "prev_model")


def set_loaded_model(prefix: str):
    """Load model weights into an in-memory SimpleMLP and set as cached model.
    Returns True on success, False on failure (and leaves previous cache intact).
    """
    global CACHED_MODEL, CACHED_MODEL_PREFIX
    try:
        # create a temporary SimpleMLP; load() will overwrite sizes and weights
        tmp = SimpleMLP([128, 64])
        tmp.load(prefix)
        CACHED_MODEL = tmp
        CACHED_MODEL_PREFIX = prefix
        return True
    except Exception:
        return False


def load_ai(depth=2, sequence=None):
    # if we have a cached model already loaded, reuse it
    if CACHED_MODEL is not None and CACHED_MODEL_PREFIX is not None:
        try:
            ai = ChessAI(
                sequence=sequence or GAME["moves"], model=CACHED_MODEL, depth=depth
            )
            return ai
        except Exception:
            pass

    # fallback: construct AI and try to load from disk (src/model/latest, src/model, src/prev_model)
    ai = ChessAI(sequence=sequence or GAME["moves"], depth=depth)
    # prefer src/model/latest then src/model then src/prev_model
    candidates = [os.path.join(MODEL_DIR, "latest"), MODEL_DIR, PREV_MODEL_DIR]
    for cand in candidates:
        if os.path.isdir(cand) and len(os.listdir(cand)) > 0:
            try:
                ai.load_model(cand)
                break
            except Exception:
                continue
    return ai


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/new_game", methods=["POST"])
def new_game():
    data = request.get_json() or {}
    mode = data.get("mode", "pve")
    ai_depth = int(data.get("ai_depth", 2))
    # optional ai delay in seconds (float)
    try:
        ai_delay = float(data.get("ai_delay", GAME.get("ai_delay", 0.4)))
    except Exception:
        ai_delay = GAME.get("ai_delay", 0.4)
    ai1_depth = int(data.get("ai1_depth", data.get("ai_depth", 2)))
    ai2_depth = int(data.get("ai2_depth", data.get("ai_depth", 2)))
    # if a game is already running, do not allow changing AI depth
    if GAME.get("game_running"):
        ai_depth = GAME.get("ai_depth", ai_depth)

    GAME["mode"] = mode
    GAME["ai_depth"] = ai_depth
    GAME["ai_delay"] = ai_delay
    GAME["ai1_depth"] = ai1_depth
    GAME["ai2_depth"] = ai2_depth
    GAME["board"] = chess.Board()
    GAME["moves"] = []
    GAME["game_running"] = True

    # randomize side in pve: sometimes human plays black or white
    if mode == "pve":
        # choose human color randomly
        import random

        human_white = random.choice([True, False])
        GAME["human_color"] = chess.WHITE if human_white else chess.BLACK
        # if human is black, AI (white) should play first
        if GAME["human_color"] == chess.BLACK:
            ai = load_ai(depth=GAME.get("ai_depth", 2))
            mv_ai = ai.choose_move(GAME["board"])
            if mv_ai:
                try:
                    san_ai = GAME["board"].san(mv_ai)
                except Exception:
                    san_ai = None
                # delay slightly to simulate thinking / allow client to show move timing
                time.sleep(float(GAME.get("ai_delay", 0.4)))
                GAME["board"].push(mv_ai)
                GAME["moves"].append(san_ai if san_ai is not None else mv_ai.uci())
    elif mode == "pvp":
        GAME["human_color"] = chess.WHITE
    elif mode == "eve":
        # AI vs AI: no human
        GAME["human_color"] = None

    # stop eve thread if running
    global EVE_THREAD, EVE_STOP
    EVE_STOP.set()
    if EVE_THREAD is not None:
        EVE_THREAD.join(timeout=0.1)
    EVE_STOP.clear()

    return jsonify(
        {
            "fen": GAME["board"].fen(),
            "moves": GAME["moves"],
            "mode": GAME["mode"],
            "ai_depth": GAME["ai_depth"],
            "human_color": (
                "w"
                if GAME["human_color"] == chess.WHITE
                else ("b" if GAME["human_color"] == chess.BLACK else None)
            ),
        }
    )


def eve_runner(ai1_depth: int, ai2_depth: int, stop_event: threading.Event):
    """Background runner for AI vs AI. Alternates moves until game over or stop_event set."""
    # create AI instances once so they keep TT and opening state across moves
    ai_white = load_ai(depth=ai1_depth)
    ai_black = load_ai(depth=ai2_depth)
    while not stop_event.is_set() and not GAME["board"].is_game_over():
        # white move
        mv = ai_white.choose_move(GAME["board"])
        if mv:
            try:
                san = GAME["board"].san(mv)
            except Exception:
                san = None
            GAME["board"].push(mv)
            GAME["moves"].append(san if san is not None else mv.uci())
        else:
            break
        if stop_event.is_set() or GAME["board"].is_game_over():
            break
        # respect configured delay
        time.sleep(float(GAME.get("ai_delay", 0.4)))
        # black move
        mv2 = ai_black.choose_move(GAME["board"])
        if mv2:
            try:
                san2 = GAME["board"].san(mv2)
            except Exception:
                san2 = None
            GAME["board"].push(mv2)
            GAME["moves"].append(san2 if san2 is not None else mv2.uci())
        else:
            break
        time.sleep(float(GAME.get("ai_delay", 0.4)))


@app.route("/api/eve_start", methods=["POST"])
def eve_start():
    data = request.get_json() or {}
    ai1 = int(data.get("ai1_depth", GAME.get("ai1_depth", 2)))
    ai2 = int(data.get("ai2_depth", GAME.get("ai2_depth", 2)))
    try:
        ai_delay = float(data.get("ai_delay", GAME.get("ai_delay", 0.4)))
    except Exception:
        ai_delay = GAME.get("ai_delay", 0.4)
    # ensure board is new or reset
    GAME["board"] = chess.Board()
    GAME["moves"] = []
    GAME["mode"] = "eve"
    GAME["ai1_depth"] = ai1
    GAME["ai2_depth"] = ai2
    GAME["ai_delay"] = ai_delay
    GAME["game_running"] = True

    global EVE_THREAD, EVE_STOP
    EVE_STOP.clear()
    EVE_THREAD = threading.Thread(
        target=eve_runner, args=(ai1, ai2, EVE_STOP), daemon=True
    )
    EVE_THREAD.start()
    return jsonify({"status": "started"})


def train_runner(
    epochs: int, lr: float, data_feed_size: int, stop_event: threading.Event
):
    # Real training: load dataset and run SimpleMLP.train_sgd
    TRAIN_STATE["running"] = True
    TRAIN_STATE["epoch"] = 0
    TRAIN_STATE["epochs"] = epochs
    TRAIN_STATE["history"] = []
    TRAIN_STATE["best_loss"] = None
    # try to load positions from CSV source if provided in params
    params = TRAIN_STATE.get("params", {})
    csv_path = params.get("csv_path") or "src/data/random_evals.csv"
    try:
        positions = load_positions_from_csv(csv_path, max_rows=data_feed_size)
        X, y = build_dataset(positions)
    except Exception as ex:
        # fallback: empty dataset -> abort
        TRAIN_STATE["running"] = False
        TRAIN_STATE["error"] = str(ex)
        return

    # validation split
    val_split = (
        float(params.get("val_split", 0.0))
        if params.get("val_split") is not None
        else 0.0
    )
    try:
        val_split = max(0.0, min(0.9, float(val_split)))
    except Exception:
        val_split = 0.0
    N = X.shape[0]
    if N == 0:
        TRAIN_STATE["running"] = False
        TRAIN_STATE["error"] = "empty dataset"
        return
    if val_split > 0.0 and N > 1:
        # shuffle indices and split
        idx = np.random.permutation(N)
        n_val = max(1, int(N * val_split))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        X_val = X[val_idx]
        y_val = y[val_idx]
        X_train = X[train_idx]
        y_train = y[train_idx]
    else:
        X_train, y_train = X, y
        X_val = None
        y_val = None

    # create model
    hidden_layers = params.get("hidden_layers") or [128, 64]
    if isinstance(hidden_layers, str):
        try:
            hidden_layers = [
                int(x.strip()) for x in hidden_layers.split(",") if x.strip()
            ]
        except Exception:
            hidden_layers = [128, 64]

    model = SimpleMLP(hidden_layers)

    # prepare checkpoint directory
    run_id = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    ckpt_dir = os.path.join(MODEL_DIR, run_id)
    os.makedirs(ckpt_dir, exist_ok=True)
    TRAIN_STATE["last_ckpt"] = ckpt_dir

    best_loss = None

    def progress_cb(epoch_num, epoch_loss):
        nonlocal best_loss
        TRAIN_STATE["epoch"] = int(epoch_num)
        TRAIN_STATE["loss"] = float(epoch_loss)
        if best_loss is None or epoch_loss < best_loss:
            best_loss = epoch_loss
            TRAIN_STATE["best_loss"] = float(best_loss)
            # save checkpoint when improvement
            try:
                model.save(ckpt_dir)
                # also write meta json
                with open(os.path.join(ckpt_dir, "train_meta.json"), "w") as f:
                    json.dump(
                        {"epoch": epoch_num, "loss": epoch_loss, "params": params}, f
                    )
            except Exception:
                pass
        # compute validation loss if available
        val_loss = None
        try:
            if X_val is not None and X_val.shape[0] > 0:
                preds = model.predict_batch(X_val)
                err = preds - y_val
                val_loss = 0.5 * float(np.mean(err**2))
                TRAIN_STATE["validation_loss"] = float(val_loss)
        except Exception:
            val_loss = None

        TRAIN_STATE["history"].append(
            {
                "epoch": int(epoch_num),
                "loss": float(epoch_loss),
                "best_loss": float(TRAIN_STATE.get("best_loss", epoch_loss)),
                "validation_loss": (float(val_loss) if val_loss is not None else None),
            }
        )
        # allow early stop
        if stop_event.is_set():
            raise KeyboardInterrupt()

    try:
        model.train_sgd(
            X_train,
            y_train,
            epochs=epochs,
            lr=lr,
            batch_size=int(params.get("batch_size", 32)),
            verbose=False,
            plot=False,
            progress_callback=progress_cb,
        )
    except KeyboardInterrupt:
        # stop requested
        pass
    except Exception as ex:
        TRAIN_STATE["error"] = str(ex)

    # final save
    try:
        model.save(ckpt_dir)
    except Exception:
        pass

    # persist history to file
    try:
        with open(os.path.join(ckpt_dir, "train_history.json"), "w") as f:
            json.dump(TRAIN_STATE.get("history", []), f)
    except Exception:
        pass

    TRAIN_STATE["running"] = False


@app.route("/api/train_save_model", methods=["POST"])
def train_save_model():
    # copy last checkpoint to model/latest
    src_ckpt = TRAIN_STATE.get("last_ckpt")
    if not src_ckpt or not os.path.isdir(src_ckpt):
        return jsonify({"error": "no checkpoint available"}), 400
    dst = os.path.join(MODEL_DIR, "latest")
    # remove dst if exists
    try:
        if os.path.exists(dst):
            import shutil

            if os.path.isdir(dst):
                shutil.rmtree(dst)
            else:
                os.remove(dst)
        import shutil

        shutil.copytree(src_ckpt, dst)
    except Exception as ex:
        return jsonify({"error": str(ex)}), 500
    return jsonify({"status": "saved", "path": dst})


@app.route("/api/train_start", methods=["POST"])
def train_start():
    data = request.get_json() or {}
    # server-side validation / sanitization
    try:
        epochs = int(data.get("epochs", 50))
        epochs = max(1, min(1000, epochs))
    except Exception:
        return jsonify({"error": "invalid epochs"}), 400
    try:
        lr = float(data.get("lr", 0.01))
        lr = max(1e-6, min(1.0, lr))
    except Exception:
        return jsonify({"error": "invalid lr"}), 400
    try:
        data_feed_size = int(data.get("data_feed_size", 1000))
        data_feed_size = max(10, min(200000, data_feed_size))
    except Exception:
        return jsonify({"error": "invalid data_feed_size"}), 400

    batch_size = (
        int(data.get("batch_size", 32))
        if str(data.get("batch_size", 32)).isdigit()
        else 32
    )
    batch_size = max(1, min(65536, batch_size))
    hidden_layers = data.get("hidden_layers", data.get("hidden", [128, 64]))
    if isinstance(hidden_layers, str):
        try:
            hidden_layers = [
                int(x.strip()) for x in hidden_layers.split(",") if x.strip()
            ]
        except Exception:
            hidden_layers = [128, 64]
    # csv path: only allow files under src/data/ for safety
    csv_path = data.get("csv_path") or "src/data/random_evals.csv"
    if ".." in csv_path or csv_path.startswith("/") or "\\" in csv_path:
        return jsonify({"error": "invalid csv_path"}), 400

    if TRAIN_STATE.get("running"):
        return jsonify({"status": "already_running"}), 400

    params = {
        "epochs": epochs,
        "lr": lr,
        "data_feed_size": data_feed_size,
        "val_split": float(data.get("val_split", 0.0)),
        "batch_size": batch_size,
        "hidden_layers": hidden_layers,
        "csv_path": csv_path,
    }
    TRAIN_STATE["params"] = params
    TRAIN_STOP.clear()
    global TRAIN_THREAD
    TRAIN_THREAD = threading.Thread(
        target=train_runner, args=(epochs, lr, data_feed_size, TRAIN_STOP), daemon=True
    )
    TRAIN_THREAD.start()
    return jsonify({"status": "started", "params": params})


@app.route("/api/models", methods=["GET"])
def models_list():
    base = MODEL_DIR
    models = []
    if os.path.isdir(base):
        for name in sorted(os.listdir(base), reverse=True):
            path = os.path.join(base, name)
            if os.path.isdir(path):
                meta = None
                history = None
                try:
                    mfile = os.path.join(path, "train_meta.json")
                    if os.path.exists(mfile):
                        with open(mfile, "r") as f:
                            mm = json.load(f)
                            meta = mm
                except Exception:
                    meta = None
                try:
                    hfile = os.path.join(path, "train_history.json")
                    if os.path.exists(hfile):
                        with open(hfile, "r") as f:
                            history = json.load(f)
                except Exception:
                    history = None
                models.append(
                    {"name": name, "path": path, "meta": meta, "history": history}
                )
    return jsonify({"models": models})


@app.route("/api/models/load", methods=["POST"])
def models_load():
    data = request.get_json() or {}
    name = data.get("name")
    if not name:
        return jsonify({"error": "missing name"}), 400
    src = os.path.join(MODEL_DIR, name)
    if not os.path.isdir(src):
        return jsonify({"error": "not found"}), 404
    dst = os.path.join(MODEL_DIR, "latest")
    try:
        import shutil

        # If the user asked to load the folder already named 'latest', don't attempt to delete and copy
        # (deleting dst would remove the src and cause WinError 3 on Windows).
        if os.path.abspath(src) == os.path.abspath(dst) or name == "latest":
            # just attempt to load into memory from src
            loaded = set_loaded_model(src)
            if not loaded:
                return jsonify({"error": "could not load model into memory"}), 500
            return jsonify({"status": "loaded", "path": src})

        # otherwise, copy selected model into model/latest (replacing existing latest)
        if os.path.exists(dst):
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            else:
                os.remove(dst)
        shutil.copytree(src, dst)
        # attempt to load into memory cache as well
        loaded = set_loaded_model(dst)
        if not loaded:
            # try loading directly from src as fallback
            set_loaded_model(src)
    except Exception as ex:
        return jsonify({"error": str(ex)}), 500
    return jsonify({"status": "loaded", "path": dst})


@app.route("/api/train_status", methods=["GET"])
def train_status():
    return jsonify(
        {
            "running": TRAIN_STATE.get("running", False),
            "epoch": TRAIN_STATE.get("epoch", 0),
            "epochs": TRAIN_STATE.get("epochs", 0),
            "loss": TRAIN_STATE.get("loss"),
            "best_loss": TRAIN_STATE.get("best_loss"),
            "history": TRAIN_STATE.get("history", []),
            "params": TRAIN_STATE.get("params", {}),
        }
    )


@app.route("/api/train_stop", methods=["POST"])
def train_stop():
    TRAIN_STOP.set()
    if TRAIN_THREAD is not None:
        TRAIN_THREAD.join(timeout=1.0)
    TRAIN_STATE["running"] = False
    return jsonify({"status": "stopped"})


@app.route("/api/eve_stop", methods=["POST"])
def eve_stop():
    global EVE_STOP, EVE_THREAD
    EVE_STOP.set()
    if EVE_THREAD is not None:
        EVE_THREAD.join(timeout=1.0)
    GAME["game_running"] = False
    return jsonify({"status": "stopped"})


@app.route("/api/move", methods=["POST"])
def make_move():
    data = request.get_json() or {}
    uci = data.get("uci")
    player = data.get("player", "human")
    board: chess.Board = GAME["board"]
    if not uci:
        return jsonify({"error": "missing uci"}), 400
    try:
        mv = chess.Move.from_uci(uci)
    except Exception:
        return jsonify({"error": "invalid uci"}), 400
    if mv not in board.legal_moves:
        return jsonify({"error": "illegal move"}), 400
    # compute SAN before pushing (san expects move legal in current position)
    try:
        san = board.san(mv)
    except Exception:
        return jsonify({"error": "could not compute SAN for move"}), 500
    board.push(mv)
    GAME["moves"].append(san)

    # Optionally let AI respond
    ai_reply = None
    if data.get("ai_reply", True):
        ai = load_ai()
        mv_ai = ai.choose_move(board)
        if mv_ai:
            try:
                san_ai = board.san(mv_ai)
            except Exception:
                # if for some reason SAN fails, fall back to UCI only
                san_ai = None
            # simulate AI delay before pushing reply
            try:
                time.sleep(float(GAME.get("ai_delay", 0.4)))
            except Exception:
                pass
            board.push(mv_ai)
            GAME["moves"].append(san_ai if san_ai is not None else mv_ai.uci())
            ai_reply = {"uci": mv_ai.uci(), "san": san_ai}

    return jsonify({"fen": board.fen(), "moves": GAME["moves"], "ai": ai_reply})


@app.route("/api/ai_move", methods=["POST"])
def ai_move():
    board: chess.Board = GAME["board"]
    if board.is_game_over():
        return jsonify({"error": "game over"}), 400
    ai = load_ai(depth=GAME.get("ai_depth", 2))
    mv_ai = ai.choose_move(board)
    if not mv_ai:
        return jsonify({"error": "no move found"}), 400
    try:
        san_ai = board.san(mv_ai)
    except Exception:
        san_ai = None
    board.push(mv_ai)
    GAME["moves"].append(san_ai if san_ai is not None else mv_ai.uci())
    return jsonify(
        {
            "fen": board.fen(),
            "moves": GAME["moves"],
            "ai": {"uci": mv_ai.uci(), "san": san_ai},
        }
    )


@app.route("/api/legal_moves", methods=["GET"])
def legal_moves():
    sq = request.args.get("from") or request.args.get("square")
    board: chess.Board = GAME["board"]
    if not sq:
        return jsonify({"error": "missing square"}), 400
    legal = []
    for mv in board.legal_moves:
        if chess.square_name(mv.from_square) == sq:
            legal.append(mv.uci())
    return jsonify(
        {
            "moves": legal,
            "fen": board.fen(),
            "turn": "w" if board.turn == chess.WHITE else "b",
        }
    )


@app.route("/api/undo", methods=["POST"])
def undo():
    board: chess.Board = GAME["board"]
    if len(GAME["moves"]) == 0:
        return jsonify({"error": "no moves to undo"}), 400
    # Pop last move
    try:
        board.pop()
        GAME["moves"].pop()
    except Exception:
        return jsonify({"error": "cannot undo"}), 400
    return jsonify({"fen": board.fen(), "moves": GAME["moves"]})


@app.route("/train", methods=["GET", "POST"])
def train_page():
    # minimal train page: GET shows a small form; POST starts training in background (stub)
    if request.method == "POST":
        params = request.form.to_dict()
        # For now, just echo params and don't run heavy training synchronously
        return jsonify({"status": "started", "params": params})
    return render_template("train.html")


@app.route("/models", methods=["GET"])
def models_page():
    return render_template("models.html")


@app.route("/api/status", methods=["GET"])
def status():
    board: chess.Board = GAME["board"]
    if board.is_game_over():
        res = board.result()
        if board.is_checkmate():
            return jsonify({"game_over": True, "result": res, "reason": "checkmate"})
        else:
            return jsonify({"game_over": True, "result": res, "reason": "draw"})
    return jsonify({"game_over": False})


if __name__ == "__main__":
    app.run(debug=True)
