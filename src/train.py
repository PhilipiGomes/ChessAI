# trunk-ignore-all(git-diff-check/error)
import argparse
import os
import random
from typing import List, Optional, Tuple

import chess
import numpy as np
import pandas as pd
import json

from .chessAI import INPUT_SIZE, MATE_SCORE, SimpleMLP, board_to_feature_vector

# --- helpers para ler CSV com avaliações ---


def parse_eval_raw(v) -> Optional[float]:
    """
    Os tipos de avaliação no CSV são:
        - string representando float (centipawns), ex: +345, -12, 0
        - string representando mate, ex: #+3, #-2
    """
    if v is None:
        return None
    try:
        s = str(v).strip()
        # mate em notação "#N"
        if s.startswith("#"):
            try:
                n = int(s[1:]) if len(s) > 1 else 0
            except Exception:
                n = 0
            return float((MATE_SCORE + abs(n)) * -1 if n < 0 else 1)

        val = float(s)
        return val / 100.0
    except Exception:
        return None


def load_positions_from_csv(
    path: str, max_rows: Optional[int] = None, random_seed: int = 42
) -> List[Tuple[str, float]]:
    """
    Lê CSV e retorna até `max_rows` posições válidas (fen, score) **sem** avaliações de mate.
    Se max_rows for fornecido, garante retornar exatamente esse número — caso não haja posições
    suficientes sem mate no arquivo, lança ValueError.
    """
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError("CSV deve ter pelo menos 2 colunas (fen, eval).")

    eval_col = df.columns[1]

    # Detecta rapidamente linhas cujo texto de avaliação indica mate (# ou 'mate')
    eval_text = df[eval_col].astype(str).fillna("")
    is_mate_mask = eval_text.str.contains(r"#|mate", case=False, regex=True)

    # candidatos sem mate
    candidates = df.loc[~is_mate_mask]

    if max_rows is None:
        # queremos todas as posições não-mate — iterar e validar fen
        df_iter = candidates.itertuples(index=False, name=None)
    else:
        # embaralha candidatos e assegura que há pelo menos max_rows possíveis
        if len(candidates) < max_rows:
            raise ValueError(
                f"Não há posições suficientes sem mate no CSV ({len(candidates)} < {max_rows})."
            )
        candidates = candidates.sample(frac=1, random_state=random_seed)
        df_iter = candidates.itertuples(index=False, name=None)

    positions: List[Tuple[str, float]] = []
    skipped = 0
    parse = parse_eval_raw  # local ref para velocidade

    for row in df_iter:
        fen = row[0]
        raw_eval = row[1]
        score = parse(raw_eval)
        # parse_eval_raw pode devolver MATE_SCORE para formatos incomuns — rejeitamos também nesses casos
        if score is None:
            skipped += 1
            continue
        # valida FEN
        try:
            board = chess.Board(str(fen))
        except Exception:
            skipped += 1
            continue
        positions.append((board.fen(), float(score)))
        if max_rows is not None and len(positions) >= max_rows:
            break

    if max_rows is not None and len(positions) < max_rows:
        # caso improvável: candidatos suficientes mas muitos inválidos; reportar claramente
        raise ValueError(
            f"Não foi possível coletar {max_rows} posições válidas sem mate. Coletadas: {len(positions)} (skipped {skipped})."
        )

    print(f"Carregou {len(positions)} posições (pulou {skipped}) de {path}")
    return positions


# --- training pipeline ---


def build_dataset(positions: List[Tuple[str, float]]) -> Tuple[np.ndarray, np.ndarray]:
    X = np.zeros((len(positions), INPUT_SIZE), dtype=np.float32)
    y = np.zeros((len(positions),), dtype=np.float32)
    for i, (fen, score) in enumerate(positions):
        board = chess.Board(fen)
        X[i] = board_to_feature_vector(board)
        y[i] = float(score)
    return X, y


# --- utilidades para evolução de arquitetura ---
def random_arch(
    min_layers: int,
    max_layers: int,
    min_neurons: int,
    max_neurons: int,
    rng: random.Random,
) -> List[int]:
    nl = rng.randint(min_layers, max_layers)
    return [rng.randint(min_neurons, max_neurons) for _ in range(nl)]


def mutate_arch(
    arch: List[int],
    min_layers: int,
    max_layers: int,
    min_neurons: int,
    max_neurons: int,
    rng: random.Random,
    mutation_rate: float,
) -> List[int]:
    new = arch.copy()
    # with prob add/remove layer
    if rng.random() < mutation_rate:
        if len(new) < max_layers and rng.random() < 0.5:
            # add layer at random position
            pos = rng.randrange(0, len(new) + 1)
            new.insert(pos, rng.randint(min_neurons, max_neurons))
        elif len(new) > min_layers:
            pos = rng.randrange(0, len(new))
            new.pop(pos)
    # mutate sizes
    for i in range(len(new)):
        if rng.random() < mutation_rate:
            change = rng.randint(-max(1, new[i] // 4), max(1, new[i] // 4))
            new[i] = int(np.clip(new[i] + change, min_neurons, max_neurons))
    return new


def evaluate_architecture(
    arch: List[int],
    X: np.ndarray,
    y: np.ndarray,
    epochs_eval: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> float:
    """
    Treina por epochs_eval rapidamente e retorna loss de validação (MSE * 0.5).
    Faz split simples train/val 80/20.
    """
    N = X.shape[0]
    if N < 5:
        return float("inf")
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    split = int(N * 0.8)
    train_idx = perm[:split]
    val_idx = perm[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    model = SimpleMLP(input_size=INPUT_SIZE, hidden_sizes=arch, seed=seed)
    model.train_sgd(
        X_train,
        y_train,
        epochs=epochs_eval,
        lr=lr,
        batch_size=batch_size,
        verbose=False,
    )
    preds = model.predict_batch(X_val)
    mse = float(np.mean((preds - y_val) ** 2) * 0.5)
    return mse


def evolve_architectures(
    X: np.ndarray,
    y: np.ndarray,
    pop: int,
    generations: int,
    epochs_eval: int,
    min_layers: int,
    max_layers: int,
    min_neurons: int,
    max_neurons: int,
    mutation_rate: float,
    lr: float,
    batch_size: int,
    seed: int,
) -> List[int] | None:
    from tqdm import tqdm

    # trunk-ignore(bandit/B311)
    rng = random.Random(seed)
    # initial population
    population = [
        random_arch(min_layers, max_layers, min_neurons, max_neurons, rng)
        for _ in range(pop)
    ]
    best_arch = None
    best_score = float("inf")
    progress_bar = tqdm(range(generations), desc="Evoluindo arquitetura")
    for gen in progress_bar:
        scores = []
        for i, arch in enumerate(population):
            score = evaluate_architecture(
                arch,
                X,
                y,
                epochs_eval=epochs_eval,
                lr=lr,
                batch_size=batch_size,
                seed=seed + gen * 100 + i,
            )
            scores.append((score, arch))
        scores.sort(key=lambda s: s[0])
        # keep elite
        elite_count = 3 if pop > 10 else 1
        elites = [arch for (_, arch) in scores[:elite_count]]
        if scores[0][0] < best_score:
            best_score = scores[0][0]
            best_arch = scores[0][1]
        # create next generation: keep elites + mutated children
        next_pop = elites.copy()
        while len(next_pop) < pop:
            parent = rng.choice(elites)
            child = mutate_arch(
                parent,
                min_layers,
                max_layers,
                min_neurons,
                max_neurons,
                rng,
                mutation_rate,
            )
            next_pop.append(child)
        population = next_pop
        progress_bar.set_postfix(
            best_loss=f"{best_score:.6f}", best_arch=str(best_arch)
        )
    print(f"[evolve] Melhor arquitetura: {best_arch}, perda geral: {best_score:.6f}")
    if best_arch is not None:
        return best_arch


# --- funções de salvar/carregar (mantive as suas) ---
def save_model_np(model: SimpleMLP, prefix: str):
    os.makedirs(prefix, exist_ok=True)
    np.save(
        os.path.join(prefix, "chess_mlp_W.npy"),
        np.array(model.W, dtype=object),
        allow_pickle=True,
    )
    np.save(
        os.path.join(prefix, "chess_mlp_b.npy"),
        np.array(model.b, dtype=object),
        allow_pickle=True,
    )
    meta = getattr(model, "sizes", None)
    if meta is not None:
        with open(os.path.join(prefix, "chess_mlp_meta.json"), "w") as f:
            json_obj = {"sizes": meta}
            # use json.dump to ensure valid JSON with double quotes
            json.dump(json_obj, f)


def load_model_np(model: SimpleMLP, prefix: str):
    W = np.load(os.path.join(prefix, "chess_mlp_W.npy"), allow_pickle=True)
    b = np.load(os.path.join(prefix, "chess_mlp_b.npy"), allow_pickle=True)
    model.W = [w.astype(np.float32) for w in list(W)]
    model.b = [bb.astype(np.float32) for bb in list(b)]


def plot(losses: dict, filename: str):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(list(losses.keys()), list(losses.values()))
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


# --- main CLI ---
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-out", default=os.path.join('src', 'model'))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None)
    # evolution args
    parser.add_argument(
        "--evolve", action="store_true", help="ativa busca evolutiva pela arquitetura"
    )
    parser.add_argument("--pop", type=int, default=10)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument(
        "--epochs-eval",
        type=int,
        default=2,
        help="épocas rápidas por avaliação de arquitetura",
    )
    parser.add_argument("--min-layers", type=int, default=1)
    parser.add_argument("--max-layers", type=int, default=4)
    parser.add_argument("--min-neurons", type=int, default=16)
    parser.add_argument("--max-neurons", type=int, default=512)
    parser.add_argument("--mutation-rate", type=float, default=0.25)
    parser.add_argument(
        "--plot-loss",
        action="store_true",
        help="Plota e salva o gráfico da evolução de perdas",
    )
    args = parser.parse_args(argv)

    # move the current model to the folder 'src/prev_model', if there's a model saved in model-out
    prev_dir = os.path.join('src', 'prev_model')
    if os.path.exists(args.model_out) and len(os.listdir(args.model_out)) > 0:
        if not os.path.exists(prev_dir):
            os.makedirs(prev_dir)
        for f in os.listdir(prev_dir):
            try:
                os.remove(os.path.join(prev_dir, f))
            except Exception:
                pass
        for f in os.listdir(args.model_out):
            src = os.path.join(args.model_out, f)
            dst = os.path.join(prev_dir, f)
            if os.path.isfile(src) or os.path.isdir(src):
                try:
                    # replace files/dirs into prev_model
                    if os.path.exists(dst):
                        if os.path.isdir(dst):
                            import shutil

                            if os.path.isdir(dst):
                                shutil.rmtree(dst)
                            else:
                                os.remove(dst)
                    os.replace(src, dst)
                except Exception:
                    try:
                        import shutil

                        shutil.copytree(src, dst)
                    except Exception:
                        pass
        print(f"Modelo existente em {args.model_out}/ foi movido para {prev_dir}/")

    print("Carregando posições do CSV...")
    positions = load_positions_from_csv(args.data, max_rows=args.max_rows)
    if len(positions) == 0:
        print("Nenhuma posição válida encontrada. Saindo.")
        return
    print("Construindo dataset...")
    X, y = build_dataset(positions)

    chosen_hidden = args.hidden
    if args.evolve:
        print("Iniciando evolução de arquiteturas...")
        best = evolve_architectures(
            X,
            y,
            pop=args.pop,
            generations=args.generations,
            epochs_eval=args.epochs_eval,
            min_layers=args.min_layers,
            max_layers=args.max_layers,
            min_neurons=args.min_neurons,
            max_neurons=args.max_neurons,
            mutation_rate=args.mutation_rate,
            lr=args.lr,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        if best is not None:
            chosen_hidden = best

    print("Arquitetura escolhida (hidden sizes):", chosen_hidden)
    model = SimpleMLP(input_size=INPUT_SIZE, hidden_sizes=chosen_hidden, seed=args.seed)
    print(
        f"Iniciando treino final: epochs: {args.epochs}, batch size: {args.batch_size}, lr: {args.lr}"
    )

    losses = model.train_sgd(
        X,
        y,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        verbose=True,
        plot=args.plot_loss,
    )

    save_model_np(model, args.model_out)
    if args.plot_loss:
        if losses is not None:
            plot(losses, "training_loss.png")

    print("Treino concluído.")


if __name__ == "__main__":
    main()
