import argparse
import json
import os
import random
from typing import List, Tuple

import chess
import numpy as np

from chessAI import ChessAI, SimpleMLP


def play_game(ai1: ChessAI, ai2: ChessAI, max_moves: int = 200) -> Tuple[int, int]:
    """
    Play a game between two AIs.
    Returns (score_ai1, score_ai2) where 1 = win, 0.5 = draw, 0 = loss
    """
    board = chess.Board()
    move_count = 0

    while not board.is_game_over() and move_count < max_moves:
        if board.turn == chess.WHITE:
            move = ai1.choose_move(board)
        else:
            move = ai2.choose_move(board)

        if move is None:
            break

        board.push(move)
        move_count += 1

    # Score based on game outcome
    if board.is_checkmate():
        # White checkmated = White wins
        return (1.0, 0.0) if not board.turn else (0.0, 1.0)
    elif board.is_stalemate() or board.is_insufficient_material():
        return (0.5, 0.5)
    else:
        # Max moves reached - draw
        return (0.5, 0.5)


def evaluate_tournament(
    population: List[ChessAI], games_per_pair: int = 2
) -> List[float]:
    """
    Run tournament and return scores for each AI.
    Each AI plays against every other AI.
    """
    n = len(population)
    scores = [0.0] * n

    # Round-robin tournament
    for i in range(n):
        for j in range(i + 1, n):
            for _ in range(games_per_pair):
                score_i, score_j = play_game(population[i], population[j])
                scores[i] += score_i
                scores[j] += score_j

    return scores


def mutate_model(model: SimpleMLP, mutation_rate: float = 0.1) -> SimpleMLP:
    """
    Create a mutated copy of the model by adding Gaussian noise to weights,
    and potentially mutating the architecture (hidden layer sizes).
    """
    hidden_sizes = [s for s in model.sizes[1:-1]]

    # Mutate architecture with lower probability
    if random.random() < mutation_rate * 0.5:
        # Add or remove a layer
        if random.random() < 0.5 and len(hidden_sizes) > 1:
            # Remove a random hidden layer
            hidden_sizes.pop(random.randint(0, len(hidden_sizes) - 1))
        else:
            # Add a new random hidden layer
            new_size = random.randint(16, 512)
            insert_pos = random.randint(0, len(hidden_sizes))
            hidden_sizes.insert(insert_pos, new_size)

    # Mutate neuron counts in existing layers
    for i in range(len(hidden_sizes)):
        if random.random() < mutation_rate * 0.3:
            # Adjust layer size by ±10% to ±50%
            adjustment = random.randint(-50, 50)
            hidden_sizes[i] = max(8, hidden_sizes[i] + adjustment)

    mutated = SimpleMLP(
        hidden_sizes=hidden_sizes,
        input_size=model.sizes[0],
        seed=None,
    )

    # Copy and mutate weights (handle size mismatches from architecture changes)
    for i in range(min(len(model.W), len(mutated.W))):
        shape = mutated.W[i].shape
        if model.W[i].shape == shape:
            mutated.W[i] = model.W[i].copy()
        # If shapes don't match due to architecture change, keep random initialization

        if random.random() < mutation_rate:
            noise = np.random.normal(0, 0.1 * np.std(mutated.W[i]), mutated.W[i].shape)
            mutated.W[i] = (mutated.W[i] + noise).astype(np.float32)

    # Copy and mutate biases
    for i in range(min(len(model.b), len(mutated.b))):
        if model.b[i].shape == mutated.b[i].shape:
            mutated.b[i] = model.b[i].copy()

        if random.random() < mutation_rate:
            noise = np.random.normal(0, 0.01, mutated.b[i].shape)
            mutated.b[i] = (mutated.b[i] + noise).astype(np.float32)

    return mutated


def save_tournament_results(
    population: List[ChessAI], scores: List[float], prefix: str, generation: int
):
    """
    Save tournament results and best model.
    """
    os.makedirs(prefix, exist_ok=True)
    os.makedirs(os.path.join(prefix, f"generation_{generation:04d}_best"), exist_ok=True)

    # Save best AI
    best_idx = np.argmax(scores)
    best_ai = population[best_idx]
    best_dir = os.path.join(prefix, f"generation_{generation:04d}_best")
    best_ai.model.save(best_dir)

    # Save tournament results
    results = {
        "generation": generation,
        "scores": [float(s) for s in scores],
        "best_score": float(scores[best_idx]),
        "best_idx": int(best_idx),
        "architectures": [best_ai.model.sizes for ai in population],
    }

    with open(os.path.join(prefix, f"tournament_gen_{generation:04d}.json"), "w") as f:
        json.dump(results, f, indent=2)


def initialize_population(
    pop_size: int, hidden_layer_options: List[List[int]], seed: int = 42
) -> List[ChessAI]:
    """
    Initialize population with different architectures.
    """
    rng = random.Random(seed)
    population = []

    for i in range(pop_size):
        # Choose architecture (cycle through options or random)
        if i < len(hidden_layer_options):
            arch = hidden_layer_options[i]
        else:
            arch = rng.choice(hidden_layer_options)

        model = SimpleMLP(hidden_sizes=arch, seed=seed + i)
        ai = ChessAI(sequence=[], model=model, depth=2, zobrist_seed=seed + i)
        population.append(ai)

    return population


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Tournament-based training for Chess AI"
    )
    parser.add_argument("--pop-size", type=int, default=10, help="Population size")
    parser.add_argument(
        "--generations", type=int, default=20, help="Number of generations"
    )
    parser.add_argument(
        "--games-per-pair", type=int, default=1, help="Games per AI pair"
    )
    parser.add_argument(
        "--mutation-rate", type=float, default=0.1, help="Mutation rate for weights"
    )
    parser.add_argument(
        "--elite-ratio", type=float, default=0.3, help="Ratio of elite AIs to preserve"
    )
    parser.add_argument("--depth", type=int, default=2, help="Search depth for ChessAI")
    parser.add_argument("--model-out", default=os.path.join("src", "tournament_models"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hidden-archs",
        type=str,
        nargs="*",
        default="128",
        help="Hidden architectures separated by | (e.g., '128,64|64,32|256,128')",
    )
    args = parser.parse_args(argv)

    # Parse architectures
    arch_strings = args.hidden_archs.split("|")
    hidden_layer_options = []
    for arch_str in arch_strings:
        if arch_str.strip() == "":
            arch = []
        else:
            arch = [int(x.strip()) for x in arch_str.split(",")]
        hidden_layer_options.append(arch)

    print("Tournament Training Configuration:")
    print(f"  Population size: {args.pop_size}")
    print(f"  Generations: {args.generations}")
    print(f"  Games per pair: {args.games_per_pair}")
    print(f"  Mutation rate: {args.mutation_rate}")
    print(f"  Elite ratio: {args.elite_ratio}")
    print(f"  Search depth: {args.depth}")
    print(f"  Architectures: {hidden_layer_options}")
    print()

    # Initialize population
    print("Initializing population...")
    population = initialize_population(
        args.pop_size, hidden_layer_options, seed=args.seed
    )

    best_scores_history = []

    # Tournament generations
    for gen in range(args.generations):
        print(f"\nGeneration {gen + 1}/{args.generations}")

        # Evaluate tournament
        print("  Running tournament...")
        scores = evaluate_tournament(population, games_per_pair=args.games_per_pair)

        # Save results
        save_tournament_results(population, scores, args.model_out, gen)

        best_score = max(scores)
        best_scores_history.append(best_score)
        avg_score = np.mean(scores)

        print(f"  Best score: {best_score:.2f}")
        print(f"  Avg score: {avg_score:.2f}")

        # Selection: keep elite, mutate rest
        elite_count = max(1, int(args.pop_size * args.elite_ratio))
        elite_indices = np.argsort(scores)[-elite_count:]

        # Create next generation
        next_population = []

        # Keep elite
        for idx in elite_indices:
            next_population.append(population[idx])

        # Mutate to fill rest
        while len(next_population) < args.pop_size:
            parent_idx = np.random.choice(elite_indices)
            parent_ai = population[parent_idx]

            # Mutate model
            mutated_model = mutate_model(
                parent_ai.model, mutation_rate=args.mutation_rate
            )

            # Create new AI with mutated model
            new_ai = ChessAI(
                sequence=[],
                model=mutated_model,
                depth=args.depth,
                zobrist_seed=args.seed + gen * 1000 + len(next_population),
            )
            next_population.append(new_ai)

        population = next_population

    # Save final best model
    print("\nEvaluating final population...")
    final_scores = evaluate_tournament(population, games_per_pair=args.games_per_pair)
    best_idx = np.argmax(final_scores)
    best_ai = population[best_idx]

    final_best_dir = os.path.join(args.model_out, "final_best_model")
    os.makedirs(final_best_dir, exist_ok=True)
    best_ai.model.save(final_best_dir)
    print(f"Best model saved to {final_best_dir}")

    # Save history
    history = {
        "generations": args.generations,
        "best_scores": best_scores_history,
        "final_best_score": float(final_scores[best_idx]),
        "final_best_architecture": best_ai.model.sizes,
    }

    with open(os.path.join(args.model_out, "tournament_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print("Tournament completed! History saved.")
    print(f"Best architecture found: {best_ai.model.sizes}")
    print(f"Best score: {final_scores[best_idx]:.2f}")


if __name__ == "__main__":
    main()
