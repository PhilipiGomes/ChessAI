# trunk-ignore-all(git-diff-check/error)
import argparse
import os
import random

import chess

from chessAI import ChessAI

# --- Exemplo rápido ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=2, help="depth of the Engine")
    parser.add_argument(
        "--dir-model", default=os.path.join('src','model'), help="Directory of the model of the AI"
    )
    args = parser.parse_args(None)
    moves = []
    board = chess.Board()
    ai = ChessAI(depth=args.depth, sequence=moves)
    ai.load_model(os.path.join('src','prev_model'))
    players = ["AI", "Human"]
    # trunk-ignore(bandit/B311)
    white = random.choice(players)
    black = "AI" if white == "Human" else "Human"

    print(board)
    while not board.is_game_over():
        if (white == "AI" and board.turn == chess.WHITE) or (
            black == "AI" and board.turn == chess.BLACK
        ):
            move = ai.choose_move(board)
            if move:
                move_san = board.san(move)
                # trunk-ignore(bandit/B605)
                # trunk-ignore(bandit/B607)
                os.system("cls")
                print(f"Brancas: {white} | Pretas: {black}")
                if white == "AI":
                    print(f"AI (Brancas) depth: {ai.depth}")
                if black == "AI":
                    print(f"AI (Pretas) depth: {ai.depth}")
                print("Escolhido:", move_san)
                moves.append(move_san)
                board.push(move)
                print(board)
            else:
                print("AI não encontrou um movimento válido.")
                break
        else:
            while True:
                move = input("Seu lance (Notação SAN ex: e4, Nf3, Qxd2): ")
                try:
                    move_converted = board.parse_san(move)
                    break
                except Exception:
                    print("Movimento inválido, tente de novo.")
            # trunk-ignore(bandit/B605)
            # trunk-ignore(bandit/B607)
            os.system("cls")
            print("Escolhido:", move)
            try:
                board.push(move_converted)
            except Exception as e:
                print(f"error: {e}")
            moves.append(move)
            print(board)
    print("Jogo terminado:", board.result())
    # Determine player names and AI depths for PGN
    if white == "AI":
        ai_white_str = "AI"
        ai_white_depth = ai.depth
    else:
        ai_white_str = "Human"
        ai_white_depth = "-"
    if black == "AI":
        ai_black_str = "AI"
        ai_black_depth = ai.depth
    else:
        ai_black_str = "Human"
        ai_black_depth = "-"

    with open("jogo.pgn", "w") as f:
        f.write('[Event "AI vs Human"]\n')
        f.write(f'[White "{ai_white_str}, depth: {ai_white_depth}"]\n')
        f.write(f'[Black "{ai_black_str}, depth: {ai_black_depth}"]\n')
        f.write(f'[Result "{board.result()}"]\n')
        f.write("\n")

        # Escreve os lances no formato PGN (com contagem de jogadas)
        for i in range(0, len(moves), 2):
            move_number = i // 2 + 1
            if i + 1 < len(moves):
                f.write(f"{move_number}. {moves[i]} {moves[i+1]} ")
            else:
                f.write(f"{move_number}. {moves[i]} ")

        f.write(board.result())
        f.write("\n")
    print("Movimentos salvos em jogo.txt")
