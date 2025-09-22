from chessAI import ChessAI, board_to_feature_vector, SimpleMLP
import chess, os, random
import argparse

# --- Exemplo rápido ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=2, help="depth of the Engine")
    parser.add_argument(
        "--dir-model", default="model", help="Directory of the model of the AI"
    )
    args = parser.parse_args(None)
    moves = []
    board = chess.Board()
    ai = ChessAI(depth=args.depth, sequence=moves)
    ai.load_model("prev_model")
    players = ["AI", "Human"]
    white = random.choice(players)
    black = "AI" if white == "Human" else "Human"

    print(board)
    while not board.is_game_over():
        if (white == "AI" and board.turn == chess.WHITE) or (
            black == "AI" and board.turn == chess.BLACK
        ):
            move = ai.choose_move(board)
            move_san = board.san(move)
            os.system("cls")
            print("Escolhido:", move_san)
            moves.append(move_san)
            board.push(move)
            print(board)
        else:
            while True:
                move = input("Seu lance (Notação SAN ex: e4, Nf3, Qxd2): ")
                try:
                    move_converted = board.parse_san(move)
                except:
                    print("Movimento inválido, tente de novo.")
                break
            os.system("cls")
            print("Escolhido:", move)
            try:
                board.push(move_converted)
            except Exception as e:
                print(f"error: {e}")
            moves.append(move)
            print(board)
    print("Jogo terminado:", board.result())
    with open("jogo.txt", "w") as f:
        f.write(" ".join(moves))
        f.write("\n")
        f.write(board.result())
    print("Movimentos salvos em jogo.txt")
