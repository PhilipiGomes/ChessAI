from chessAI import ChessAI
import chess, os, random, argparse

# --- Exemplo rápido ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--depth-ai-1", type=int, default=2, help="depth of the Engine 1"
    )
    parser.add_argument(
        "--depth-ai-2", type=int, default=2, help="depth of the Engine 2"
    )
    args = parser.parse_args(None)
    moves = []
    board = chess.Board()
    if len(os.listdir("model")) == 0 and len(os.listdir("prev_model")) == 0:
        raise ValueError("No models found. Please train a model first.")
    ai1 = ChessAI(depth=args.depth_ai_1, sequence=moves)
    ai1.load_model("model") if len(os.listdir("model")) > 0 else ai1.load_model("prev_model")
    ai2 = ChessAI(depth=args.depth_ai_2, sequence=moves)
    ai2.load_model("prev_model") if len(os.listdir("prev_model")) > 0 else ai2.load_model("model")
    # exemplo: avaliar posição inicial; como é branco a mover, último elemento será +1.0
    print(board)
    ai_white = random.choice([ai1, ai2])
    ai_black = ai2 if ai_white == ai1 else ai1
    ai_white_str = "AI 1" if ai_white == ai1 else "AI 2"
    ai_black_str = "AI 2" if ai_black == ai2 else "AI 1"
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move = ai_white.choose_move(board)
        else:
            move = ai_black.choose_move(board)
        move_san = board.san(move)
        os.system("cls")
        print(f"Brancas: {ai_white_str}, depth: {ai_white.depth} | Pretas: {ai_black_str}, depth: {ai_black.depth}")
        print("Escolhido:", move_san)
        moves.append(move_san)
        board.push(move)
        print(board)
    print("Jogo terminado:", board.result())
    with open("jogo.pgn", "w") as f:
        f.write(f'[Event "AI vs AI"]\n')
        f.write(f'[White "{ai_white_str}, depth: {ai_white.depth}"]\n')
        f.write(f'[Black "{ai_black_str}, depth: {ai_black.depth}"]\n')
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
