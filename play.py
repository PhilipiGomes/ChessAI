from chessAI import ChessAI
import chess, os, random, argparse

# --- Exemplo rápido ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth-ai-1", default=2, required=True, help="depth of the Engine 1")
    parser.add_argument("--depth-ai-2", default=2, required=True, help="depth of the Engine 2")
    args = parser.parse_args(None)
    moves = []
    board = chess.Board()
    ai1 = ChessAI(depth=args.depth_ai_1, sequence=moves)
    ai1.load_model('model')
    ai2 = ChessAI(depth=args.depth_ai_2, sequence=moves)
    ai2.load_model('prev_model')
    # exemplo: avaliar posição inicial; como é branco a mover, último elemento será +1.0
    print(board)
    ai_white = random.choice([ai1, ai2])
    ai_black = ai2 if ai_white == ai1 else ai1
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move = ai_white.get_move(board)
        else:
            move = ai_black.get_move(board)
        move_san = board.san(move)
        os.system('cls')
        print("Profundidade das IAs:", f'{ai1.depth} x {ai2.depth}')
        print("Escolhido:", move_san)
        moves.append(move_san)
        board.push(move)
        print(board)
    print("Jogo terminado:", board.result())
    with open('jogo.txt', 'w') as f:
        f.write(' '.join(moves))
        f.write('\n')
        f.write(board.result())
    print("Movimentos salvos em jogo.txt")