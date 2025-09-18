from chessAI import ChessAI, board_to_feature_vector, SimpleMLP
import chess, os, random
import argparse

# --- Exemplo rápido ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", default=2, required=True, help="depth of the Engine")
    args = parser.parse_args(None)
    moves = []
    board = chess.Board()
    ai = ChessAI(depth=args.depth, sequence=moves)
    ai.load_model('models')
    players = ['AI', 'Human']
    white = random.choice(players)
    black = 'AI' if white == 'Human' else 'Human'
    
    
    print(board)
    while not board.is_game_over():
        if (white == 'AI' and board.turn == chess.WHITE) or (black == 'AI' and board.turn == chess.BLACK):
            move = ai.choose_move(board)
            move_san = board.san(move)
            os.system('cls')
            print("Escolhido:", move_san)
            moves.append(move_san)
            board.push(move)
            print(board)
        else:
            while True:
                move = input('Your Move (SAN notation): ')
                try:
                    move_converted = board.parse_san(move)
                except:
                    print('Invalid move! Try again.')
                break
            os.system('cls')
            print("Escolhido:", move)
            try:
                board.push(move_converted)
            except Exception as e:
                print(f'error: {e}')
            moves.append(move)
            print(board)
    print("Jogo terminado:", board.result())
    with open('jogo.txt', 'w') as f:
        f.write(' '.join(moves))
        f.write('\n')
        f.write(board.result())
    print("Movimentos salvos em jogo.txt")