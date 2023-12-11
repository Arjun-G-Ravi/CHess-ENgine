import chess.pgn

board = chess.Board()
print(board)
print(len(str(board)))
print(board.legal_moves)
board.push_san("e4")
board.push_san("e5")
board.push_san("Qh5")
print(board.legal_moves)
# board.push_uci("d4")
print(board)