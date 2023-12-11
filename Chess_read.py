import chess.pgn
import os
import json

dirpath = "C:\\Users\\ROHIT FRANCIS\\Downloads\\ccrl-pgn\\cclr\\train"
file_list = os.listdir(dirpath)

print(len(file_list))
# exit(0)
def read_pgn_file(file_path):
    with open(file_path, 'r') as f:
        game = chess.pgn.read_game(f)
        return game

# def print_moves(game):
#     board = game.board()
#     for move in game.mainline_moves():
#         board.push(move)
#         print(board)

def Make_data(game):
    data = []
    board = game.board()
    for move in game.mainline_moves():
        board.push(move)
        # print(board)
        data.append(str(board))

    return data
# Replace 'your_file.pgn' with the actual file path of your PGN file
file_path = "C:\\Users\\ROHIT FRANCIS\\Downloads\\ccrl-pgn\\cclr\\train\\96.pgn"
game = read_pgn_file(file_path)
board = str(game.board())
print(len())
exit(0)
dataset = []
for filename in file_list:
    game = read_pgn_file(dirpath+f"\\{filename}")
    data = Make_data(game)
    dataset.append(data)

print(len(dataset))
exit(0)
dataset = {"title":"Chess Data", "data":dataset}
with open("Chess_data.json","w") as f:
    s = json.dumps(dataset)
    f.write(s)
print("Data saved")