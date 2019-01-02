
# GLOBAL VARS
board_layout = [["_" for x in range(7)] for x in range(6)]
column_str = "1  2  3  4  5  6  7"
turn = "Player 2"
game_won = False
###

def print_board():
    # prints the current board
    print(column_str)
    for row in board_layout:
        print('  '.join(row))
    print("\n")


def verify_drop_input():
    # verify a legitimate player input for the piece
    pass # CREATE LATER

def drop_piece():
    # gets player input and drops piece. then switches turn.
    print(turn + "'s Turn")
    player_column_input = input("Choose a column: ")
    player_column_integer_input = int(player_column_input) - 1
    print("You chose: " + player_column_input + "\n\n")
    for i in range(5, -1, -1):
        if board_layout[i][player_column_integer_input] == "_":
            if turn == "Player 1":
                board_layout[i][player_column_integer_input] = "X"
                break
            elif turn == "Player 2":
                board_layout[i][player_column_integer_input] = "0"
                break

def change_turn():
    global turn
    if turn == "Player 1":
        turn = "Player 2"
    elif turn == "Player 2":
        turn = "Player 1"


def test_horiz(test_val):
    # _
    # return 1 if won
    # return 0 if not-won
    for row in board_layout:
        in_a_row_count = 0
        for col in row:
            if col == test_val:
                in_a_row_count += 1
                if in_a_row_count == 4:
                    return 1
            else:
                in_a_row_count = 0


def test_vert(test_val):
    # |
    # return 1 if won
    # return 0 if not-won
    for col in range(7):
        in_a_row_count = 0
        for row in range(6):
            if board_layout[row][col] == test_val:
                in_a_row_count += 1
                if in_a_row_count == 4:
                    return 1
            else:
                in_a_row_count = 0

def test_forward_diag(test_val):
    # /
    # return 1 if won
    # return 0 if not-won
    for i in range(7):
        in_a_row_count = 0
        for j in range(6):
            try:
                if board_layout[5 - j][i + j] == test_val:
                    in_a_row_count += 1
                    if in_a_row_count == 4:
                        return 1
                else:
                    in_a_row_count = 0
            except IndexError:
                break
    for i in range(6):
        in_a_row_count = 0
        for j in range(6):
            try:
                if board_layout[i - j][j] == test_val:
                    in_a_row_count += 1
                    if in_a_row_count == 4:
                        return 1
                else:
                    in_a_row_count = 0
            except IndexError:
                break


def test_back_diag(test_val):
    # \
    # return 1 if won
    # return 0 if not-won
    for i in range(7):
        in_a_row_count = 0
        for j in range(6):
            try:
                if board_layout[j][i + j] == test_val:
                    in_a_row_count += 1
                    if in_a_row_count == 4:
                        return 1
                else:
                    in_a_row_count = 0
            except IndexError:
                break
    for i in range(6):
        in_a_row_count = 0
        for j in range(6):
            try:
                if board_layout[i + j][j] == test_val:
                    in_a_row_count += 1
                    if in_a_row_count == 4:
                        return 1
                else:
                    in_a_row_count = 0
            except IndexError:
                break

def game_won_test():
    # test if four in a row
    # do this by testing _ | / \
    global game_won
    test_val = ""
    if turn == "Player 1":
        test_val = "X"
    elif turn == "Player 2":
        test_val = "0"
    # tests return 1 if game won, or 0 if not-won
    # test _
    if test_horiz(test_val):
        game_won = True
    # test |
    if test_vert(test_val):
        game_won = True
    # test /
    if test_forward_diag(test_val):
        game_won = True
    # test \
    if test_back_diag(test_val):
        game_won = True

def play_game():
    # iterates through ... until game won
    # 1. print the board
    # 2. player input and drop piece
    # 3. test if game won
    # 4. switch turn
    while not game_won:
        change_turn()
        print_board()
        drop_piece()
        game_won_test()
    print_board()
    if "_" not in board_layout:
        print("TIE GAME.")
        return 0
    print(turn.upper() + " WINS THE GAME!!!")





# run functions / play the game
# play_game()






