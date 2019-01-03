def test_horiz(board_layout, test_val):
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


def test_vert(board_layout, test_val):
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

def test_forward_diag(board_layout, test_val):
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


def test_back_diag(board_layout, test_val):
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



#########################################################
# API for machine learning

def apply_action(turn, board, action):
    # update the board after applying the action
    # turn is int representing player 1 or 2
    # board is list
    # action is int representing column (0 - 5)
    for i in range(5, -1, -1):
        if board[i][action] == 0:
            if turn == 1:
                board[i][action] = 1
                break
            elif turn == 2:
                board[i][action] = 2
                break


def get_actions(board):
    # return a list of possible actions
    avail_actions = []
    for i in range(7):
        if board[0][i] == 0:
            avail_actions.append(i)
    return avail_actions


def detect_winner(board, turn):
    # return 1 or 2 if either of these players won, or return None
    # turn == player turn, but also the value for that player on the board
    # test _
    if test_horiz(board, turn):
        return turn
    # test |
    if test_vert(board, turn):
        return turn
    # test /
    if test_forward_diag(board, turn):
        return turn
    # test \
    if test_back_diag(board, turn):
        return turn
    return None

def initialize_board():
    # create a new board and return it
    return [[0 for x in range(7)] for x in range(6)]

def play_game_ai():
    # functions available to you: load_model() and get_model_action(model, board)
    # initialize everything
    board = initialize_board()
    turn = 1
    model = load_model()
    game_won = False
    # play the game!
    while not game_won:
        avail_actions = get_actions(board)
        action = get_model_action(turn, board, model)
        apply_action(turn, board, action)
        if detect_winner(board, turn):
            return turn
        else:
            if turn == 1:
                turn = 2
            else:
                turn = 1
