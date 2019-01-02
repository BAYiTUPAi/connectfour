from torch import nn
import torch.functional as F
import numpy as np

# from


###
### Q-Learner
###
def serialize_board(board):
    # This function couples the board data structure to Pytorch tensors
    # Convert the board to a Pytorch Tensor
    board_T = torch.tensor(board)
    # Make a new tensor the same dimensions as the board, but with two channels
    multichanneled = torch.zeros((2, *board_T.shape))
    multichanneled[0, board_T==1] = 1.0
    multichanneled[1, board_T==2] = 1.0
    return multichanneled

class DeepQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 6, 6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x, a):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def copy_weights_to(self, other):
        weights = self.state_dict()
        other.load_state_dict(weights)


def train_model():
    # Initialize a convolutional neural network to represent the Q estimate
    netA = deepQ()
    netB = deepQ()
    netA2 = deepQ()
    netB2 = deepQ()
    for param in netA.parameters():
        param.requires_grad = False
    for param in netB.parameters():
        param.requires_grad = False

    # Train the network
    n_trials = 20
    gamma = 0.95
    available_actions = get_actions()
    for trial in range(n_trials):
        print(f"Beginning game {trial}/{n_trials}")
        # Play the game using the Q-strategy
        board = initialize_board()

        # Until the game is not done:
        done = False
        is_1s_turn = True
        while not done:
            # Calculate the estimate of the best move
            actions = get_actions(board)
            if len(actions) == 0:
                break
            board_repr = serialize_board(board)
            if not is_1s_turn:
                perm = torch.LongTensor([1, 0])
                board_repr = board_repr[perm]
            Q_estimator = netA if is_1s_turn else netB
            action_values = [Q_estimator(board_repr, action) for action in actions]

            # Make the move, and get a reward & next state
            chosen_i = np.argmax(action_values)
            chosen_action = actions[chosen_i]
            predicted_Q = action_values[chosen_i]
            apply_action(board, chosen_action)

            # Calculate a better estimate of the move
            reward = 0.0
            game_result = detect_winner(board)
            if game_result is not None:
                if game_result == 1 and is_1s_turn or game_result == 2 and not is_1s_turn:
                    reward = 1.0
                else:
                    reward = -1.0

            newboard_repr = serialize_board(board)
            if not is_1s_turn:
                perm = torch.LongTensor([1, 0])
                newboard_repr = newboard_repr[perm]

            future_reward = gamma * np.max([Q_estimator(newboard_repr, action) for action in get_actions(board)])
            better_prediction_Q = reward + future_reward

            # Backpropagate the error

            # Set up for next turn
            is_1s_turn = not is_1s_turn

def load_model():
    pass

def get_model_action(model, state) -> int:
    pass

