import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from game_ai import apply_action, get_actions, detect_winner, initialize_board, play_game_ai


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
        in_dim = 2*6*7+7
        self.l1 = nn.Linear(in_dim, 80)
        self.l2 = nn.Linear(80, 10)
        self.l3 = nn.Linear(10, 10)
        self.l4 = nn.Linear(10, 1)

    def forward(self, x, a):
        # Reshape input to be long and flat
        x = x.reshape((1, -1))
        a_vec = torch.zeros(1, 7)
        a_vec[0, a] = 1.0
        feat = torch.cat((x, a_vec), dim=1)

        l1 = F.relu(self.l1(feat))
        l2 = F.relu(self.l2(l1))
        l3 = F.relu(self.l3(l2))
        out = F.relu(self.l4(l3))

        return out.reshape(-1)

    def copy_weights_to(self, other):
        weights = self.state_dict()
        other.load_state_dict(weights)

model_path_A = "qagentB.pt"
model_path_B = "qagentA.pt"

def train_model():
    # Initialize a convolutional neural network to represent the Q estimate
    netA = DeepQ()
    netB = DeepQ()
    netA2 = DeepQ()
    netB2 = DeepQ()
    for param in netA.parameters():
        param.requires_grad = False
    for param in netB.parameters():
        param.requires_grad = False

    # Train the network
    try:
        gamma = 0.95
        learning_rate = 1e-4
        loss_fn = torch.nn.MSELoss(reduction='sum')
        optimizerA = torch.optim.Adam(netA2.parameters(), lr=learning_rate)
        optimizerB = torch.optim.Adam(netB2.parameters(), lr=learning_rate)
        game_n = 1
        while True:
            print(f"Beginning game {game_n}")
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
                apply_action((1 if is_1s_turn else 2), board, chosen_action)

                # Calculate a better estimate of the move
                reward = 0.0
                game_result = detect_winner(board, (1 if is_1s_turn else 2))
                if game_result is not None:
                    if game_result == 1 and is_1s_turn or game_result == 2 and not is_1s_turn:
                        reward = 1.0
                    else:
                        reward = -1.0
                    done = True

                newboard_repr = serialize_board(board)
                if not is_1s_turn:
                    perm = torch.LongTensor([1, 0])
                    newboard_repr = newboard_repr[perm]

                future_reward = gamma * np.max([Q_estimator(newboard_repr, action) for action in get_actions(board)])
                better_prediction_Q = reward + future_reward

                # Backpropagate the error to the 2 models
                Q_estimator_2 = netA2 if is_1s_turn else netB2
                optimizer = optimizerA if is_1s_turn else optimizerB
                q_pred = Q_estimator_2(board_repr, chosen_action)
                q_pred = q_pred.reshape(1, -1)
                better_prediction_Q = torch.tensor(better_prediction_Q).reshape(1, -1)
                loss = loss_fn(q_pred, better_prediction_Q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Set up for next turn
                is_1s_turn = not is_1s_turn

            # Move the updates we made on the 2nd models into the first
            netA2.copy_weights_to(netA)
            netB2.copy_weights_to(netB)

            game_n += 1

    except KeyboardInterrupt:
        # Save the models
        torch.save(netA.state_dict(), model_path_A)
        torch.save(netB.state_dict(), model_path_B)

def load_model(is_A=True):
    model = DeepQ()
    path = model_path_A if is_A else model_path_B
    model.load_state_dict(torch.load(path))
    for param in netA.parameters():
        param.requires_grad = False
    return model

def get_model_action(turn, board, model) -> int:
    actions_avail = get_actions(board)
    if len(actions) == 0:
        raise Exception("No possible actions were provided to the AI.")
    board_repr = serialize_board(board)
    if turn == 2:
        perm = torch.LongTensor([1, 0])
        board_repr = board_repr[perm]
    action_values = [model(board_repr, action) for action in actions_avail]
    chosen_i = np.argmax(action_values)
    chosen_action = actions_avail[chosen_i]
    predicted_Q = action_values[chosen_i]
    return chosen_action
