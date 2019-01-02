from q_learn import train_model
from game_ai import play_game_ai
from game import play_game

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    modes = ["train", "play", "play_ai"]
    mode_default = "test"
    print(f"Set mode with --mode command line flag (default {mode_default}, choices {modes})")
    parser.add_argument("--mode", default=mode_default, choices=modes)
    args = parser.parse_args()
    mode = args.mode


    if mode == "train":
        train_model()

    if mode == "play":
        play_game()

    if mode == "play_ai":
        play_game_ai()