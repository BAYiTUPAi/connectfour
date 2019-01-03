from q_learn import train_model
from game_ai import play_game_ai
from game import play_game

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    modes = ["train", "play", "play_ai"]
    print(f"Set mode with --mode command line flag (choices {modes})")
    parser.add_argument("--mode", choices=modes)
    args = parser.parse_args()
    mode = args.mode

    if mode is None:
        raise Exception("Need to supply --mode=X")

    if mode == "train":
        train_model()

    if mode == "play":
        play_game()

    if mode == "play_ai":
        play_game_ai()