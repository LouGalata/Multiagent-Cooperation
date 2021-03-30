import pickle
import os


def main():
    with open(loadpath, 'rb') as f:
        data = pickle.load(f)
        # load domain parameters
        params = vars(data)
        print(params)


if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    rel_path = "c2a6d/args.pkl"
    loadpath = os.path.join(script_dir, rel_path)
    main()