import os
import shutil
import argparse


def main(args):
    if args.split == "val":
        seqs = range(80, 90)
    elif args.split == "test":
        seqs = range(90, 100)

    save_dir = f"./TrackEval/data/trackers/mot_challenge/V2X-{args.split}/sort-{args.mode}/data"
    os.makedirs(save_dir, exist_ok=True)
    for seq in seqs:
        shutil.copy(
            os.path.join(args.root, f"{seq}.txt"), os.path.join(save_dir, f"{seq}.txt")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--split", type=str)
    parser.add_argument("--root", type=str)
    args = parser.parse_args()
    main(args)
