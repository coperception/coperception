import os
import argparse


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="SORT demo")
    parser.add_argument(
        "--scene_idxes_file",
        type=str,
        help="File containing idxes of scenes to run tracking",
    )
    parser.add_argument(
        "--from_agent", default=0, type=int, help="start from which agent"
    )
    parser.add_argument(
        "--to_agent", default=6, type=int, help="until which agent (index + 1)"
    )
    parser.add_argument("--split", type=str, help="[test/val]")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    scene_idxes_file = args.scene_idxes_file
    from_agent = args.from_agent
    to_agent = args.to_agent
    split = args.split
    
    scene_idxes_file = open(scene_idxes_file, "r")
    scene_idxes = [int(line.strip()) for line in scene_idxes_file]

    seqmaps_dir = "TrackEval/data/gt/mot_challenge/seqmaps"
    os.makedirs(seqmaps_dir, exist_ok=True)
    for ii in range(from_agent, to_agent):
        seqmap_file = os.path.join(seqmaps_dir, f"V2X-{split}{ii}.txt")
        seqmap_file = open(seqmap_file, "w")
        seqmap_file.write("name\n")
        for scene_idx in scene_idxes:
            seqmap_file.write(f"{scene_idx}\n")
