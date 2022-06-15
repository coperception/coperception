# !!! WARNING !!!
# Some of the "0.npy" files inside the folder of each scene might be moved out of the folder, and the folder will be disappeared.
# E.g. We encountered this on scene 45_30
# Please check manually for whether some scenes have this problem.
import os
import shutil

scene_file = 'test_scenes.txt'
train_scene_file = open(scene_file, 'r')

train_idxs = set()
for line in train_scene_file:
    line = line.strip()
    train_idxs.add(int(line))

from_loc = '/scratch/dm4524/data/V2X-Sim-det/all'
to_loc = '/scratch/dm4524/data/V2X-Sim-det/test'

for agent_dir in os.listdir(from_loc):
    to_dir = os.path.join(to_loc, agent_dir)
    agent_dir = os.path.join(from_loc, agent_dir)
    for f in os.listdir(agent_dir):
        scene_file_path = os.path.join(agent_dir, f)
        scene_idx = int(f.split('_')[0])
        if scene_idx in train_idxs:
            shutil.move(scene_file_path, to_dir)