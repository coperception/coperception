from distutils.sysconfig import customize_compiler
import os

def parse_scene_files(scene_files_loc):
    scenes = {}

    for split in ['train', 'val', 'test']:
        with open(os.path.join(scene_files_loc, f'{split}_scenes.txt'), mode='r') as f:
            for line in f:
                current_set = scenes.get(split, set())
                current_set.add(int(line))
                scenes[split] = current_set
    
    return scenes