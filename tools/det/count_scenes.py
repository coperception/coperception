# from nuscenes.nuscenes import NuScenes

# nusc = NuScenes(version='v1.0-mini', dataroot='/scratch/dm4524/data/V2X-Sim-2', verbose=True)

# def get_scene_idxes(scene_file_path):
#     scene_file = open(scene_file_path, 'r')
#     idxs = []
#     for line in scene_file:
#         line = line.strip()
#         idxs.append(int(line))
    
#     return idxs

# train_idxes = get_scene_idxes('train_scenes.txt')
# test_idxes = get_scene_idxes('test_scenes.txt')
# val_idxes = get_scene_idxes('val_scenes.txt')

# def get_scene_count(idxes):
#     count = 0
#     for ii in idxes:
#         count += nusc.scene[ii]['nbr_samples']
#     return count

# print('Train:', get_scene_count(train_idxes))
# print('Test:', get_scene_count(test_idxes))
# print('Val:', get_scene_count(val_idxes))

from coperception.datasets import V2XSimDet
from torch.utils.data import DataLoader
import torch
from coperception.configs import Config, ConfigGlobal
from tqdm import tqdm


split='test'
config = Config(split, binary=True, only_det=True)
config_global = ConfigGlobal(split, binary=True, only_det=True)

training_dataset = V2XSimDet(
    dataset_roots=[f"/scratch/dm4524/data/V2X-Sim-det/{split}/agent{i}" for i in range(6)],
    config=config,
    config_global=config_global,
    split=split,
    bound='lowerbound',
    kd_flag=False,
    no_cross_road=False,
)
training_data_loader = DataLoader(
    training_dataset, shuffle=True, batch_size=1, num_workers=2
)

total_num = 0
for sample in tqdm(training_data_loader):
    (
        padded_voxel_point_list,
        padded_voxel_points_teacher_list,
        label_one_hot_list,
        reg_target_list,
        reg_loss_mask_list,
        anchors_map_list,
        vis_maps_list,
        target_agent_id_list,
        num_agent_list,
        trans_matrices_list,
    ) = zip(*sample)

    total_num += num_agent_list[0][0].item()

print(total_num)