import os

import matplotlib
import numpy as np
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
from simple_dataset import SimpleDataset
from tqdm import tqdm

COLOR = ['red','springgreen','dodgerblue', 'darkviolet', 'orange']
COLOR_RGB = [ tuple([int(cc * 255) for cc in matplotlib.colors.to_rgb(c)]) for c in COLOR]
COLOR_PC = [tuple([int(cc*0.2 + 255*0.8) for cc in c]) for c in COLOR_RGB]
CLASSES = ['agent1', 'agent2', 'agent3', 'agent4', 'agent5']

def main():
    from_scene = 31
    to_scene = 33
    output_dir = './result_v2x'

    ## basic setting
    dataset = SimpleDataset(root_dir=f'./v2xsim2_info', from_scene=from_scene, to_scene=to_scene)
    
    ## matplotlib setting
    plt.figure()

    ## draw
    current_scene = from_scene - 1
    current_frame_idx = 0
    print(f"Start visualization. Results save to {output_dir}")
    dataset_len = len(dataset)
    for idx in tqdm(range(dataset_len)):
        if idx % 100 == 0:
            current_scene += 1
            current_frame_idx = 0

        base_data_dict = dataset[idx]
        cav_ids = list(base_data_dict.keys())
        color_map = dict()
        posx = dict()
        posy = dict()
        for (idx, cav_id) in enumerate(cav_ids):
            color_map[cav_id] = COLOR[idx]
            posx[cav_id] = []
            posy[cav_id] = []
        recs = []
        classes = []
        for i in range(0,len(cav_ids)):
            recs.append(mpatches.Rectangle((0,0),1,1,fc=COLOR[i]))
            classes.append(CLASSES[i])


        plt.style.use('dark_background')
        plt.xlim((-100,120))
        plt.ylim((-70,250))
        plt.xticks([])
        plt.yticks([])

        plt.legend(recs,classes,loc='lower left')

        for cav_id, cav_content in base_data_dict.items():
            pos = cav_content['params']['lidar_pose'][:2,3] 
            posx[cav_id].append(pos[0])
            posy[cav_id].append(pos[1])

            start_idx = max(0, idx-10)
            end_idx = idx
            for inner_idx in range(start_idx, end_idx + 1):
                plt.scatter(np.array(posx[cav_id][inner_idx:inner_idx+1]), 
                            np.array(posy[cav_id][inner_idx:inner_idx+1]),
                            s=(inner_idx-start_idx + 1)*4, 
                            alpha=(1 - (end_idx - inner_idx) * 0.09),
                            c=color_map[cav_id])

        
        save_path = os.path.join(output_dir, f'location_in_bev/{current_scene}')
        os.makedirs(save_path, exist_ok=True)

        plt.savefig(f"{save_path}/trajectory_{current_frame_idx:02d}.png", dpi=300)
        plt.clf()
        current_frame_idx += 1
    

if __name__ == "__main__":
    main()
