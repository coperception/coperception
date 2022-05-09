import cv2
import glob
import os
from tqdm import tqdm

mode = 'scene_overview_Mixed'
from_scene = 31
to_scene = 33

for idx in range(from_scene, to_scene + 1):
    img_dir = 'result_v2x'
    output_path = './result_v2x_video'
    os.makedirs(output_path, exist_ok=True)

    input_path = os.path.join(img_dir, mode, str(idx))

    print(f'Reading input from: {input_path}')

    img_array = []
    for filename in tqdm(sorted(glob.glob(f'{input_path}/*.png'))):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    output_name = os.path.join(mode, str(idx))
    output_path = os.path.join(output_path, f'{mode}_{idx}.mp4')
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 9, size)

    print(f'Write .mp4 video to {output_path}')

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()