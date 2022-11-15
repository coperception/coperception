# Visualization tools for Collaborative Perception dataset

Code in this directory are from [Yifan Lu](https://github.com/yifanlu0227/v2xsim_vistool)

## Usage

I first reconstruct the nuscenes-format data, since it's very slow to read?
1.  `process_v2xsim_v2.py` will generate pkl file containing some meta-information of v2x-sim dataset, including lidar pose, gt boxes, and path to actually lidar file.
```
python dataset_process/process_v2xsim_v2.py
```


2. folder `simple_plot3d` is highly from [this repo](https://github.com/Divadi/simple_plot3d), `simple_plot3d/simple_vis.py` provide a warpper function for tensor format data. It can draw both BEV image and 3D view image. You can try to integrate the `visualize` function into your deep learning framework.


3. `simple_dataset.py` provide a simple dataset design for v2x-sim 2.0, it will retrieve meta information from pkl file and lidar numpy array using lidar file path.

4. Other files like `collaboration_view.py`, `single_view.py`, `scene_overview.py`, `location_in_bev.py` can draw seqences of pictures that provide a comprehensive overview of the dataset(given a scene). 
```
python single_view.py
python collaboration_view.py
python location_in_bev.py
python scene_overview.py
```
Then you can use `img2video.py` to make image sequence to video.

5. You can run `visualize_data_seq.py` to see how to use the warp function `visualize` mentioned in 2., just 
```
python visualize_data_seq.py
```
Result:
![3d_00000](https://user-images.githubusercontent.com/53892579/190858420-d0e90a45-139c-4bd2-bfc1-9d55e0498230.png)
![bev_00000](https://user-images.githubusercontent.com/53892579/190858428-f5afe1e2-0446-44ac-a022-c5b4f898763b.png)


GIFs are shown as below:
![single_view_agent1](https://user-images.githubusercontent.com/53892579/190858435-4bdc55ae-2144-4eda-a3ef-87beae2e5d0d.gif)
![collaboration_view_agent1](https://user-images.githubusercontent.com/53892579/190858456-3ed721f1-4ed7-4b75-a3de-9541c2925561.gif)
![scene_overview_Mixed](https://user-images.githubusercontent.com/53892579/190858478-ee9bfe45-3378-4340-bc13-52b20541a1b7.gif)
![location_in_bev](https://user-images.githubusercontent.com/53892579/190858483-677e6036-5c43-4adc-aef2-ceaebb66e2f3.gif)
