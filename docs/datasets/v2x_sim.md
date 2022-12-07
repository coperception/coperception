# V2X-Sim

## Introduction
**V2X-Sim** is a Comprehensive Synthetic Multi-agent Perception Dataset.  
You can find more information [on its website](https://ai4ce.github.io/V2X-Sim/index.html).  
Download links of V2X-Sim dataset can be found [here](https://ai4ce.github.io/V2X-Sim/download.html).  

## File Structure
**V2X-Sim** Follows the same file structure as the [Nuscenes dataset](https://www.nuscenes.org/).
```
V2X-Sim
├── maps # images for the map of one of the towns
├── sweeps # sensor data
|   ├── LIDAR_TOP_id_0 # top lidar data for the top camera, agent 0 (RSU)
|   ├── LIDAR_TOP_id_1 # top lidar data for the top camera, agent 1
|   ├── LIDAR_TOP_id_2 # top lidar data for the top camera, agent 2
|   ...
├── v2.0-mini # metadata
|   ├── scene.json # metadata for all the scenes
|   ├── sample.json # metadata for each sample, organized like linked-list
|   ├── sample_annotation.json # sample annotation metadata for each scene
|   ...
```  

## Data Preparation
V2X-Sim dataset needs to be preprocessed before running our example tasks.  
Currently, 3 tasks are supported:   

- `det`: detection 
- `seg`: semantic segmentation
- `track`: tracking


Data creation steps:  

1. `cd` into either of the directories of these tasks in `tools/`
2. Open `Makefile` in your text editor
3. Change the required arguments of task `create_data` based on your needs
4. Run `make create_data`