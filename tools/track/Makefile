# Path to the original V2X-Sim dataset
original_data_path := /scratch/dm4524/data/V2X-Sim-2
# Path to the dataset for detection
det_data_path := /scratch/dm4524/data/V2X-Sim-det
# [lowerbound / upperbound/ v2v / disco / when2com / when2com_warp / who2com / who2com_warp]
mode := disco
# [with_rsu / no_rsu]
rsu := no_rsu
# Index of current agent
current_agent := 0
# [train / test / val]
split := test
# det path
det_path := /scratch/dm4524/ai4ce/coperception/tools/det
# file containing idxes of scenes to run tracking
scene_idxes_file := /scratch/dm4524/ai4ce/coperception/tools/utils/test_scenes.txt
# Index of the start agent
from_agent := 1
# Index of the end agent + 1
to_agent := 6
# Det logs path (to get the tracking input)
det_logs_path := $(det_path)/logs_compress_0

.PHONY: sort

create_data:
	python create_data_com.py --root $(original_data_path) --data $(det_data_path)/$(split) --split $(split) --from_agent $(from_agent) --to_agent $(to_agent) --scene_idxes_file $(scene_idxes_file)

sort:
	cd sort && python sort.py --mode $(mode)/$(rsu) --split $(split) --from_agent $(from_agent) --to_agent $(to_agent) --scene_idxes_file $(scene_idxes_file) --det_logs_path $(det_logs_path)

eval:
	python run_multi_agent_mot_challenge.py --mode $(mode) --rsu $(rsu) --from_agent $(from_agent) --to_agent $(to_agent) --scene_idxes_file $(scene_idxes_file) --split $(split)