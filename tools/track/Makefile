# Path to the original V2X-Sim dataset
original_data_path := /scratch/dm4524/data/V2X-Sim-2
# Path to the dataset for detection
det_data_path := /scratch/dm4524/data/V2X-Sim-det/
# [lowerbound / upperbound/ v2v / disco / when2com / when2com_warp / who2com / who2com_warp]
mode := disco
# [with_cross / no_cross]  with_cross ==> with RSU; no_cross ==> no RSU
cross := no_cross
# Index of current agent
current_agent := 0
# [train / test / val]
split := test
# Index of begining scene
scene_begin := 90
# Index of ending scene
scene_end := 100

.PHONY: sort

create_data:
	python create_data_com.py --root $(original_data_path) --data $(det_data_path)/$(split) --split $(split) -b $(scene_begin) -e $(scene_end) --current_agent $(current_agent)

sort:
	cd sort && python sort.py --mode $(mode)/$(cross) --current_agent $(current_agent) --split $(split) -b $(scene_begin) -e $(scene_end)

# Evaludate MOTA, MOTP
eval1:
	cd TrackEval && python ./scripts/run_mot_challenge.py --BENCHMARK V2X --SPLIT_TO_EVAL $(split)$(current_agent) --TRACKERS_TO_EVAL sort-$(mode)/$(cross) --METRICS CLEAR --DO_PREPROC False

# Evaludate other matrics
eval2:
	cd TrackEval && python ./scripts/run_mot_challenge.py --BENCHMARK V2X --SPLIT_TO_EVAL $(split)$(current_agent) --TRACKERS_TO_EVAL sort-$(mode)/$(cross) --METRICS HOTA --DO_PREPROC False