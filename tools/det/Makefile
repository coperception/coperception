# Path to the original V2X-Sim dataset
original_data_path := /scratch/dm4524/data/V2X-Sim-2
# Where to save the created data
create_data_save_path := /scratch/dm4524/data/V2X-Sim-det
# Index of the begining scene
scene_begin := 0
 # Index of the ending scene + 1
scene_end := 80
# Index of the start agent
from_agent := 0
# Index of the end agent + 1
to_agent := 6

training_script := train_codet.py
# Path to the created training data
training_data := $(create_data_save_path)/train

testing_script := test_codet.py
# Path to the test/val data
testing_data := $(create_data_save_path)/test
# [lowerbound / upperbound]
bound := lowerbound
# [v2v / disco / when2com]
com := v2v
batch_size := 4
# [train / test / val]
split := train
# Where to store the logs
log_base := logs
# Train for how many epochs
epoch_num := 100
# If given, the model will resume from its most recent (by modification time) check point
auto_resume_path := $(log_base)
# compress_level
compress_level := 0

create_data:
	python create_data_det.py --root $(original_data_path) --split $(split) --scene_begin $(scene_begin) --scene_end $(scene_end) --savepath $(create_data_save_path) --from_agent $(from_agent) --to_agent $(to_agent)

train_bound:
	python $(training_script) --data $(training_data) --bound $(bound) --log --batch $(batch_size) --auto_resume_path $(auto_resume_path) --nepoch $(epoch_num) --logpath $(log_base) --compress_level $(compress_level)

train_bound_nc:
	python $(training_script) --data $(training_data) --bound $(bound) --log --batch $(batch_size) --no_cross_road --auto_resume_path $(auto_resume_path) --nepoch $(epoch_num) --logpath $(log_base) --compress_level $(compress_level)

train:
	python $(training_script) --data $(training_data) --bound lowerbound --com $(com) --log --batch $(batch_size) --auto_resume_path $(auto_resume_path) --nepoch $(epoch_num) --logpath $(log_base) --compress_level $(compress_level)

train_nc:
	python $(training_script) --data $(training_data) --bound lowerbound --com $(com) --log --batch $(batch_size) --no_cross_road --auto_resume_path $(auto_resume_path) --nepoch $(epoch_num) --logpath $(log_base) --compress_level $(compress_level)

train_warp:
	python $(training_script) --data $(training_data) --bound $(bound) --com $(com) --log --batch $(batch_size) --warp_flag --auto_resume_path $(auto_resume_path) --nepoch $(epoch_num) --logpath $(log_base) --compress_level $(compress_level)

train_warp_nc:
	python $(training_script) --data $(training_data) --bound $(bound) --com $(com) --log --batch $(batch_size) --warp_flag --no_cross_road --auto_resume_path $(auto_resume_path) --nepoch $(epoch_num) --logpath $(log_base) --compress_level $(compress_level)

train_disco:
	python $(training_script) --data $(training_data) --bound $(bound) --com disco --log --batch $(batch_size) --kd_flag 1 --resume_teacher $(log_base)/upperbound/with_cross/epoch_$(epoch_num).pth --auto_resume_path $(auto_resume_path) --logpath $(log_base) --nepoch $(epoch_num) --compress_level $(compress_level)

train_disco_nc:
	python $(training_script) --data $(training_data) --bound $(bound) --com disco --log --batch $(batch_size) --kd_flag 1 --resume_teacher	$(log_base)/upperbound/no_cross/epoch_$(epoch_num).pth --no_cross_road --auto_resume_path $(auto_resume_path) --logpath $(log_base) --nepoch $(epoch_num) --compress_level $(compress_level)

test_bound:
	python $(testing_script) --data $(testing_data) --bound $(bound) --resume $(log_base)/$(bound)/with_cross/epoch_$(epoch_num).pth --tracking --logpath $(log_base)

test_bound_nc:
	python $(testing_script) --data $(testing_data) --bound $(bound) --resume $(log_base)/$(bound)/no_cross/epoch_$(epoch_num).pth --no_cross_road --tracking --logpath $(log_base)

test:
	python $(testing_script) --data $(testing_data) --bound lowerbound --resume $(log_base)/$(com)/with_cross/epoch_$(epoch_num).pth --com $(com) --tracking --logpath $(log_base)

test_nc:
	python $(testing_script) --data $(testing_data) --bound lowerbound --resume $(log_base)/$(com)/no_cross/epoch_$(epoch_num).pth --com $(com) --no_cross_road --tracking --logpath $(log_base)

test_w:
	python $(testing_script) --data $(testing_data) --bound lowerbound --resume $(log_base)/when2com/with_cross/epoch_$(epoch_num).pth --com when2com --inference $(inference) --tracking --logpath $(log_base)

test_w_nc:
	python $(testing_script) --data $(testing_data) --bound lowerbound --resume $(log_base)/when2com/no_cross/epoch_$(epoch_num).pth --com when2com --no_cross_road --inference $(inference) --tracking --logpath $(log_base)

test_warp:
	python $(testing_script) --data $(testing_data) --bound lowerbound --resume $(log_base)/when2com_warp/with_cross/epoch_$(epoch_num).pth --com when2com --warp_flag --inference $(inference) --tracking --logpath $(log_base)

test_warp_nc:
	python $(testing_script) --data $(testing_data) --bound lowerbound --resume $(log_base)/when2com_warp/no_cross/epoch_$(epoch_num).pth --com when2com --warp_flag --no_cross_road --inference $(inference) --tracking --logpath $(log_base)
