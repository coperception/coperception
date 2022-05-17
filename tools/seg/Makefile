# Path to the original V2X-Sim dataset
original_data_path := /scratch/dm4524/data/V2X-Sim-2
# Where to save the created data
create_data_save_path := /scratch/dm4524/data/V2X-Sim-seg
# Index of the begining scene
scene_begin := 0
# Index of the ending scene + 1
scene_end := 80
# Index of the start agent
from_agent := 0
# Index of the end agent + 1
to_agent := 6

training_script := train_seg.py
# Path to the created training data
training_data := $(create_data_save_path)/train

testing_script := test_seg.py
# Path to the test/val data
testring_data := $(create_data_save_path)/test
# [lowerbound / upperbound]
bound := lowerbound
# [v2v / disco / when2com]
com := disco
batch_size := 4
# [train / test / val]
split := train

log_base := logs
# Train for how many epochs
epoch_num := 100
# The index number of check point to resume
resume_epoch_num := 
# If given, the model will resume from its most recent (by modification time) check point
auto_resume_path := $(log_base)

create_data: 
	python create_data_seg.py --root $(original_data_path) --split $(split) --scene_begin $(scene_begin) --scene_end $(scene_end) --savepath $(create_data_save_path) --from_agent $(from_agent) --to_agent $(to_agent)

train_bound:
	python $(training_script) --data $(training_data) --bound $(bound) --log --batch $(batch_size) --logpath $(log_base) --nepoch $(epoch_num) --auto_resume_path $(auto_resume_path)

train_bound_nc:
	python $(training_script) --data $(training_data) --bound $(bound) --log --batch $(batch_size) --no_cross_road --logpath $(log_base) --nepoch $(epoch_num) --auto_resume_path $(auto_resume_path)

train: 
	python $(training_script) --data $(training_data) --bound $(bound) --log --com $(com) --batch $(batch_size) --logpath $(log_base) --nepoch $(epoch_num) --auto_resume_path $(auto_resume_path)

train_nc:
	python $(training_script) --data $(training_data) --bound $(bound) --log --com $(com) --batch $(batch_size) --no_cross_road --logpath $(log_base) --nepoch $(epoch_num) --auto_resume_path $(auto_resume_path)

train_warp: 
	python $(training_script) --data $(training_data) --bound $(bound) --log --com $(com) --batch $(batch_size) --warp_flag --logpath $(log_base) --nepoch $(epoch_num) --auto_resume_path $(auto_resume_path)

train_warp_nc:
	python $(training_script) --data $(training_data) --bound $(bound) --log --com $(com) --batch $(batch_size) --no_cross_road --warp_flag --logpath $(log_base) --nepoch $(epoch_num) --auto_resume_path $(auto_resume_path)

train_disco:
	python $(training_script) --data $(training_data) --bound $(bound) --log --com $(com) --batch $(batch_size) --resume_teacher $(log_base)/upperbound/with_cross/epoch_$(epoch_num).pth --logpath $(log_base) --nepoch $(epoch_num) --auto_resume_path $(auto_resume_path)

train_disco_nc:
	python $(training_script) --data $(training_data) --bound $(bound) --log --com $(com) --batch $(batch_size) --no_cross_road --resume_teacher $(log_base)/upperbound/no_cross/epoch_$(epoch_num).pth --logpath $(log_base) --nepoch $(epoch_num) --auto_resume_path $(auto_resume_path)

test_bound:
	python test_seg.py --data $(testring_data) --bound $(bound) --vis --resume $(log_base)/$(bound)/with_cross/epoch_$(epoch_num).pth --logpath $(log_base)

test_bound_nc:
	python test_seg.py --data $(testring_data) --bound $(bound) --vis --resume $(log_base)/$(bound)/no_cross/epoch_$(epoch_num).pth --no_cross_road --logpath $(log_base)

test:
	python test_seg.py --data $(testring_data) --bound lowerbound --vis --com $(com) --resume $(log_base)/$(com)/with_cross/epoch_$(epoch_num).pth --logpath $(log_base)

test_nc:
	python test_seg.py --data $(testring_data) --bound lowerbound --vis --com $(com) --resume $(log_base)/$(com)/no_cross/epoch_$(epoch_num).pth --no_cross_road --logpath $(log_base)

test_w:
	python test_seg.py --data $(testring_data) --bound lowerbound --vis --com when2com --resume $(log_base)/when2com/with_cross/epoch_$(epoch_num).pth --inference $(inference)

test_w_nc:
	python test_seg.py --data $(testring_data) --bound lowerbound --vis --com when2com --resume $(log_base)/when2com/with_cross/epoch_$(epoch_num).pth --inference $(inference) --no_cross_road

test_warp:
	python test_seg.py --data $(testring_data) --bound lowerbound --vis --com when2com --resume $(log_base)/when2com_warp/with_cross/epoch_$(epoch_num).pth --warp_flag --inference $(inference)

test_warp_nc:
	python test_seg.py --data $(testring_data) --bound lowerbound --vis --com when2com --resume $(log_base)/when2com_warp/no_cross/epoch_$(epoch_num).pth --warp_flag --no_cross_road --inference $(inference)

visualize:
	python visualizer.py --split $(split) --agent_idx $(agent_idx)


train_disco_resume:
	python $(training_script) --data $(data) --bound $(bound) --log --com $(com) --batch $(batch_size) --resume $(log_base)/disco/with_cross/epoch_$(resume_epoch_num).pth --resume_teacher ./logs/upperbound/with_cross/epoch_100.pth --logpath $(log_base) --nepoch $(epoch_num)

train_disco_resume_nc:
	python $(training_script) --data $(data) --bound $(bound) --log --com $(com) --batch $(batch_size) --resume $(log_base)/disco/no_cross/epoch_$(resume_epoch_num).pth --no_cross_road --resume_teacher ./logs/upperbound/no_cross/epoch_100.pth --logpath $(log_base)  --nepoch $(epoch_num)

train_resume:
	python $(training_script) --data $(data) --bound $(bound) --log --com $(com) --batch $(batch_size) --resume $(log_base)/$(com)/with_cross/epoch_$(resume_epoch_num).pth --logpath $(log_base) --nepoch $(epoch_num)

train_resume_nc:
	python $(training_script) --data $(data) --bound $(bound) --log --com $(com) --batch $(batch_size) --no_cross_road --resume $(log_base)/$(com)/no_cross/epoch_$(resume_epoch_num).pth --logpath $(log_base) --nepoch $(epoch_num)

train_bound_resume:
	python $(training_script) --data $(data) --bound $(bound) --log --batch $(batch_size) --resume $(log_base)/$(bound)/with_cross/epoch_$(resume_epoch_num).pth --logpath $(log_base) --nepoch $(epoch_num)

train_warp_resume:
	python $(training_script) --data $(data) --bound $(bound) --log --com $(com) --batch $(batch_size) --warp_flag --resume $(log_base)/when2com_warp/with_cross/epoch_$(resume_epoch_num).pth --logpath $(log_base) --nepoch $(epoch_num)

train_nc_warp_resume:
	python $(training_script) --data $(data) --bound $(bound) --log --com $(com) --batch $(batch_size) --warp_flag --resume $(log_base)/when2com_warp/no_cross/epoch_$(resume_epoch_num).pth --no_cross_road --logpath $(log_base) --nepoch $(epoch_num)