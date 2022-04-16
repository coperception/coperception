import numpy as np
import math
class Config(object):
	def __init__(self,split,binary=True,only_det=True,code_type='faf',loss_type='faf_loss',savepath='',root='',is_cross_road=False,use_vis=False):
		# for segmentaion task only
		# =========================
		self.num_class = 8
		self.in_channels = 13
		self.nepoch = 10

		self.class_to_rgb = {
			0: [255, 255, 255],  # Unlabeled
			1: [71, 141, 230],  # Vehicles
			2: [122, 217, 209],  # Sidewalk
			3: [145, 171, 100],  # Ground / Terrain
			4: [231, 136, 101],  # Road / Traffic light / Pole
			5: [142, 80, 204],  # Buildings
			6: [224, 8, 50],  # Pedestrian
			7: [106, 142, 34]  # Vegetation
			# 7: [102, 102, 156],  # Walls
			# 0: [55, 90, 80],  # Other
		}

		# Remap pixel values given by carla
		self.classes_remap = {
			0: 0,  # Unlabeled (so that we don't forget this class)
			10: 1,  # Vehicles
			8: 2,  # Sidewalk
			14: 3,  # Ground (non-drivable)
			22: 3,  # Terrain (non-drivable)
			7: 4,  # Road
			6: 4,  # Road line
			18: 4,  # Traffic light
			5: 4,  # Pole
			1: 5,  # Building
			4: 6,  # Pedestrian
			9: 7,  # Vegetation
		}

		self.class_idx_to_name = {
			0: 'Unlabeled',
			1: 'Vehicles',
			2: 'Sidewalk',
			3: 'Ground & Terrain',
			4: 'Road',
			5: 'Buildings',
			6: 'Pedestrian',
			7: 'Vegetation'
		}
		# =========================
		
		self.device = None
		self.split = split
		self.savepath = savepath
		self.binary = binary
		self.only_det = only_det
		self.code_type = code_type
		self.loss_type = loss_type #corner_loss faf_loss

		# The specifications for BEV maps
		self.voxel_size = (0.25, 0.25, 0.4)
		self.area_extents = np.array([[-32., 32.], [-32., 32.], [-8., -3.]]) if is_cross_road else np.array([[-32., 32.], [-32., 32.], [-3., 2.]])
		self.is_cross_road = is_cross_road
		self.past_frame_skip = 3  # when generating the BEV maps, how many history frames need to be skipped
		self.future_frame_skip = 0  # when generating the BEV maps, how many future frames need to be skipped
		self.num_past_frames_for_bev_seq = 1  # the number of past frames for BEV map sequence
		self.num_past_pcs = 1 #duplicate self.num_past_frames_for_bev_seq

		self.map_dims = [math.ceil((self.area_extents[0][1]-self.area_extents[0][0])/self.voxel_size[0]),
						 math.ceil((self.area_extents[1][1]-self.area_extents[1][0])/self.voxel_size[1]),
						 math.ceil((self.area_extents[2][1]-self.area_extents[2][0])/self.voxel_size[2])]
		self.only_det = True
		self.root = root
		#debug Data:
		self.code_type = 'faf'
		self.pred_type = 'motion'
		#debug Loss
		self.loss_type= 'corner_loss'
		#debug MGDA
		self.MGDA=False
		#debug when2com
		self.MIMO=True
		#debug Motion Classification
		self.motion_state = False
		self.static_thre = 0.2 # speed lower bound

		#debug use_vis
		self.use_vis = use_vis
		self.use_map = False

		# The specifications for object detection encode
		if self.code_type in ['corner_1','corner_2']:
			self.box_code_size = 8 #(\delta{x1},\delta{y1},\delta{x2},\delta{y2},\delta{x3},\delta{y3},\delta{x4},\delta{y4})
		elif self.code_type in ['corner_3']:
			self.box_code_size = 10
		elif self.code_type[0] == 'f':
			self.box_code_size = 6 #(x,y,w,h,sin,cos)
		else:
			print(code_type,' code type is not implemented yet!')
			exit()
		

		self.pred_len = 1 #the number of frames for prediction, including the current frame


		#anchor size: (w,h,angle) (according to nuscenes w < h)
		if not self.binary:
			self.anchor_size = np.asarray([[2.,4.,0],[2.,4.,math.pi/2.],
										   [1.,1.,0],[1.,2.,0.],[1.,2.,math.pi/2.],
										   [3.,12.,0.],[3.,12.,math.pi/2.]])
		else:
			self.anchor_size = np.asarray([[2.,4.,0],[2.,4.,math.pi/2.],
										   [2.,4.,-math.pi/4.],[3.,12.,0],[3.,12.,math.pi/2.],[3.,12.,-math.pi/4.]])


		self.category_threshold = [0.4,0.4,0.25,0.25,0.4]
		self.class_map = {'vehicle.car': 1, 'vehicle.emergency.police':1, 'vehicle.bicycle':3, 'vehicle.motorcycle':3, 'vehicle.bus.rigid':2}

		if self.binary:
			self.category_num = 2
		else:
			self.category_num = len(self.category_threshold)
		self.print_feq = 100
		if self.split == 'train':
			self.num_keyframe_skipped = 0  # The number of keyframes we will skip when dumping the data
			self.nsweeps_back = 1  # Number of frames back to the history (including the current timestamp)
			self.nsweeps_forward = 0  # Number of frames into the future (does not include the current timestamp)
			self.skip_frame = 0  # The number of frames skipped for the adjacent sequence
			self.num_adj_seqs = 1  # number of adjacent sequences, among which the time gap is \delta t
		else:
			self.num_keyframe_skipped = 0
			self.nsweeps_back = 1  # Setting this to 30 (for training) or 25 (for testing) allows conducting ablation studies on frame numbers
			self.nsweeps_forward =0
			self.skip_frame = 0
			self.num_adj_seqs = 1
