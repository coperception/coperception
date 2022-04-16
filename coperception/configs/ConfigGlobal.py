import numpy as np
import math
class ConfigGlobal(object):
	def __init__(self,split,binary=True,only_det=True,code_type='faf',loss_type='faf_loss',savepath='',root=''):

		self.device = None
		self.split = split
		self.savepath = savepath
		self.binary = binary
		self.only_det = only_det
		self.code_type = code_type
		self.loss_type = loss_type #corner_loss faf_loss

		# The specifications for BEV maps
		self.voxel_size = (0.25, 0.25, 0.4)
		self.area_extents = np.array([[-96., 96.], [-96., 96.], [-3., 2.]])
		self.past_frame_skip = 0  # when generating the BEV maps, how many history frames need to be skipped
		self.future_frame_skip = 0  # when generating the BEV maps, how many future frames need to be skipped
		self.num_past_frames_for_bev_seq = 1  # the number of past frames for BEV map sequence
		self.num_past_pcs = 4 #duplicate self.num_past_frames_for_bev_seq

		self.map_dims = [math.ceil((self.area_extents[0][1]-self.area_extents[0][0])/self.voxel_size[0]),\
		math.ceil((self.area_extents[1][1]-self.area_extents[1][0])/self.voxel_size[1]),\
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
		self.MIMO=False
		#debug Motion Classification
		self.motion_state = False
		self.static_thre = 0.2 # speed lower bound

		#debug use_vis
		self.use_vis = True
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
			self.anchor_size = np.asarray([[2.,4.,0],[2.,4.,math.pi/2.],\
							[1.,1.,0],[1.,2.,0.],[1.,2.,math.pi/2.],\
							[3.,12.,0.],[3.,12.,math.pi/2.]]) 
		else:
			self.anchor_size = np.asarray([[2.,4.,0],[2.,4.,math.pi/2.],\
							[2.,4.,-math.pi/4.],[3.,12.,0],[3.,12.,math.pi/2.],[3.,12.,-math.pi/4.]])

		self.category_threshold = [0.4,0.4,0.25,0.25,0.4]
		self.class_map = {'vehicle.audi.a2': 1, 'vehicle.audi.etron': 1, 'vehicle.audi.tt': 1,
						  'vehicle.bmw.grandtourer': 1, 'vehicle.bmw.isetta': 1, 'vehicle.chevrolet.impala': 1,
						  'vehicle.citroen.c3': 1, 'vehicle.dodge_charger.police': 1, 'vehicle.jeep.wrangler_rubicon': 1,
						  'vehicle.lincoln.mkz2017': 1, 'vehicle.mercedes-benz.coupe': 1, 'vehicle.mini.cooperst': 1,
						  'vehicle.mustang.mustang': 1, 'vehicle.nissan.micra': 1, 'vehicle.nissan.patrol': 1,
						  'vehicle.seat.leon': 1, 'vehicle.tesla.cybertruck': 1, 'vehicle.tesla.model3': 1,
						  'vehicle.toyota.prius': 1, 'vehicle.volkswagen.t2': 1, 'vehicle.carlamotors.carlacola': 1,
						  'human.pedestrian': 2, 'vehicle.bh.crossbike': 3, 'vehicle.diamondback.century': 3,
						  'vehicle.gazelle.omafiets': 3, 'vehicle.harley-davidson.low_rider': 3,
						  'vehicle.kawasaki.ninja': 3, 'vehicle.yamaha.yzf': 3}  # background: 0, other: 4
		#self.class_map = {'vehicle.car': 1, 'vehicle.truck': 1, 'vehicle.bus': 1, 'human.pedestrian': 2, 'vehicle.bicycle': 3, 'vehicle.motorcycle': 3}  # background: 0, other: 4
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
			self.nsweeps_forward = 0
			self.skip_frame = 0
			self.num_adj_seqs = 1
