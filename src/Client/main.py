import os
import random
import math
import numbers
import collections
import numpy as np
import pandas as pd
import torch
import cv2
import scipy.ndimage
from PIL import Image, ImageOps
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from models import mobilenetv2
from opts import parse_opts
from spatial_transforms import *
from temporal_transforms import *
from torchvision import transforms
from torchvision.transforms import ToPILImage

from controls import performAction

import time
try:
    import accimage
except ImportError:
    accimage = None
    

"""
Initialize model - 
Loading the pretrained model that is received from the server
"""
def intialize(model):
      
    if not opt.no_cuda:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)


        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path)
            model.load_state_dict(pretrain)
    else:
        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path,map_location=torch.device('cpu'))
            model.load_state_dict(pretrain)
    
"""
Preprocessing the continuous stream of video frames for live prediction
Applying Temporal transformation to reduce the frames to 16 frames
then applying spatial transformation to scale, and crop the image

"""
def pre_process(list_of_frames):
	#global save_cntr
	if opt.train_crop == 'random':
		crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
	elif opt.train_crop == 'corner':
		crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)	
	elif opt.train_crop == 'center':
		crop_method = MultiScaleCornerCrop(
			opt.scales, opt.sample_size, crop_positions=['c'])
	
	frame_indices = range(1,len(list_of_frames))
	
	# Temporal Transform
	temporal_transform = TemporalRandomCrop(opt.sample_duration, opt.downsample)
       
	# Applying temporal transform
	frame_indices = temporal_transform(frame_indices)
	clip = [cv2.resize(list_of_frames[i],(112,112),interpolation=cv2.INTER_AREA) for i in frame_indices]
	
	# Spatial Transform
	spatial_transform = Compose([
		Scale(int(opt.sample_size / opt.scale_in_test)),
		CornerCrop(opt.sample_size, opt.crop_position_in_test),
		#crop_method,
		ToTensor(opt.norm_value) #, norm_method
	])

	# Applying Spatial transform
	spatial_transform.randomize_parameters()
	clip = [spatial_transform(Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))) for img in clip] #converting image fromBGR to RGB
	
	im_dim = clip[0].size()[-2:]

	# clip is a list of tensors here, if we use torch.stack(clip,0) it will give clip as a tensor of size - 16x3x112x112
	# then permute will chang dimensions of the clip tensor to 3x16x112x112
	clip = torch.stack(clip, 0).permute(1, 0, 2, 3) 

	# unsqueeze will add another dimension - 1x3x16x112x112, this needed because loader only accepts in batches so 1 i the batchsize 	here. 
	clip = clip.unsqueeze(0)
	if not opt.no_cuda:
		clip = clip.cuda()
	
	return clip
	
if __name__ == '__main__':

	# Loading class labels (gestures)
	df=pd.read_csv('./recorded_gestures/class_labels.csv', sep=',',header=None)
	labels = df.values
	print(labels)

	# Initializing Opts
	opt = parse_opts()
	opt.confidence_threshold = 0.4
	opt.frame_to_send = 60
	opt.downsample = 2
	opt.width_mult = 1.0
	opt.n_classes = len(labels)
	opt.sample_size = 112
	opt.scale_in_test = 1.0
	opt.no_cuda = True
	opt.model = "mobilenetv2"
	opt.arch = "mobilenetv2"
	opt.pretrain_path = f"./FineTuned/output.pth"

	opt.scales = [opt.initial_scale]
	for i in range(1, opt.n_scales):
		opt.scales.append(opt.scales[-1] * opt.scale_step)

	opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
	opt.std = get_std(opt.norm_value)

	# Normalization (NOT USED)
	if opt.no_mean_norm and not opt.std_norm:
	    norm_method = Normalize([0, 0, 0], [1, 1, 1])
	elif not opt.std_norm:
	    norm_method = Normalize(opt.mean, [1, 1, 1])
	else:
	    norm_method = Normalize(opt.mean, opt.std)
	

	# Model definition
	model = mobilenetv2.get_model(
		    num_classes=opt.n_classes,
		    sample_size=opt.sample_size,
		    width_mult=opt.width_mult)
	

	# Model Initialization
	intialize(model)
	model.eval()  
	
	save_cntr=0  

	# define a video capture object
	vid = cv2.VideoCapture(0)
	list_of_frames = []
	cnt=0
	while(True):
		
		# Capture the video frame
		# by frame
		ret, frame = vid.read()
		#frame = cv2.flip(frame,1)

		# Display the resulting frame
		cv2.imshow('webcam', frame)
		#print(frame.shape)
	
		list_of_frames.append(frame)
		cnt+=1
		if(cnt==opt.frame_to_send):
			tensor_out = model(pre_process(list_of_frames))
			tensor_out = F.softmax(tensor_out, dim=1)
			out = np.squeeze((tensor_out.cpu().detach().numpy()))
			argmax = np.argmax(out)
			#print(cnt)
			classname = labels[argmax][0]
			print("ClassName:",classname,"Confidence:",out[argmax])

			if out[argmax] >= opt.confidence_threshold:
				cv2.destroyAllWindows()
				performAction(classname,vid)
			list_of_frames=[]
			cnt=0
		
		# the 'q' button is set as the
		# quitting button you may use any
		# desired button of your choice
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	
	# After the loop release the cap object
	vid.release()
	# Destroy all the windows
	cv2.destroyAllWindows()
 


		

