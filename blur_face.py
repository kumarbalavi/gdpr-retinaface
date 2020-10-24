from __future__ import print_function
import os
import sys
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import time
from tqdm import tqdm
import shutil

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./weights/mobilenet0.25_Final.pth',
					type=str, help='Trained state_dict file path to open')
parser.add_argument("--input_path", type=str, default="curve/test.jpg", help="path to input file")
parser.add_argument("--output_path", type=str, default="curve/test-out.jpg", help="output file path")
parser.add_argument("--is_video", type=bool, default=False, help="process video or image")
parser.add_argument("--save_video", type=bool, default=True, help="save output video or not")
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')
parser.add_argument('--scale', type=float, default=1.0, help='reduce original video by scale factor for faster processing')
parser.add_argument('--size', type=int, default=8, help='size to downsample each object for gdpr')
parser.add_argument('--extend', type=float, default=1.0, help='extend crop area')
parser.add_argument('--confidence_threshold', default=0.01, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=8000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=1750, type=int, help='keep_top_k')
parser.add_argument('--obj_min_size', default=8, type=int, help='min size od object to apply gdpr')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
parser.add_argument('--keep_time', default=25, type=int, help='keep detection for some frames')
parser.add_argument('--show_window', default=False, type=bool, help='show output video in a window')
parser.add_argument('--encoder_settings', default='ffmpeg -start_number 0 -i tmp/frame%6d.jpg -c:v libx264 -crf 26 -preset fast -c:a aac -b:a 128k ', type=str, help='setting for encoder such as ffmpeg')
args = parser.parse_args()

device = torch.device("cpu" if args.cpu else "cuda")

def check_keys(model, pretrained_state_dict):
	ckpt_keys = set(pretrained_state_dict.keys())
	model_keys = set(model.state_dict().keys())
	used_pretrained_keys = model_keys & ckpt_keys
	unused_pretrained_keys = ckpt_keys - model_keys
	missing_keys = model_keys - ckpt_keys
	assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
	return True

def remove_prefix(state_dict, prefix):
	''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
	#print('remove prefix \'{}\''.format(prefix))
	f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
	return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
	#print('Loading pretrained model from {}'.format(pretrained_path))
	if load_to_cpu:
		pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
	else:
		device = torch.cuda.current_device()
		pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
	if "state_dict" in pretrained_dict.keys():
		pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
	else:
		pretrained_dict = remove_prefix(pretrained_dict, 'module.')
	check_keys(model, pretrained_dict)
	model.load_state_dict(pretrained_dict, strict=False)
	return model

def init_net(arch, cpu):

	torch.set_grad_enabled(False)
	cfg = None
	if arch == "mobile0.25":
		cfg = cfg_mnet
	elif arch == "resnet50":
		cfg = cfg_re50

	# net and model
	net = RetinaFace(cfg=cfg, phase = 'test')
	net = load_model(net, args.trained_model, cpu)
	net.eval()
	#print('Finished loading model!')
	#print(net)
	cudnn.benchmark = True
	net = net.to(device)
	return net, cfg

def detect(img_raw, prior_data, cfg, net):
	resize = 1

	# preprocess frame for detection network
	tic = time.time()
	img = np.float32(img_raw)

	im_height, im_width, _ = img.shape
	scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
	img -= (104, 117, 123)
	img = img.transpose(2, 0, 1)
	img = torch.from_numpy(img).unsqueeze(0)
	img = img.to(device)
	scale = scale.to(device)

	# detect objects by network
	tic = time.time()
	loc, conf, landms = net(img)  # forward pass
	
	# decode detection results
	tic = time.time()        
	boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
	boxes = boxes * scale / resize
	boxes = boxes.cpu().numpy()
	scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

	# ignore low scores
	inds = np.where(scores > args.confidence_threshold)[0]
	boxes = boxes[inds]
	# landms = landms[inds]
	scores = scores[inds]

	# keep top-K before NMS
	order = scores.argsort()[::-1][:args.top_k]
	boxes = boxes[order]
	# landms = landms[order]
	scores = scores[order]

	# do NMS
	dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
	keep = py_cpu_nms(dets, args.nms_threshold)
	# keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
	dets = dets[keep, :]
	# landms = landms[keep]

	# keep top-K faster NMS
	dets = dets[:args.keep_top_k, :]

	return dets

def apply_gdpr(dets, img_raw0, mask_stat):

	mask_stat[:,:,:] -= 1
	np.clip(mask_stat,0,args.keep_time)

	# show image
	if args.save_video:
		for b in dets:
			if b[4] < args.vis_thres:
				continue
			text = "{:.4f}".format(b[4])
			b = list(map(int, b))

			w = (b[2]-b[0])*0.5
			h = (b[3]-b[1])*0.5

			x1 = max(0, int((b[0]-w*args.extend)*args.scale))
			y1 = max(0, int((b[1]-h*args.extend)*args.scale))
			x2 = min(img_raw0.shape[1]-1, int((b[2]+w*args.extend)*args.scale))
			y2 = min(img_raw0.shape[0]-1, int((b[3]+h*args.extend)*args.scale))

			if (x2-x1 > args.obj_min_size and y2-y1 > args.obj_min_size):
				#print(y2-y1, x2-x1, org_frame.size)
				mask_stat[y1:y2, x1:x2, :] = args.keep_time
	
	#mask = np.zeros(mask_stat.shape, np.uint8)
	img_raw1a = cv2.resize(img_raw0,(int(img_raw0.shape[1]*0.05),int(img_raw0.shape[0]*0.05)),0)
	img_raw1b = cv2.resize(img_raw1a,(int(img_raw0.shape[1]),int(img_raw0.shape[0])),0)
	img_raw0[mask_stat>0] = img_raw1b[mask_stat>0]
	

def main():

	print('CPU_USE is ' + str(args.cpu) + '. Applying GDPR to input: ' + args.input_path + ' with output: ' + args.output_path + ' by using ' + args.network + ' model')
		
	# initialize detection network
	net, cfg = init_net(arch=args.network, cpu=args.cpu)
	if (net is None):
		return
	
	print(args.keep_time)

	# process video
	if (args.is_video is True):

		# read video
		cap = cv2.VideoCapture(args.input_path)
		assert cap.isOpened(), 'Cannot capture source'

		# get video feed dimensions
		v_width  = int(cap.get(3))
		v_height = int(cap.get(4))
		v_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

		# init window to show video output
		if (args.show_window):
			cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

			# setup bbox for text in output
			t_size = cv2.getTextSize(" ", cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]

		if (os.path.isdir('tmp')):
			shutil.rmtree('tmp')

		command = 'mkdir tmp'
		os.system(command)

		# frame counter and fps debug info
		frames = fps = 0
		start = time.time()
		
		# set detection settings
		priorbox = PriorBox(cfg, image_size=(int(v_height/args.scale), int(v_width/args.scale)))
		priors = priorbox.forward()
		priors = priors.to(device)
		prior_data = priors.data

		# gdpr mask
		mask_stat = np.zeros((v_height,v_width,3), np.int32)

		# read video
		with tqdm(total=v_total, file=sys.stdout) as pbar:
			ret = True
			while (ret is True):

				# frame extraction
				tic = time.time()
				ret, img_raw0 = cap.read()

				# if no more frames
				if (ret is False):
					break
				
				# resize original frame for faster processing
				img_raw = cv2.resize(img_raw0, (int(v_width/args.scale), int(v_height/args.scale)))
				
				# detect objects by network
				dets = detect(img_raw=img_raw, prior_data=prior_data, cfg=cfg, net=net)

				# apply gdpr
				apply_gdpr(dets, img_raw0, mask_stat)

				# vizualize output video in window
				if (args.show_window):

					# FPS PRINTING
					cv2.rectangle(img_raw0, (0, 0), (175, 20), (0, 0, 0), -1)
					cv2.putText(img_raw0,"FPS : %3.2f" % (fps), (0, t_size[1] + 4),
								cv2.FONT_HERSHEY_PLAIN, 1,
								[255, 255, 255], 1)

					cv2.imshow('frame', img_raw0)
					if cv2.waitKey(1) & 0xFF == ord('q'):
						break

				# save output frames
				if (args.save_video):
					path_out = 'tmp/frame' + str(frames).zfill(6) + '.jpg'
					cv2.imwrite(path_out, img_raw0)

				# update frame counter
				frames += 1
				fps = frames / (time.time() - start)  
				pbar.update(1)

			# convert output frames to video
			command = args.encoder_settings + args.output_path
			os.system(command)

			shutil.rmtree('tmp')

		# close video file
		cap.release()

	# process image	
	else:

		# read image
		img_raw0 = cv2.imread(args.input_path)
		
		# get video feed dimensions
		v_width  = img_raw0.shape[1]
		v_height = img_raw0.shape[0]
		
		# set detection settings
		priorbox = PriorBox(cfg, image_size=(int(v_height/args.scale), int(v_width/args.scale)))
		priors = priorbox.forward()
		priors = priors.to(device)
		prior_data = priors.data

		# resize original frame for faster processing
		img_raw = cv2.resize(img_raw0, (int(v_width/args.scale), int(v_height/args.scale)))

		# gdpr mask
		mask_stat = np.zeros((v_height,v_width,3), np.int32)
		
		# detect objects by network
		dets = detect(img_raw=img_raw, prior_data=prior_data, cfg=cfg, net=net)

		# apply gdpr
		apply_gdpr(dets, img_raw0, mask_stat)

		# vizualize output video in window
		if (args.show_window):

			# FPS PRINTING
			cv2.rectangle(img_raw0, (0, 0), (175, 20), (0, 0, 0), -1)
			cv2.putText(img_raw0,"FPS : %3.2f" % (fps), (0, t_size[1] + 4),
						cv2.FONT_HERSHEY_PLAIN, 1,
						[255, 255, 255], 1)

			cv2.imshow('frame', img_raw0)

		cv2.imwrite(args.output_path, img_raw0)

	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()

	