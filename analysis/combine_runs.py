import numpy as np
import os
from skimage.measure import block_reduce
from scipy.misc import imread
import matplotlib.pylab as plt
import cPickle as pkl
import scipy.signal
import numpy.fft as fft
import sys

import matplotlib.cm as cm
import re
import itertools

from libtiff import TIFF

import PIL.Image as Image
import libtiff

import optparse

# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import matplotlib.animation as manimation


#import cv2

#import tifffile as tiff

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'valid')

parser = optparse.OptionParser()
parser.add_option('--headless', action="store_true", dest="headless", default=False, help="run in headless mode, no figs")
parser.add_option('--freq', action="store", dest="target_freq", default="0.05", help="stimulation frequency")
parser.add_option('--reduce', action="store", dest="reduce_val", default="2", help="block_reduce value")
parser.add_option('--sigma', action="store", dest="gauss_kernel", default="0", help="size of Gaussian kernel for smoothing")
parser.add_option('--format', action="store", dest="im_format", default="tif", help="saved image format")
parser.add_option('--fps', action="store", dest="fps", default="60.", help="camera acquisition rate (fps)")
parser.add_option('-k', '--keys', type='string', action="append", dest="keys", default=[], help="condition type(s) (stimulus, blank, etc.)")

# GET OPTIONS ARGS:
(options, args) = parser.parse_args()
im_format = '.'+options.im_format
target_freq = float(options.target_freq)
fps = float(options.fps)
keys = options.keys
print keys

reduce_factor = (int(options.reduce_val), int(options.reduce_val))
if reduce_factor[0] > 0:
	reduceit=1
else:
	reduceit=0
gsigma = int(options.gauss_kernel)
headless = options.headless
if headless:
	import matplotlib as mpl
	mpl.use('Agg')
import matplotlib.pylab as plt
import matplotlib.cm as cm


# GET PATHS:
datadir = sys.argv[1]
sessiondir = os.path.split(datadir)[0]
run_type = os.path.split(datadir)[1]

# crop_fov = 0
# if len(sys.argv) > 2:
# 	cropped = sys.argv[2].split(',')
# 	[strtX, endX, strtY, endY] = [int(i) for i in cropped]
# 	crop_fov = 1

run_reps = [s for s in os.listdir(sessiondir) if run_type in s]
print "N RUN REPS: ", len(run_reps)

# CYCLE THRU RUNS TO PULL DATA (AND AVG):
D = dict()
for run_num in run_reps:
	path_to_run = os.path.join(sessiondir, run_num)
	runs = os.listdir(path_to_run)
	conds = [r for r in runs for key in keys if key in r]

	for cond in conds:
		if cond not in D:
			print("creating new key... %s " % cond)
			D[cond] = dict()

	for cond in conds:
		imdir = os.path.join(path_to_run, cond)

		files = os.listdir(imdir)
		files = sorted([f for f in files if os.path.splitext(f)[1] == '.tif'])

		# METHOD 1:
		#sample = imread(os.path.join(imdir, files[0]))

		# METHOD 2:
		tiff = TIFF.open(os.path.join(imdir, files[0]), mode='r')
		sample = tiff.read_image().astype('float')
		print sample.dtype, [sample.max(), sample.min()]
		tiff.close()

		if crop_fov:
			sample = sample[strtX:endX, strtY:endY]

		if reduceit:
			sample = block_reduce(sample, reduce_factor, func=np.mean)

		# READ IN THE FRAMES:
		stack = np.empty((sample.shape[0], sample.shape[1], len(files)))
		print len(files)

		print('copying files, run: %s' % imdir)

		for i, f in enumerate(files):

			if i % 100 == 0:
				print('%d images processed...' % i)
			# print f
			#im = imread(os.path.join(imdir, f)).astype('float')
			
			tiff = TIFF.open(os.path.join(imdir, f), mode='r')
			im = tiff.read_image().astype('float')
			tiff.close()

			if crop_fov:
				im = im[strtX:endX, strtY:endY]

			if reduceit:
				im = block_reduce(im, reduce_factor, func=np.mean)

			stack[:,:,i] = im #im_reduced

		for i in range(stack.shape[2]):
			print i
			pix = stack[:,:,i]
			stack[:,:,i] = scipy.signal.detrend(pix, type='linear') # default = linear; constant (only mean subtracted)

		D[cond][os.path.split(path_to_run)[1]] = stack

avg_stacks = dict()
for cond in D.keys():
	stack = [];
	[stack.append(D[cond][i]) for i in D[cond].keys()]
	nframes = [s.shape[2] for s in stack] # get num of frames for each run
	cutoff = min(nframes) # just use the min number of frames...
	stack = [s[:,:,0:cutoff] for s in stack] # and cut them off
	avg_stacks[cond] = sum(stack) / len(stack)


	framedir = os.path.join(sessiondir, 'averaged_movies') #, cond)
	if not os.path.exists(framedir):
		os.makedirs(framedir)

	print("saving averaged frames in: %s" % framedir)


	# FFMPEG_BIN = "ffmpeg" # on Linux ans Mac OS
	# command = [ FFMPEG_BIN,
	#         '-y', # (optional) overwrite output file if it exists
	#         '-f', 'rawvideo',
	#         '-vcodec','rawvideo',
	#         '-s', '420x360', # size of one frame
	#         '-pix_fmt', 'rgb24',
	#         '-r', '24', # frames per second
	#         '-i', '-', # The imput comes from a pipe
	#         '-an', # Tells FFMPEG not to expect any audio
	#         '-vcodec', 'mpeg',
	#         'my_output_videofile.mp4' ]

	# import subprocess as sp
	# pipe = sp.Popen( command, stdin=sp.PIPE, stderr=sp.PIPE)

	# vname = '%s/%s_video.avi' % (framedir, cond)
	# H, W, F = avg_stacks[cond].shape

	# Define the codec and create VideoWriter object
	# fourcc = cv2.VideoWriter_fourcc(*'XVID'
	
	# fourcc = cv2.cv.CV_FOURCC(*'XVID')
	# vid = cv2.VideoWriter(vname,fourcc,60,(W,H),isColor=False)

	# fourcc = cv2.cv.CV_FOURCC('I','4','2','0')
	# fourcc=cv2.cv.CV_FOURCC('I', 'Y', 'U', 'V')
	# writer = cv2.cv.CreateVideoWriter('out.avi', fourcc, fps, (W, H), 1)

	for i in range(F):
		oldmin = avg_stacks[cond].min() #-2**15 #np.finfo('float64').min
		oldmax = avg_stacks[cond].max() # 2**15 - 1 #np.finfo('float64').max
		oldrange = oldmax - oldmin
		newmin = 0
		newmax = 255
		newrange = newmax - newmin

		# vid = cv2.VideoWriter(vname,fourcc,1,(W,H),isColor=False)
		frame = (((avg_stacks[cond][:,:,i] - oldmin) * newrange) / oldrange) + newmin
		
		fname = '%s/%0.4i.png' % (framedir, i)
		tiff = TIFF.open(fname, mode='w')
		tiff.write_image(frame)
		tiff.close()

	# 	vid.write(im)
	# 	cv2.cv.WriteFrame(writer, Image.fromarray(frame))

	# cv2.destroyAllWindows()
	# vid.release()



		# image_array = Image.fromarray(np.uint8(avg_stacks[cond]*255))
		# pipe.proc.stdin.write( image_array.tostring() )


		# fname = '%s/%0.4i.tif' % (framedir, i)
		# image = Image.fromarray(np.uint8(M[:,:,i]*255))
		# #image = Image.fromarray(M[:,:,i])
		# # image = Image.fromarray(meanD[:,:,i])
		# tiff = TIFF.open(fname, mode='w')
		# tiff.write_image(image)
		# tiff.close()

# FFMpegWriter = manimation.writers['ffmpeg']
# metadata = dict(title='Movie Test', artist='Matplotlib',
#         comment='Movie support!')
# writer = FFMpegWriter(fps=15, metadata=metadata)

# fig = plt.figure()
# l, = plt.plot([], [], 'k-o')

# plt.xlim(-5, 5)
# plt.ylim(-5, 5)

# x0,y0 = 0, 0

# frames = os.listdir(framedir)

# with writer.saving(fig, "writer_test.mp4", 100):
# 	for i in frames:
# 		writer.grab_frame()

# # for i in range(S.shape[2]):
# # 	fname = '%s/%0.4i.tif' % (framedir, i)
# # 	imarray = S[:,:,i]
# # 	#tiff = TIFF.open(fname, mode='w')
# # 	#tiff.imsave(fname, imarray)
# # 	#tiff.close()
# # 	#plt.imshow(imarray)
# # 	#plt.show()
# # 	cv2.imwrite(fname, imarray)

# # 	#img = scipy.misc.toimage(S[:,:,i], high=imarray.max(), low=imarray.min(), mode='I')
# # 	#img = scipy.misc.toimage(S[:,:,i], high=65536, low=0, mode='I')
# # 	#img.save(fname)

