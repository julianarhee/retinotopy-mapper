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

# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import matplotlib.animation as manimation


#import cv2

#import tifffile as tiff

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'valid')


imdir = sys.argv[1]

crop_fov = 0
if len(sys.argv) > 2:
	cropped = sys.argv[2].split(',')
	[strtX, endX, strtY, endY] = [int(i) for i in cropped]
	crop_fov = 1

files = os.listdir(imdir)
files = sorted([f for f in files if os.path.splitext(f)[1] == '.tif'])
print files[-1]

# FIND CYCLE STARTS:
# This will depend on the file-saving name -- if 0'ed for frame # with each cycle, can just look for 0 in the fn
# Otherwise, prob better to just use position, which is also stored in brackets [x, y]

# strtidxs = []
# strts = [i for i,f in enumerate(files) if '_0_' in f]
# for f in [files[i] for i in strts]:
# 	idx = f.split('_')[3]
# 	strtidxs.append(int(idx))

# strtidxs[0:-3]

#files = files[0:strts[-1]]
# files = files[0:int(round(len(files)*0.5))]

condition = os.path.split(imdir)[1]
positions = [re.findall("\[([^[\]]*)\]", f) for f in files]
plist = list(itertools.chain.from_iterable(positions))
positions = [map(float, i.split(',')) for i in plist]
if 'H-Up' in condition:
	find_cycs = list(itertools.chain.from_iterable(np.where(np.diff([p[1] for p in positions]) < 0)))
if 'H-Down' in condition:
	find_cycs = list(itertools.chain.from_iterable(np.where(np.diff([p[1] for p in positions]) > 0)))
if 'V-Left' in condition:
	find_cycs = list(itertools.chain.from_iterable(np.where(np.diff([p[0] for p in positions]) < 0)))
if 'V-Right' in condition:
	find_cycs = list(itertools.chain.from_iterable(np.where(np.diff([p[0] for p in positions]) > 0)))
idxs = [i+1 for i in find_cycs]

# strt_idxs.append(0)
# strt_idxs = sorted(strt_idxs)
# strt_idxs.append(len(files))
# idxs = strt_idxs

# METHOD 1:
#sample = imread(os.path.join(imdir, files[0]))

# METHOD 2:
tiff = TIFF.open(os.path.join(imdir, files[0]), mode='r')
sample = tiff.read_image().astype('float')
print sample.dtype, [sample.max(), sample.min()]
tiff.close()

# METHOD 3:
#sample = tiff.imread(os.path.join(imdir, files[0]))

# sample = cv2.imread(os.path.join(imdir, files[0]), -1)
# print sample.shape
# print sample.dtype, [sample.max(), sample.min()]
# plt.imshow(sample)
# plt.show()

# # Divide into cycles:
# chunks = []
# for i in range(0, len(idxs)-1):
# 	print i
# 	chunks.append(files[idxs[i]:idxs[i+1]])

# for chunk in chunks:

# 	for i, f in enumerate(chunk):

# 		if i % 100 == 0:
# 			print('%d images processed...' % i)
# 		# print f
# 		im = imread(os.path.join(imdir, f)).astype('float')

# If need to crop, can just replace strtX, strtY, etc.
# Otherwise, just use the whole thing.

# sample = sample[30:-1, 50:310]
# sample = sample[:,70:300]
#sample = sample[20:230,40:275]

if crop_fov:
	sample = sample[strtX:endX, strtY:endY]

	# strtX = 0 #35
	# endX = -1 #251
	# strtY = 0 #50
	# endY = -1 #275

# print "FIRST", sample.dtype
# sample = block_reduce(sample, reduce_factor)

# plt.figure()
# plt.imshow(sample)
# plt.show()


# cycle_dur = 10.
# sampling_rate = 60. #np.mean(np.diff(sorted(strt_idxs)))/cycle_dur #60.0
# reduce_factor = (4, 4)
# cache_file = True
# target_freq = 0.1
# binspread = 0

# print "FIRST", sample.dtype
# sample = block_reduce(sample, reduce_factor, func=np.mean)


# READ IN THE FRAMES:
stack = np.empty((sample.shape[0], sample.shape[1], len(files)))
print len(files)

print('copying files')

for i, f in enumerate(files):

	if i % 100 == 0:
		print('%d images processed...' % i)
	# print f
	#im = imread(os.path.join(imdir, f)).astype('float')
	
	tiff = TIFF.open(os.path.join(imdir, f), mode='r')
	im = tiff.read_image().astype('float')
	tiff.close()

	#im = tiff.imread(os.path.join(imdir, f))
	#im = cv2.imread(os.path.join(imdir, f), -1)

	#im = im[strtX:endX,strtY:endY]
	# print im.shape

	#im_reduced = block_reduce(im, reduce_factor, func=np.mean)
	stack[:,:,i] = im #im_reduced


D = []
prev = 0
for i in idxs:
    D.append(stack[:,:,prev:i])
    prev = i
D.append(stack[:,:,prev:])

del stack

nframes_per_cycle = [d.shape[2] for d in D]

#xtra = [i for i,d in enumerate(D) if d.shape[2]!=max(nframes_per_cycle)]
print np.mean(nframes_per_cycle)

# good = 0
# for i,xt in enumerate(xtra):
# 	plt.subplot(1, len(xtra)+1, i)
# 	currbin = positions[idxs[xt]:idxs[xt+1]]
# 	posx = [currbin[x][0] for x in range(len(currbin))]
# 	posy = [currbin[x][1] for x in range(len(currbin))]
# 	plt.plot(np.diff(posy))
# 	plt.title('nframes: %i' % int(nframes_per_cycle[xt]))

# while not good in xtra:
# 	good += 1
# plt.subplot(1,len(xtra)+1, i+1)
# currbin = positions[idxs[good]:idxs[good+1]]
# posx = [currbin[x][0] for x in range(len(currbin))]
# posy = [currbin[x][1] for x in range(len(currbin))]
# plt.plot(np.diff(posy))
# plt.title('nframes: %i' % int(nframes_per_cycle[good]))


# for x in xtra:
# 	D[x] = D[x][:,:,0:595] 

D = map(lambda x: x[:,:,0:min(nframes_per_cycle)], D)

meanD = sum(D) / len(D)
print meanD.shape

# NORMALIZE with min=0, max=1
M = (meanD - meanD.min()) / (meanD.max() - meanD.min())

# SUBTRACT FIRST FRAME FROM EACH:
# S = np.empty((meanD.shape[0], meanD.shape[1], meanD.shape[2]), np.uint16)
# for i in range(1,meanD.shape[2]):
# 	S[:,:,i-1] = meanD[:,:,i] - meanD[:,:,0]

# del D


# os.path.split(imdir)[0]
condition = os.path.split(imdir)[1]
framedir = os.path.join(os.path.split(imdir)[0], 'processed', condition)
if not os.path.exists(framedir):
	os.makedirs(framedir)

print framedir
for i in range(M.shape[2]):
	fname = '%s/%0.4i.png' % (framedir, i)
	#image = Image.fromarray(np.uint8(M[:,:,i]*255))
	image = Image.fromarray(M[:,:,i])
	tiff = TIFF.open(fname, mode='w')
	tiff.write_image(image)
	tiff.close()

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

