#!/usr/bin/env python2

'''

This script analyzes data acquired using flashBar.py.

Run:  python find_depth.py /path/to/imaging/directory [strtX, endX, strtY, endY]

It will output change in response to flashing rectangles at 2 diff locs (normalized to baseline).

'''


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

#import PIL.Image as Image
# import libtiff
#import cv2

#import tifffile as tiff

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'valid')


imdir = sys.argv[1]
reduce_factor = (2,2)

crop_fov = 0
if len(sys.argv) > 2:
	cropped = sys.argv[2].split(',') # include 2nd arg after imdir for strt/end, comma-sep
	[strtX, endX, strtY, endY] = [int(i) for i in cropped]
	crop_fov = 1

files = os.listdir(imdir)
files = sorted([f for f in files if os.path.splitext(f)[1] == '.png'])
print files[-1]
# cutit = int(round(len(files)*0.5))
# files = files[0:cutit]

condition = os.path.split(imdir)[1]
positions = [re.findall("\[([^[\]]*)\]", f) for f in files]
plist = list(itertools.chain.from_iterable(positions))
positions = [map(float, i.split(',')) for i in plist]

find_lefts = list(itertools.chain.from_iterable(np.where(np.array([p[0] for p in positions]) == 1.0)))
find_rights = list(itertools.chain.from_iterable(np.where(np.array([p[1] for p in positions]) == 1.0)))

# pop open one image to check size and data type
tiff = TIFF.open(os.path.join(imdir, files[0]), mode='r')
sample = tiff.read_image().astype('float')
sample = block_reduce(sample, reduce_factor, func=np.mean)
tiff.close()
print sample.dtype

if crop_fov:
	sample = sample[strtX:endX, strtY:endY]


# READ IN THE FRAMES:
stack = np.empty((sample.shape[0], sample.shape[1], len(files)))
print len(files)

print('copying files')

for i, f in enumerate(files):

	if i % 100 == 0:
		print('%d images processed...' % i)
	# print f
	# im = imread(os.path.join(imdir, f)).astype('float')
	
	tiff = TIFF.open(os.path.join(imdir, f), mode='r')
	im = tiff.read_image().astype('float')
	tiff.close()

	im = block_reduce(im, reduce_factor, func=np.mean)
	stack[:,:,i] = im #im_reduced

L = stack[:,:,find_lefts]
R = stack[:,:,find_rights]
del stack

avgL = np.mean(L, axis=2)
avgR = np.mean(R, axis=2)

imdiff = avgL - avgR

plt.subplot(1,3,1)
plt.imshow(avgL, cmap = plt.get_cmap('gray'))
plt.subplot(1,3,2)
plt.imshow(avgR, cmap = plt.get_cmap('gray'))
plt.subplot(1,3,3)
plt.imshow(imdiff, cmap = plt.get_cmap('gray'))
plt.colorbar()

fname = '%s/flashbar.png' % os.path.split(imdir)[0]
plt.savefig(fname)

plt.show()
