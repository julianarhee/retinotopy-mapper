#!/usr/bin/env python2

'''

This script analyzes data acquired using driftGabor.py.

Run:  python plotAZ.py /path/to/imaging/directory

It will output change in response to upward drifting gabors at 2 diff locs (normalized to baseline).

'''

import numpy as np
import os
#from skimage.measure import block_reduce
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

imdir = sys.argv[1]
outdir = os.path.join(os.path.split(os.path.split(imdir)[0])[0], 'figures')
if not os.path.exists(outdir):
	os.makedirs(outdir)

reduce_factor = (2,2)
reduceit = 0

crop_fov = 0
if len(sys.argv) > 2:
	cropped = sys.argv[2].split(',')
	[strtX, endX, strtY, endY] = [int(i) for i in cropped]
	crop_fov = 1

conditions = os.listdir(imdir)
conditions = [c for c in conditions if not 'tif' in c]
print conditions
D = dict()
for cond in conditions:
	print cond
	files = os.listdir(os.path.join(imdir, cond))
	files = sorted([f for f in files if os.path.splitext(f)[1] == '.tif'])
	strtidxs = [i for i, item in enumerate(files) if re.search('_0_', item)]
	strtidxs.append(len(files))

	tiff = TIFF.open(os.path.join(imdir, cond, files[0]), mode='r')
	sample = tiff.read_image().astype('float')
	if reduceit:
		sample = block_reduce(sample, reduce_factor, func=np.mean)

	tiff.close()
	print sample.dtype

	# READ IN THE FRAMES:
	ST = []
	for iterx, idx in enumerate(strtidxs[0:-1]):
		tmp_files = files[idx:strtidxs[iterx+1]]

		stack = np.empty((sample.shape[0], sample.shape[1], len(tmp_files)))
		#print tmp_files[0], tmp_files[-1]

		print('copying files')

		for i, f in enumerate(tmp_files):

			if i % 100 == 0:
				print('%d images processed...' % i)

			#im = imread(os.path.join(imdir, cond, f)).astype('float')
			
			tiff = TIFF.open(os.path.join(imdir, cond, f), mode='r')
			im = tiff.read_image().astype('float')
			tiff.close()

			if reduceit:
				im = block_reduce(im, reduce_factor, func=np.mean)
			
			stack[:,:,i] = im

		ST.append(np.mean(stack, axis=2))

	ST = np.asarray(ST)
	print ST.shape
	D[cond] = np.mean(ST, axis=0)
	print D[cond].shape

blank = D['blank']
del D['blank']
print blank.shape

for dkey in D.keys():
	D[dkey+'_norm'] = (blank - D[dkey]) / blank


imdiff = D['gab-left'] - D['gab-right']

plt.subplot(2,3,1)
plt.imshow(D['gab-left'], cmap = plt.get_cmap('gray'))
plt.title('gab-left')
plt.colorbar()

plt.subplot(2,3,2)
plt.imshow(D['gab-right'], cmap = plt.get_cmap('gray'))
plt.title('gab-right')
plt.colorbar()

plt.subplot(2,3,4)
plt.imshow(D['gab-left_norm'], cmap = plt.get_cmap('gray'))
plt.title('gab-left_norm')
plt.colorbar()

plt.subplot(2,3,5)
plt.imshow(D['gab-right_norm'], cmap = plt.get_cmap('gray'))
plt.title('gab-right_norm')
plt.colorbar()

plt.subplot(2,3,6)
plt.imshow(D['gab-left_norm'] - D['gab-right_norm'], cmap = plt.get_cmap('gray'))
plt.title('norm diff')
plt.colorbar()

plt.subplot(2,3,3)
plt.imshow(imdiff, cmap = plt.get_cmap('gray'))
plt.title('plain diff')
plt.colorbar()

plt.suptitle(os.path.split(os.path.split(imdir)[0])[1])

fname = '%s/flashbar_%s.png' % (outdir, os.path.split(os.path.split(imdir)[0])[1])
print fname
plt.savefig(fname)

plt.show()
