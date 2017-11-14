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
from skimage.measure import block_reduce


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

imdir = sys.argv[1]
imtype = str(sys.argv[2])

outdir = os.path.join(os.path.split(imdir)[0], 'figures')
if not os.path.exists(outdir):
	os.makedirs(outdir)


# if itype=='gcamp':
# 	cond_start = 0
# 	cond_endend = 3 # number of seconds for stimulus ON
# 	blank_start = 9
# 	blank_end = 19
# else:
# 	cond_start = 0
# 	cond_endend = 3 # number of seconds for stimulus ON
# 	blank_start = 9
# 	blank_end = 19


reduce_factor = (2,2)
reduceit = 1
fps = 60.0

if imtype=='auto':
	stim_start = 0
	stim_end = 3 # number of seconds for stimulus ON
	blank_start = 9
	blank_end = 19
else:
	blank_start = -2
	blank_end = 0
	stim_start = 0
	stim_end = 5

conditions = os.listdir(imdir)
conditions = [c for c in conditions if not 'tif' in c]
conditions = [c for c in conditions if not 'blank' in c]
print conditions

D = dict()
for cond in conditions:
	print cond
	D[cond] = dict()

	trials = sorted(os.listdir(os.path.join(imdir, cond)), key=natural_keys)

	for t in trials:
		print t
		files = os.listdir(os.path.join(imdir, cond, t))
		files = sorted([f for f in files if os.path.splitext(f)[1] == '.tif'])

		tiff = TIFF.open(os.path.join(imdir, cond, t, files[0]), mode='r')
		sample = tiff.read_image().astype('float')
		if reduceit:
			sample = block_reduce(sample, reduce_factor, func=np.mean)

		tiff.close()
		print sample.dtype

		# READ IN THE FRAMES:
		stack = np.empty((sample.shape[0], sample.shape[1], len(files)))
		print('copying files')

		for i,fn in enumerate(files):

			if i % 100 == 0:
				print('%d images processed...' % i)

			tiff = TIFF.open(os.path.join(imdir, cond, t, f), mode='r')
			im = tiff.read_image().astype('float')
			if reduceit:
				im = block_reduce(im, reduce_factor, func=np.mean)

			tiff.close()

			stack[:,:,i] = im

		D[cond][t] = stack

N = dict()
for k in D.keys():
	curr_cond = D[k]

	normD = np.empty((sample.shape[0], sample.shape[1], len(curr_cond.keys())-1))
	for i in range(1, len(curr_cond.keys())):
		curr_trial = curr_cond[sorted(curr_cond.keys(), key=natural_keys)[i]]
		prev_trial = curr_cond[sorted(curr_cond.keys(), key=natural_keys)[i-1]]

		stim_frames = np.mean(curr_trial[:, :, stim_start:stim_end*fps], axis=2)
		blank_frames = np.mean(prev_trial[:,:,blank_start*fps:], axis=2)

		norm_frame = (stim_frames - blank_frames) / blank_frames

		normD[:, :, i-1] = norm_frame

	N[k] = np.mean(normD, axis=2)

del D

leftmap = N['gab-left']
rightmap = N['gab-right']

plt.figure()
plt.subplot(1,3,1)
plt.imshow(leftmap, cmap='gray')
plt.colorbar()
plt.title('left-center')

#plt.figure()
plt.subplot(1,3,2)
plt.imshow(rightmap, cmap='gray')
plt.colorbar()
plt.title('right')

#plt.figure()
plt.subplot(1,3,3)
plt.imshow(leftmap-rightmap, cmap='gray')
plt.colorbar()
plt.title('difference')


plt.suptitle(os.path.split(imdir)[1])

plt.suptitle(imdir)
fname = '%s/driftGabor_%s.png' % (outdir, os.path.split(imdir)[1])
print fname
plt.savefig(fname)

plt.show()



# 	# ST = []
# 	# for iterx, idx in enumerate(strtidxs[0:-1]):
# 	# 	tmp_files = files[idx:strtidxs[iterx+1]]

# 	# 	stack = np.empty((sample.shape[0], sample.shape[1], len(tmp_files)))
# 	# 	#print tmp_files[0], tmp_files[-1]

# 	# 	print('copying files')

# 	# 	for i, f in enumerate(tmp_files):

# 	# 		if i % 100 == 0:
# 	# 			print('%d images processed...' % i)

# 	# 		#im = imread(os.path.join(imdir, cond, f)).astype('float')
			
# 	# 		tiff = TIFF.open(os.path.join(imdir, cond, f), mode='r')
# 	# 		im = tiff.read_image().astype('float')
# 	# 		tiff.close()

# 	# 		if reduceit:
# 	# 			im = block_reduce(im, reduce_factor, func=np.mean)
			
# 	# 		stack[:,:,i] = im

# 	# 	ST.append(np.mean(stack, axis=2))

# 	#ST = np.asarray(ST)
# 	#print ST.shape
# 	print stack.shape
# 	#D[cond] = np.mean(ST, axis=0)
# 	D[cond] = np.mean(stack[:, :, stim_start:stim_end], axis=2)
# 	D[cond+'_blank'] = np.mean(stack[:, :, blank_strt:blank_end], axis=2)

# 	print D[cond].shape

# # blank = D['blank']
# # del D['blank']
# # print blank.shape

# # for dkey in D.keys():
# # 	blank = 
# # 	D[dkey+'_norm'] = (blank - D[dkey]) / blank

# blanks_keys = [k for k in D.keys() if '_blank' in k]
# conds_keys = [k[:-6] for k in blanks_keys]
# print conds_keys
# print blanks_keys

# for cond_key in conds_keys:
# 	blank_key = [k for k in blanks_keys if cond_key in k][0]
# 	blank = D[blank_key]
# 	print "blank", blank_key, blank.shape
# 	print "cond", cond_key, D[cond_key].shape
# 	D[cond_key+'_norm'] = (blank - D[cond_key]) / blank

# normkeys = [k for k in D.keys() if 'norm' in k]
# print normkeys
# for n in normkeys:
# 	print D[k]

# imdiff = D['gab-left'] - D['gab-right']

# plt.subplot(2,3,1)
# plt.imshow(D['gab-left'], cmap = plt.get_cmap('gray'))
# plt.title('gab-left')
# plt.colorbar()

# plt.subplot(2,3,2)
# plt.imshow(D['gab-right'], cmap = plt.get_cmap('gray'))
# plt.title('gab-right')
# plt.colorbar()

# plt.subplot(2,3,4)
# plt.imshow(D['gab-left_norm'], cmap = plt.get_cmap('gray'))
# plt.title('gab-left_norm')
# plt.colorbar()

# plt.subplot(2,3,5)
# plt.imshow(D['gab-right_norm'], cmap = plt.get_cmap('gray'))
# plt.title('gab-right_norm')
# plt.colorbar()

# plt.subplot(2,3,6)
# plt.imshow(D['gab-left_norm'] - D['gab-right_norm'], cmap = plt.get_cmap('gray'))
# plt.title('norm diff')
# plt.colorbar()

# plt.subplot(2,3,3)
# plt.imshow(imdiff, cmap = plt.get_cmap('gray'))
# plt.title('plain diff')
# plt.colorbar()

# plt.suptitle(os.path.split(imdir)[1])

# plt.suptitle(imdir)
# fname = '%s/driftGabor_%s.png' % (outdir, os.path.split(imdir)[1])
# print fname
# plt.savefig(fname)

# plt.show()
