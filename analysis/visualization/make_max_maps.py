#!/usr/bin/env python2

'''

This script analyzes data acquired using movingBar_tmp.py.

It is the longer way of doing FFT (akin to fiedmap_demodulate_orig.py)

It creates maps based on reversal directions of vertical and horizontal bars.

Run:  python make+maps.py /path/to/imaging/directory

It will output change in response to...

'''

import numpy as np
import os
from skimage.measure import block_reduce
from scipy.misc import imread
import cPickle as pkl
import scipy.signal
import numpy.fft as fft
import sys
import optparse
from libtiff import TIFF
from PIL import Image
import re
import itertools
from scipy import ndimage

import math
import matplotlib as mpl
import matplotlib.pylab as plt
import matplotlib.cm as cm
import pandas as pd


def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'valid')


parser = optparse.OptionParser()
parser.add_option('--headless', action="store_true", dest="headless", default=False, help="run in headless mode, no figs")
parser.add_option('--reduce', action="store", dest="reduce_val", default="4", help="block_reduce value")
parser.add_option('--sigma', action="store", dest="gauss_kernel", default="0", help="size of Gaussian kernel for smoothing")
parser.add_option('--format', action="store", dest="im_format", default="png", help="saved image format")

(options, args) = parser.parse_args()

im_format = '.'+options.im_format
headless = options.headless
reduce_factor = (int(options.reduce_val), int(options.reduce_val))
if reduce_factor[0] > 0:
	reduceit=1
else:
	reduceit=0
gsigma = int(options.gauss_kernel)
if headless:
	mpl.use('Agg')
	

#################################################################################
# GET PATH INFO:
#################################################################################
outdir = sys.argv[1]

rundir = os.path.split(outdir)[0]
sessiondir = os.path.split(rundir)[0]


#################################################################################
# GET BLOOD VESSEL IMAGE:
#################################################################################

folders = os.listdir(sessiondir)
figdir = [f for f in folders if f == 'figures'][0]
ims = os.listdir(os.path.join(sessiondir, figdir))
print ims
impath = os.path.join(sessiondir, figdir, ims[0])
# image = Image.open(impath) #.convert('L')
# imarray = np.asarray(image)
tiff = TIFF.open(impath, mode='r')
im = tiff.read_image().astype('float')
tiff.close()
plt.imshow(im)

# files = os.listdir(outdir)

# # GET BLOOD VESSEL IMAGE:
# ims = [f for f in files if os.path.splitext(f)[1] == str(im_format)]
# print ims
# impath = os.path.join(outdir, ims[0])
# image = Image.open(impath).convert('L')
# imarray = np.asarray(image)

# GET DATA STRUCT FILES:
# sessions = [f for f in flist if os.path.splitext(f)[1] != '.png']
# session_path = os.path.join(outdir, sessions[int(0)]) ## LOOP THIS


#################################################################################
# GET DATA STRUCT FILES:
#################################################################################

files = os.listdir(outdir)
files = [f for f in files if os.path.splitext(f)[1] == '.pkl']

dstructs = [f for f in files if 'D_target' in f and str(reduce_factor) in f]
print dstructs

D = dict()
F = dict()
for cond in dstructs:

	outfile = os.path.join(outdir, cond)
	with open(outfile,'rb') as fp:
		D[cond] = pkl.load(fp)

	# fps = D[cond]['fps']
	# freqs = D[cond]['freqs']
	target_freq = D[cond]['target_freq']
	target_bin = D[cond]['target_bin']

	# #dynrange = D[cond]['dynrange']
	# mean_intensity = D[cond]['mean_intensity']

	# ft = D[cond]['ft']

	# del D[cond]
	print cond

	# 1.  Get maps based on max/min values of the magnitude for each pixels:
	phase_max = np.empty(D[cond]['mean_intensity'].shape)
	phase_min = np.empty(D[cond]['mean_intensity'].shape)
	phase_target = np.empty(D[cond]['mean_intensity'].shape)
	mag_target = np.empty(D[cond]['mean_intensity'].shape)
	for i in range(len(D[cond]['ft'])):

		N = len(D[cond]['ft'].loc[i][2])
		f = D[cond]['ft'].loc[i][2][0:N/2]
		#mag = np.abs(f)
		#phase = np.angle(f)

		#phase_max[ft.loc[i][0], ft.loc[i][1]] = phase[np.where(mag == mag.max())]
		#phase_min[ft.loc[i][0], ft.loc[i][1]] = phase[np.where(mag == mag.min())]

		# phase_target[D[cond]['ft'].loc[i][0], D[cond]['ft'].loc[i][1]] = phase[target_bin]
		# mag_target[D[cond]['ft'].loc[i][0], D[cond]['ft'].loc[i][1]] = mag[target_bin]
		fft[D[cond]['ft'].loc[i][0], D[cond]['ft'].loc[i][1]] = f[target_bin]

	F[cond] = dict()
	#F[cond]['phase_map'] = phase_target
	#F[cond]['mag_map'] = mag_target
	F[cond]['fft_map'] = fft
	#F[cond]['phase_magmin'] = phase_min
	#F[cond]['phase_magmax'] = phase_max
	#F[cond]['target_freq'] = target_freq


V_keys = [k for k in D.keys() if 'V' in k]
H_keys = [k for k in D.keys() if 'H' in k]

if len(V_keys) == 2:
	# HAVE BOTH DIRECTION CONDS:
	azimuth_phase = np.angle(D[V_keys[0]]['ft'] / F[V_keys[1]])
elif len(V_keys) == 1:
	# ONLY HAVE 1 CONDITION:
	azimuth_phase = F[V_keys[0]]['phase_map']
else:
	azimuth_phase = np.zeros(F[F.keys()[0]]['mean_intensity'].shape)


if len(H_keys) == 2:
	# HAVE BOTH DIRECTION CONDS:
	elevation_phase = np.angle(F[H_keys[0]] / F[H_keys[1]])
elif len(V_keys) == 1:
	# ONLY HAVE 1 CONDITION:
	elevation_phase = F[H_keys[0]]['phase_map']
else:
	elevation_phase = np.zeros(F[F.keys()[0]]['mean_intensity'].shape)


# freqs = D[V_keys[0]]['freqs']
# target_freq = D[V_keys[0]]['target_freq']
# target_bin = D[V_keys[0]]['target_bin']


#################################################################################
# PLOT IT:
#################################################################################

plt.subplot(3,4,1) # GREEN LED image
plt.imshow(imarray,cmap=cm.Greys_r)

plt.subplot(3,4,2) # ABS PHASE -- elevation
fig = plt.imshow(elevation_phase, cmap="spectral")
plt.colorbar()
plt.title("elevation")

plt.subplot(3, 4, 3) # ABS PHASE -- azimuth
fig = plt.imshow(azimuth_phase, cmap="spectral")
plt.colorbar()
plt.title("azimuth")


# GET ALL RELATIVE CONDITIONS:

# PHASE:
for i,k in enumerate(H_keys): #enumerate(ftmap.keys()):
	plt.subplot(3,4,i+5)
	# phase_map = np.angle(F[k]) #np.angle(complex(D[k]['ft_real'], D[k]['ft_imag']))
	# #plt.figure()
	fig = plt.imshow(F[k]['phase_map'], cmap=cm.spectral)
	plt.title(k)
	plt.colorbar()

for i,k in enumerate(V_keys): #enumerate(ftmap.keys()):
	plt.subplot(3,4,i+7)
	# phase_map = np.angle(F[k]) #np.angle(complex(D[k]['ft_real'], D[k]['ft_imag']))
	# #plt.figure()
	fig = plt.imshow(F[k]['phase_map'], cmap=cm.spectral)
	plt.title(k)
	plt.colorbar()

# MAG:
for i,k in enumerate(H_keys): #enumerate(D.keys()):
	plt.subplot(3,4,i+9)
	# mag_map = D[k]['mag_map']
	fig = plt.imshow(F[k]['mag_map'], cmap=cm.Greys_r)
	plt.title(k)
	plt.colorbar()

for i,k in enumerate(V_keys): #enumerate(D.keys()):
	plt.subplot(3,4,i+11)
	# mag_map = D[k]['mag_map']
	fig = plt.imshow(F[k]['mag_map'], cmap=cm.Greys_r)
	plt.title(k)
	plt.colorbar()

#plt.suptitle(session_path)
sessionpath = os.path.split(outdir)[0]
plt.suptitle(sessionpath)


# SAVE FIG
outdirs = os.path.join(sessionpath, 'figures')
which_sesh = os.path.split(sessionpath)[1]
print outdirs
if not os.path.exists(outdirs):
	os.makedirs(outdirs)
imname = which_sesh  + '_allmaps_' + str(reduce_factor) + '.svg'
plt.savefig(outdirs + '/' + imname, format='svg', dpi=1200)
print outdirs + '/' + imname
plt.show()


