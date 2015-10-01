#!/usr/bin/env python2

'''

This script analyzes data acquired using stimCircle.py.

Run:  python make_masked_circle_map.py /path/to/imaging/directory

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


def  cart2pol(x,y, units='deg'):
    """Convert from cartesian to polar coordinates

    :usage:

        theta, radius = pol2cart(x, y, units='deg')

    units refers to the units (rad or deg) for theta that should be returned
    """
    radius= np.hypot(x,y)
    theta= np.arctan2(y,x)
    if units in ['deg', 'degs']:
        theta=theta*180/np.pi
    return theta, radius


def pol2cart(theta, radius, units='deg'):
    """Convert from polar to cartesian coordinates

    usage::

        x,y = pol2cart(theta, radius, units='deg')

    """
    if units in ['deg', 'degs']:
        theta = theta*np.pi/180.0
    xx = radius*np.cos(theta)
    yy = radius*np.sin(theta)

    return xx,yy



parser = optparse.OptionParser()
parser.add_option('--headless', action="store_true", dest="headless", default=False, help="run in headless mode, no figs")
parser.add_option('--freq', action="store", dest="target_freq", default="0.05", help="stimulation frequency")
parser.add_option('--reduce', action="store", dest="reduce_val", default="4", help="block_reduce value")
parser.add_option('--sigma', action="store", dest="gauss_kernel", default="0", help="size of Gaussian kernel for smoothing")
parser.add_option('--format', action="store", dest="im_format", default="png", help="saved image format")
parser.add_option('--key', action="store", dest="key", default="stimulus", help="stimulus or blank condition")
parser.add_option('--reverse', action="store_true", dest="reverse", default=False, help="subtract reverse condition")
parser.add_option('--backward', action="store_true", dest="backward", default=False, help="CW = backward")
parser.add_option('--thresh', action="store", dest="threshold", default="0.5", help="percent of mag max")

(options, args) = parser.parse_args()

threshold = float(options.threshold)
backward = options.backward
reverse = options.reverse
key = options.key
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
	

# GET PATH INFO:
outdir = sys.argv[1]
# files = os.listdir(outdir)
# files = [f for f in files if os.path.splitext(f)[1] == '.pkl']

rundir = os.path.split(outdir)[0]
sessiondir = os.path.split(rundir)[0]


#################################################################################
# GET BLOOD VESSEL IMAGE:
#################################################################################
folders = os.listdir(sessiondir)
figpath = [f for f in folders if f == 'figures']


if figpath:
    figdir = figpath[0]
    ims = os.listdir(os.path.join(sessiondir, figdir))
    print ims
    impath = os.path.join(sessiondir, figdir, ims[0])
    # image = Image.open(impath) #.convert('L')
    # imarray = np.asarray(image)
    print os.path.splitext(impath)[1]
    if os.path.splitext(impath)[1] == '.tif':
        tiff = TIFF.open(impath, mode='r')
        imarray = tiff.read_image().astype('float')
        tiff.close()
        plt.imshow(imarray)
    else:
        image = Image.open(impath) #.convert('L')
        imarray = np.asarray(image)


else: # NO BLOOD VESSEL IMAGE...
    imarray = np.zeros([200,300])

if reduceit:
    imarray = block_reduce(imarray, reduce_factor, func=np.mean)


# ims = os.listdir(os.path.join(sessiondir, figdir))
# print ims
# impath = os.path.join(sessiondir, figdir, ims[0])
# # image = Image.open(impath) #.convert('L')
# # imarray = np.asarray(image)
# tiff = TIFF.open(impath, mode='r')
# imarray = tiff.read_image().astype('float')
# tiff.close()
# print imarray.shape

# if reduceit:
#     imarray = block_reduce(imarray, reduce_factor, func=np.mean)

# # plt.imshow(imarray)


#################################################################################
# GET DATA STRUCT FILES:
#################################################################################
# sessions = [f for f in flist if os.path.splitext(f)[1] != '.png']
# session_path = os.path.join(outdir, sessions[int(0)]) ## LOOP THIS

files = os.listdir(outdir)
files = [f for f in files if os.path.splitext(f)[1] == '.pkl']
dstructs = [f for f in files if 'D_target' in f and str(reduce_factor) and key in f]
#dstructs = [f for f in files if 'DF_' in f and str(reduce_factor) in f]

print dstructs

D = dict()
for f in dstructs:
	outfile = os.path.join(outdir, f)
	with open(outfile,'rb') as fp:
		D[f] = pkl.load(fp)

print D.keys()
if reverse:
    Dkey = [k for k in D.keys() if 'reverse' not in k][0]
    DkeyR = [k for k in D.keys() if 'reverse' in k][0]
else:
    Dkey = [k for k in D.keys() if key in k][0]
    if len(D.keys()) > 1:
        Dkey = [k for k in D.keys() if 'reverse' not in k][0]
    print Dkey

# GET CONDITION KEY:
ftmap = dict()
outshape = D[Dkey]['ft_real'].shape

reals = D[Dkey]['ft_real'].ravel()
imags = D[Dkey]['ft_imag'].ravel()
ftmap[Dkey] = [complex(x[0], x[1]) for x in zip(reals, imags)]
ftmap[Dkey] = np.reshape(np.array(ftmap[Dkey]), outshape)

# REVERSE COND:
if reverse:
    reals = D[DkeyR]['ft_real'].ravel()
    imags = D[DkeyR]['ft_imag'].ravel()
    ftmap[DkeyR] = [complex(x[0], x[1]) for x in zip(reals, imags)]
    ftmap[DkeyR] = np.reshape(np.array(ftmap[DkeyR]), outshape)


# GET PHASE MAP:
if reverse:
    phase_map = np.angle(ftmap[Dkey] / ftmap[DkeyR])
    mag_map = (np.abs(ftmap[Dkey]) + np.abs(ftmap[DkeyR])) / 2.
else:
    phase_map = np.angle(ftmap[Dkey])
    mag_map = np.abs(ftmap[Dkey])


# mag_map = 20*np.log10(np.abs(ftmap[Dkey])) # DECIBELS # not normalized...
# mag_map = np.abs(ftmap[Dkey])
# power_map = np.log10(np.abs(ftmap[Dkey])**2)


###########################################
# COMBINE PHASE AND MAG: ??? HSV
###########################################

# # hue = phase_map/360.
# hue = (phase_map - phase_map.min()) / (phase_map.max() - phase_map.min())
# #hue = (phase_map_shift - phase_map_shift.min()) / (phase_map_shift.max() - phase_map_shift.min())
# sat = np.ones(hue.shape)*0.3
# val = (mag_map - mag_map.min()) / (mag_map.max() - mag_map.min())

# HSV = np.ones(val.shape + (3,))
# HSV[...,0] = hue
# HSV[...,2] = sat
# HSV[...,1] = val

# fig = plt.figure()
# fig.add_subplot(1,3,1)
# plt.imshow(imarray,cmap=cm.Greys_r)

# fig.add_subplot(1,3,2)
# plt.imshow(HSV)

# ax = fig.add_subplot(1,3,3, projection='polar')
# ax.set_theta_zero_location('W')
# ax._direction = 2*np.pi

# norm = mpl.colors.Normalize(1*np.pi, -1*np.pi)
# #norm = mpl.colors.Normalize(0, 1)
# #norm = mpl.colors.Normalize(vmin=max(thetas), vmax=max(thetas))
# #quant_steps = 2056
# cb = mpl.colorbar.ColorbarBase(ax, cmap=cm.get_cmap('hsv'),
# 								norm=norm, orientation='horizontal')
# # cb.outline.set_visible(False)
# # ax.set_axis_off()
# ax.set_rlim([-1, 1])
# plt.show() # replace with plt.savefig to save

# #plt.suptitle(session_path)
# sessionpath = os.path.split(outdir)[0]
# plt.suptitle(sessionpath)



###########################################
# PLOT IT ALL:
###########################################


# threshold = 0.5
fig = plt.figure()

if not mag_map.shape == imarray.shape:
    # Need to re-read image and resize:
    if figpath:

        tiff = TIFF.open(impath, mode='r')
        imarray = tiff.read_image().astype('float')
        tiff.close()
        imarray = scipy.misc.imresize(imarray, mag_map.shape)
    else: # NO BLOOD VESSEL IMAGE...
        imarray = np.zeros(mag_map.shape)


print "MAG: ", mag_map.shape
print "IM: ", imarray.shape

# SURFACE + PHASE-MASKED MAP OVERLAY:
fig.add_subplot(1,2,1)
plt.imshow(imarray,cmap=cm.Greys_r)

[x, y] = np.where(mag_map >= threshold*mag_map.max())
print mag_map.min(), threshold*mag_map.max()
phase_mask = np.ones(mag_map.shape) * 100
phase_mask[x, y] = phase_map[x, y]

[nullx, nully] = np.where(phase_mask == 100)
print nullx, nully
phase_mask[nullx, nully] = np.nan
phase_mask = np.ma.array(phase_mask)
plt.imshow(phase_mask, cmap='spectral', vmin=-1*math.pi, vmax=math.pi)

tit = 'Threshold, %.2f of MAG max' % (threshold)
plt.title(tit)

# POLAR / CIRCULAR COLORMAP:
ax = fig.add_subplot(1,2,2, projection='polar')
ax.set_theta_zero_location('W') # 0 on RIGHT side...
if backward:
    ax._direction = 2*np.pi # object moves toward bottom first (CW)
else:
    ax._direction = -2*np.pi # objecct moves toward top first (CCW)

norm = mpl.colors.Normalize(vmax=1*np.pi, vmin=-1*np.pi)
#norm = mpl.colors.Normalize(vmax=2*np.pi, vmin=0)
cb = mpl.colorbar.ColorbarBase(ax, cmap=cm.get_cmap('spectral'),
                                norm=norm, orientation='horizontal')
cb.ax.invert_xaxis()
# cb.outline.set_visible(False)
# ax.set_axis_off()
ax.set_rlim([-1, 1])


################################
# SAVE FIG:
################################

sessionpath = os.path.split(outdir)[0]

outdirs = os.path.join(sessionpath, 'figures')
which_sesh = os.path.split(sessionpath)[1]
plt.suptitle(which_sesh)

print outdirs
if not os.path.exists(outdirs):
    os.makedirs(outdirs)

if reverse:
    imname = which_sesh  + '_MAGmasked_REV_' + str(reduce_factor) + '_' +  key + '_' + 'thresh' + str(threshold) + '.png'
else:
    imname = which_sesh  + '_MAGmasked_' + str(reduce_factor) + '_' +  key + '_' + 'thresh' + str(threshold) + '.png'

fig.savefig(outdirs + '/' + imname)
print outdirs + '/' + imname


# Save as SVG format
if reverse:
    imname = which_sesh  + '_MAGmasked_REV_' + str(reduce_factor) + '_' +  key + '_' + 'thresh' + str(threshold) + '.png'
else:
    imname = which_sesh  + '_MAGmasked_' + str(reduce_factor) + '_' +  key + '_' + 'thresh' + str(threshold) + '.png'

plt.savefig(outdirs + '/' + imname, format='svg', dpi=1200)
print outdirs + '/' + imname

# plt.show()


#################################
# MASK
#################################

# threshold = 0.5
fig = plt.figure()


fig.add_subplot(2,2,1)
plt.imshow(imarray,cmap=cm.Greys_r)


fig.add_subplot(2,2,2) # heat map of mean intensity
plt.imshow(D[Dkey]['mean_intensity'], cmap="hot")
plt.colorbar()
plt.title("mean intensity")


fig.add_subplot(2,2,3)
mean_intensity = D[Dkey]['mean_intensity']
[x, y] = np.where(mean_intensity >= threshold*mean_intensity.max())
phase_mask = np.ones(mean_intensity.shape) * 100
phase_mask[x, y] = phase_map[x, y]

[nullx, nully] = np.where(phase_mask == 100)
print nullx, nully
phase_mask[nullx, nully] = np.nan
phase_mask = np.ma.array(phase_mask)
plt.imshow(phase_mask, cmap='spectral', vmin=-1*math.pi, vmax=math.pi)

tit = 'Threshold, %.2f of mean intensity max' % (threshold)
plt.title(tit)
# plt.colorbar()

ax = fig.add_subplot(2,2,4, projection='polar')
ax.set_theta_zero_location('W') # 0 on RIGHT side...
if backward:
    ax._direction = 2*np.pi # object moves toward bottom first (CW)
else:
    ax._direction = -2*np.pi # objecct moves toward top first (CCW)


norm = mpl.colors.Normalize(vmax=1*np.pi, vmin=-1*np.pi)
#norm = mpl.colors.Normalize(vmax=2*np.pi, vmin=0)

cb = mpl.colorbar.ColorbarBase(ax, cmap=cm.get_cmap('spectral'),
                                norm=norm, orientation='horizontal')
# cb.outline.set_visible(False)
# ax.set_axis_off()
ax.set_rlim([-1, 1])

plt.suptitle(which_sesh)

#####################
# SAVE FIG
#####################

if reverse:
    imname = which_sesh  + '_intensitymasked_REV_' + str(reduce_factor) + '_' +  key + '_' + 'thresh' + str(threshold) + '.png'
else:
    imname = which_sesh  + '_intensitymasked_' + str(reduce_factor) + '_' +  key + '_' + 'thresh' + str(threshold) + '.png'

fig.savefig(outdirs + '/' + imname)
print outdirs + '/' + imname


# Save as SVG format
if reverse:
    imname = which_sesh  + '_intensitymasked_REV_' + str(reduce_factor) + '_' +  key + '_' + 'thresh' + str(threshold) + '.svg'
else:
    imname = which_sesh  + '_intensitymasked_' + str(reduce_factor) + '_' +  key + '_' + 'thresh' + str(threshold) + '.svg'

plt.savefig(outdirs + '/' + imname, format='svg', dpi=1200)
print outdirs + '/' + imname

plt.show()
