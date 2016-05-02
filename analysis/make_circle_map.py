#!/usr/bin/env python2

'''

This script analyzes data acquired using stimCircle.py.

Run:  python get_circle_map.py /path/to/imaging/directory

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
parser.add_option('--reduce', action="store", dest="reduce_val", default="2", help="block_reduce value")
parser.add_option('--sigma', action="store", dest="gauss_kernel", default="0", help="size of Gaussian kernel for smoothing")
parser.add_option('--format', action="store", dest="im_format", default="png", help="saved image format")
parser.add_option('--key', action="store", dest="key", default="stimulus", help="stimulus or blank condition")
parser.add_option('--rev', action="store_true", dest="rev", default="False", help="CCW is standard, CW is reverse")


(options, args) = parser.parse_args()

rev = options.rev
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
files = os.listdir(outdir)
files = [f for f in files if os.path.splitext(f)[1] == '.pkl']

rundir = os.path.split(outdir)[0]
sessiondir = os.path.split(rundir)[0]


#################################################################################
# GET BLOOD VESSEL IMAGE:
#################################################################################
folders = os.listdir(sessiondir)
figdir = [f for f in folders if f == 'figures'][0]
# ims = os.listdir(os.path.join(sessiondir, figdir))
tmp_ims = os.listdir(os.path.join(sessiondir, figdir))
ims = [i for i in tmp_ims if 'surface' in i or 'green' in i]
print ims
impath = os.path.join(sessiondir, figdir, ims[0])
# image = Image.open(impath) #.convert('L')
# imarray = np.asarray(image)
tiff = TIFF.open(impath, mode='r')
imarray = tiff.read_image().astype('float')
tiff.close()
plt.imshow(imarray)


#################################################################################
# GET DATA STRUCT FILES:
#################################################################################
# sessions = [f for f in flist if os.path.splitext(f)[1] != '.png']
# session_path = os.path.join(outdir, sessions[int(0)]) ## LOOP THIS

#files = os.listdir(outdir)
files = [f for f in files if os.path.splitext(f)[1] == '.pkl']
print files
dstructs = [f for f in files if 'D_target' in f and str(reduce_factor) in f]
#dstructs = [f for f in files if 'DF_' in f and str(reduce_factor) in f]

print dstructs

D = dict()
for f in dstructs:
	outfile = os.path.join(outdir, f)
	with open(outfile,'rb') as fp:
		D[f] = pkl.load(fp)

print D.keys()
Dkey = [k for k in D.keys() if key in k][0]
# MATCH ELEV vs. AZIM conditions:
ftmap = dict()
# ftmap_shift = dict()

outshape = D[Dkey]['ft_real'].shape
# for curr_key in D.keys():
# 	reals = D[curr_key]['ft_real'].ravel()
# 	imags = D[curr_key]['ft_imag'].ravel()
# 	ftmap[curr_key] = [complex(x[0], x[1]) for x in zip(reals, imags)]
# 	ftmap[curr_key] = np.reshape(np.array(ftmap[curr_key]), outshape)

# 	reals = D[curr_key]['ft_real_shift'].ravel()
# 	imags = D[curr_key]['ft_imag_shift'].ravel()
# 	ftmap_shift[curr_key] = [complex(x[0], x[1]) for x in zip(reals, imags)]
# 	ftmap_shift[curr_key] = np.reshape(np.array(ftmap_shift[curr_key]), outshape)

reals = D[Dkey]['ft_real'].ravel()
imags = D[Dkey]['ft_imag'].ravel()
ftmap[Dkey] = [complex(x[0], x[1]) for x in zip(reals, imags)]
ftmap[Dkey] = np.reshape(np.array(ftmap[Dkey]), outshape)

# reals = D[Dkey]['ft_real_shift'].ravel()
# imags = D[Dkey]['ft_imag_shift'].ravel()
# ftmap_shift[Dkey] = [complex(x[0], x[1]) for x in zip(reals, imags)]
# ftmap_shift[Dkey] = np.reshape(np.array(ftmap_shift[Dkey]), outshape)


# V_keys = [k for k in ftmap.keys() if 'V' in k]
# H_keys = [k for k in ftmap.keys() if 'H' in k]

#print ftmap.keys()
phase_map = np.angle(ftmap[Dkey])

# phase_map_shift = np.angle(ftmap_shift[Dkey], deg=True)

# phase_shifted = [(idx, i+360) for (idx,i) in enumerate(phase_map.ravel()) if i<0.0]
# for i in phase_shifted:
# 	phase_map.ravel()[i[0]] = i[1]

# ###########################################
# # PLOT SHIAT:
# ###########################################

# #----------------------------------------
# # DEFAULT FROM FFT:
# #----------------------------------------

# fig = plt.figure()
# fig.add_subplot(2,2,1)
# plt.imshow(phase_map_shift, cmap='spectral')
# #plt.colorbar()

# ax = fig.add_subplot(2,2,2, projection='polar')
# ax.set_theta_zero_location('E')
# ax._direction = 2*np.pi

# #norm = mpl.colors.Normalize(1*np.pi, -1*np.pi)
# norm = mpl.colors.Normalize(0, 2*np.pi)
# #norm = mpl.colors.Normalize(vmin=-180, vmax = 180)
# # norm = mpl.colors.Normalize(vmin=0, vmax=360)
# #quant_steps = 2056
# cb = mpl.colorbar.ColorbarBase(ax, cmap=cm.get_cmap('spectral'),
# 								norm=norm, orientation='horizontal')
# # cb.outline.set_visible(False)
# # ax.set_axis_off()
# ax.set_rlim([-1, 1])


# #----------------------------------------
# # mapping from 0 to 360 degrees:
# #----------------------------------------

# fig.add_subplot(2,2,3)
# plt.imshow(phase_map, cmap='spectral')
# #plt.colorbar()

# ax = fig.add_subplot(2,2,4, projection='polar')
# ax.set_theta_zero_location('E')
# ax._direction = 2*np.pi

# #norm = mpl.colors.Normalize(1*np.pi, -1*np.pi)
# # norm = mpl.colors.Normalize(vmin=-180, vmax = 180)
# norm = mpl.colors.Normalize(vmin=0, vmax=360)
# #quant_steps = 2056
# cb = mpl.colorbar.ColorbarBase(ax, cmap=cm.get_cmap('spectral'),
# 								norm=norm, orientation='horizontal')
# # cb.outline.set_visible(False)
# # ax.set_axis_off()
# ax.set_rlim([-1, 1])



# # fig = plt.figure()
# # fig.add_subplot(2,2,1)
# # plt.imshow(phase_map, cmap='spectral')
# # plt.colorbar()

# # fig.add_subplot(2,2,2)
# # plt.imshow(phase_map_shift, cmap='spectral')
# # plt.colorbar()

# ###########################################


# mag_map = 20*np.log10(np.abs(ftmap[Dkey])) # DECIBELS # not normalized...
mag_map = np.abs(ftmap[Dkey])
power_map = np.log10(np.abs(ftmap[Dkey])**2)


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

fig = plt.figure()

fig.add_subplot(2,3,1) # GREEN LED image
plt.imshow(imarray,cmap=cm.Greys_r)

fig.add_subplot(2,3,2) # heat map of mean intensity
plt.imshow(D[Dkey]['mean_intensity'], cmap="hot")
plt.colorbar()
plt.title("mean intensity")


fig.add_subplot(2,3,3) # dyn range??
plt.imshow(D[Dkey]['dynrange'])
plt.colorbar()
plt.title("dynamic range (log2)")

fig.add_subplot(2,3,4)
#mag_map = D[Dkey]['mag_map']
plt.imshow(mag_map, cmap=cm.Greys_r)
plt.title("magnitude (dB)")
plt.colorbar()


fig.add_subplot(2,3,5) # ABS PHASE -- azimuth
plt.imshow(phase_map, cmap="spectral", vmin=-1*math.pi, vmax=math.pi)
#plt.colorbar()
plt.title("phase")

# new subplot for radial color map:

# # FIND CYCLE STARTS:
# imdir = os.path.join(os.path.split(outdir)[0], 'stimulus')
# files = os.listdir(imdir)
# files = sorted([f for f in files if os.path.splitext(f)[1] == str(im_format)])
# print len(files)

# positions = [re.findall("\[([^[\]]*)\]", f) for f in files]
# plist = list(itertools.chain.from_iterable(positions))
# positions = [map(float, i.split()) for i in plist] #[map(float, i.split(' ')) for i in plist]

# thetas = []
# radii = []
# for p in positions:
# 	theta, R = cart2pol(p[0], p[1])
# 	thetas.append(theta)
# 	radius = R

# tmp_thetas = np.unwrap(thetas)
# thetas = [i % 360. for i in tmp_thetas]
# del tmp_thetas
# norm_thetas = thetas/360.


ax = fig.add_subplot(2,3,6, projection='polar')
ax.set_theta_zero_location('W') # W puts 0 on RIGHT side...
if rev:
    ax._direction = 2*np.pi # object moves toward bottom first (CW)
else:
    ax._direction = -2*np.pi # objecct moves toward top first (CCW)

norm = mpl.colors.Normalize(vmax=1*np.pi, vmin=-1*np.pi)
#norm = mpl.colors.Normalize(vmax=2*np.pi, vmin=0)

#norm = mpl.colors.Normalize(vmin=max(thetas), vmax=max(thetas))
#quant_steps = 2056
cb = mpl.colorbar.ColorbarBase(ax, cmap=cm.get_cmap('spectral'),
								norm=norm, orientation='horizontal')
# cb.outline.set_visible(False)
# ax.set_axis_off()
ax.set_rlim([-1, 1])

#plt.show() # replace with plt.savefig to save

#####################
# SAVE FIG
#####################

sessionpath = os.path.split(outdir)[0]
plt.suptitle(sessionpath)

outdirs = os.path.join(sessionpath, 'figures')
which_sesh = os.path.split(sessionpath)[1]
print outdirs
if not os.path.exists(outdirs):
    os.makedirs(outdirs)

imname = which_sesh  + '_allmaps_' + str(reduce_factor) + '_' +  key + '.png'
fig.savefig(outdirs + '/' + imname)
print outdirs + '/' + imname


# Save as SVG format
imname = which_sesh  + '_allmaps_' + str(reduce_factor) + '_' +  key + '.svg'
plt.savefig(outdirs + '/' + imname, format='svg', dpi=1200)
print outdirs + '/' + imname

plt.show()



#################################
# Make a new figure, with HSV coloramp (continuous)
#################################

# fig = plt.figure()
# fig.add_subplot(2,2,1) # GREEN LED image
# plt.imshow(imarray,cmap=cm.Greys_r)

# fig.add_subplot(2,2,2)
# plt.imshow(mag_map, cmap=cm.Greys_r)
# plt.title("magnitude")
# plt.colorbar()

# fig.add_subplot(2,2,3) # ABS PHASE -- azimuth
# plt.imshow(phase_map, cmap="hsv", vmin=-1*math.pi, vmax=math.pi)
# #plt.colorbar()
# plt.title("phase")

# ax = fig.add_subplot(2,2,4, projection='polar')
# ax.set_theta_zero_location('W') # 0 on RIGHT side...
# ax._direction = 2*np.pi

# norm = mpl.colors.Normalize(1*np.pi, -1*np.pi)
# #quant_steps = 2056
# cb = mpl.colorbar.ColorbarBase(ax, cmap=cm.get_cmap('hsv'),
# 								norm=norm, orientation='horizontal')
# # cb.outline.set_visible(False)
# # ax.set_axis_off()
# ax.set_rlim([-1, 1])


# imname = which_sesh  + '_allmaps_HSV' + str(reduce_factor) + '_' + key + '.png'
# fig.savefig(outdirs + '/' + imname)
# print outdirs + '/' + imname

# plt.show()



# #################################
# # MASK
# #################################

# threshold = 0.8
# fig = plt.figure()


# fig.add_subplot(2,2,1)
# plt.imshow(imarray,cmap=cm.Greys_r)


# fig.add_subplot(2,2,2) # heat map of mean intensity
# plt.imshow(D[Dkey]['mean_intensity'], cmap="hot")
# plt.colorbar()
# plt.title("mean intensity")


# fig.add_subplot(2,2,3)
# # mean_intensity = D[Dkey]['mean_intensity']
# [x, y] = np.where(mag_map >= threshold*mag_map.max())
# print mag_map.min(), threshold*mag_map.max()
# phase_mask = np.ones(mag_map.shape) * 100
# phase_mask[x, y] = phase_map[x, y]

# [nullx, nully] = np.where(phase_mask == 100)
# print nullx, nully
# phase_mask[nullx, nully] = np.nan
# phase_mask = np.ma.array(phase_mask)

# # palette = cm.spectral
# # maxval = 1*np.pi
# # minval = -1*np.pi
# # palette.set_over('k', maxval)
# # palette.set_under('k', minval)
# # # palette.set_bad('k', 1.0)

# # plt.imshow(phase_mask, cmap=palette, norm=mpl.colors.Normalize(vmin=minval, vmax=maxval, clip=False))
# # # cs.cmap.set_under(color='k')
# plt.imshow(phase_mask, cmap='spectral')

# tit = 'Threshold, %.2f of mag max' % (threshold)
# plt.title(tit)
# # plt.colorbar()


# ax = fig.add_subplot(2,2,4, projection='polar')
# ax.set_theta_zero_location('E') # 0 on RIGHT side...
# ax._direction = 2*np.pi

# norm = mpl.colors.Normalize(vmax=1*np.pi, vmin=-1*np.pi)
# #norm = mpl.colors.Normalize(vmax=2*np.pi, vmin=0)

# cb = mpl.colorbar.ColorbarBase(ax, cmap=cm.get_cmap('spectral'),
#                                 norm=norm, orientation='horizontal')
# # cb.outline.set_visible(False)
# # ax.set_axis_off()
# ax.set_rlim([-1, 1])


# # SAVE:
# imname = which_sesh  + '_magmask' + str(reduce_factor) + '_' + key + '.png'
# fig.savefig(outdirs + '/' + imname)
# print outdirs + '/' + imname



# plt.show()
