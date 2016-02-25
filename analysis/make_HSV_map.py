#!/usr/bin/env python2

'''

This script analyzes data acquired using stimCircle.py.
It differs from make_circle_map.py in that it creates a mask
out of the magnitude map...

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
#parser.add_option('--freq', action="store", dest="target_freq", default="0.05", help="stimulation frequency")
parser.add_option('--reduce', action="store", dest="reduce_val", default="4", help="block_reduce value")
parser.add_option('--sigma', action="store", dest="gauss_kernel", default="0", help="size of Gaussian kernel for smoothing")
parser.add_option('--format', action="store", dest="im_format", default="png", help="saved image format")
parser.add_option('--key', action="store", dest="key", default="stimulus", help="stimulus or blank condition")

(options, args) = parser.parse_args()

key = options.key
im_format = '.'+options.im_format
headless = options.headless
if headless:
    mpl.use('Agg')
gsigma = int(options.gauss_kernel)
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


# STORED AS DATAFRAME...
dstructs = [f for f in files if 'D_fft' in f and str(reduce_factor) and key in f]
print "Full FFT struct: ", dstructs

f = dstructs[0]
D = dict()
outfile = os.path.join(outdir, dstructs[0])
with open(outfile,'rb') as fp:
    D[f] = pkl.load(fp)

print "Full FFT keys: ", D.keys()
Dkey = D.keys()[0] #[k for k in D.keys() if key in k][0]


#################################################################################
# GET THE STUFF NEEDED FOR ANALYSIS:
#################################################################################

dd_structs = [f for f in files if 'D_target' in f and str(reduce_factor) and key in f]
print "Target dict with session info: ", dd_structs

f = dd_structs[0]
S = dict()
outfile = os.path.join(outdir, dd_structs[0])
with open(outfile,'rb') as fp:
    S[f] = pkl.load(fp)

print "Session info keys: ", S.keys()

print "Session info saved: ", S[S.keys()[0]].keys()
fps = S[S.keys()[0]]['fps']
freqs = S[S.keys()[0]]['freqs']
target_freq = S[S.keys()[0]]['target_freq']
target_bin = S[S.keys()[0]]['target_bin']

dynrange = S[S.keys()[0]]['dynrange']
mean_intensity = S[S.keys()[0]]['mean_intensity']

ft = D[Dkey]['ft']

del D


#################################################################################
# GET BLOOD VESSEL IMAGE:
#################################################################################
folders = os.listdir(sessiondir)
print folders
figdir = [f for f in folders if f == 'figures'][0]
ims = os.listdir(os.path.join(sessiondir, figdir))
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


################################################################################
# GET MAPS!
#################################################################################

# 1.  Get maps based on max/min values of the magnitude for each pixels:
phase_max = np.empty(mean_intensity.shape)
phase_min = np.empty(mean_intensity.shape)
phase_target = np.empty(mean_intensity.shape)
mag_target = np.empty(mean_intensity.shape)
for i in range(len(ft)):
    
    N = len(ft.loc[i][2])
    f = ft.loc[i][2][0:N/2]
    mag = np.abs(f)
    phase = np.angle(f)

    phase_max[ft.loc[i][0], ft.loc[i][1]] = phase[np.where(mag == mag.max())]
    phase_min[ft.loc[i][0], ft.loc[i][1]] = phase[np.where(mag == mag.min())]

    phase_target[ft.loc[i][0], ft.loc[i][1]] = phase[target_bin]
    mag_target[ft.loc[i][0], ft.loc[i][1]] = mag[target_bin]


maptype = 'hsv'
low = -1*np.pi
high = np.pi

fig, axes = plt.subplots(nrows=2, ncols=2)
axidx = axes.flat
im = axidx[0].imshow(imarray,cmap=cm.Greys_r)

im = axidx[1].imshow(phase_target, vmin=low, vmax=high, cmap=maptype)
axidx[1].set_title('phase at stim frequency')

im = axidx[2].imshow(phase_max, vmin=low, vmax=high, cmap=maptype)
axidx[2].set_title('phase at max mag')

im = axidx[3].imshow(phase_min, vmin=low, vmax=high, cmap=maptype)
axidx[3].set_title('phase at min mag')

# fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
# fig.colorbar(im, cax=cbar_ax)

legend = fig.add_subplot(1,1,1, projection='polar')
legend.set_theta_zero_location('E')
legend._direction = 2*np.pi
norm = mpl.colors.Normalize(0, 2*np.pi)
#norm = mpl.colors.Normalize(0, 1)
#norm = mpl.colors.Normalize(vmin=max(thetas), vmax=max(thetas))
#quant_steps = 2056
cb = mpl.colorbar.ColorbarBase(legend, cmap=cm.get_cmap('hsv'),
                                norm=norm, orientation='horizontal')
# cb.outline.set_visible(False)
# ax.set_axis_off()
legend.set_rlim([-1, 1])
plt.show() # replace with plt.savefig to save
0


# 2.  Combine PHASE and MAG info:
phase_map = phase_target
mag_map = 20*np.log10(mag_target)
# mag_map = 20*np.log10(np.abs(ftmap[Dkey])) # DECIBELS # not normalized...
# power_map = np.log10(np.abs(ftmap[Dkey])**2)

# hue = phase_map/360.
hue = (phase_map - phase_map.min()) / (phase_map.max() - phase_map.min())
#hue = (phase_map_shift - phase_map_shift.min()) / (phase_map_shift.max() - phase_map_shift.min())
sat = np.ones(hue.shape)*0.5
val = (mag_map - mag_map.min()) / (mag_map.max() - mag_map.min())

HSV = np.ones(val.shape + (3,))
HSV[...,0] = hue
HSV[...,2] = sat
HSV[...,1] = val

fig = plt.figure()
fig.add_subplot(2,2,1)
plt.imshow(imarray,cmap=cm.Greys_r)

fig.add_subplot(2,2,2)
plt.imshow(mag_map, cmap=cm.get_cmap('gray'))
plt.colorbar()
plt.title('magnitude (dB)')

fig.add_subplot(2,2,3)
plt.imshow(HSV)
plt.title('combined')

ax = fig.add_subplot(2,2,4, projection='polar')
ax.set_theta_zero_location('E')
ax._direction = 2*np.pi
norm = mpl.colors.Normalize(0, 2*np.pi)
#norm = mpl.colors.Normalize(0, 1)
#norm = mpl.colors.Normalize(vmin=max(thetas), vmax=max(thetas))
#quant_steps = 2056
cb = mpl.colorbar.ColorbarBase(ax, cmap=cm.get_cmap('hsv'),
                                norm=norm, orientation='horizontal')
# cb.outline.set_visible(False)
# ax.set_axis_off()
ax.set_rlim([-1, 1])
plt.show() # replace with plt.savefig to save


#plt.suptitle(session_path)
sessionpath = os.path.split(outdir)[0]
plt.suptitle(sessionpath)


imname = which_sesh  + '_combinedHSV' + str(reduce_factor) + '_' + key + '.png'
plt.savefig(outdirs + '/' + imname)
print outdirs + '/' + imname











ftmap = dict()
outshape = D[Dkey]['ft_real'].shape

reals = D[Dkey]['ft_real'].ravel()
imags = D[Dkey]['ft_imag'].ravel()
ftmap[Dkey] = [complex(x[0], x[1]) for x in zip(reals, imags)]
ftmap[Dkey] = np.reshape(np.array(ftmap[Dkey]), outshape)

reals = D[Dkey]['ft_real_shift'].ravel()
imags = D[Dkey]['ft_imag_shift'].ravel()
ftmap_shift[Dkey] = [complex(x[0], x[1]) for x in zip(reals, imags)]

#print ftmap.keys()
phase_map = np.angle(ftmap[Dkey], deg=True)

phase_map_shift = np.angle(ftmap_shift[Dkey], deg=True)

phase_shifted = [(idx, i+360) for (idx,i) in enumerate(phase_map.ravel()) if i<0.0]
for i in phase_shifted:
    phase_map.ravel()[i[0]] = i[1]

###########################################
# PLOT SHIAT:
###########################################

#----------------------------------------
# DEFAULT FROM FFT:
#----------------------------------------

fig = plt.figure()
fig.add_subplot(2,2,1)
plt.imshow(phase_map_shift, cmap='spectral')
#plt.colorbar()

ax = fig.add_subplot(2,2,2, projection='polar')
ax.set_theta_zero_location('E')
ax._direction = 2*np.pi

#norm = mpl.colors.Normalize(1*np.pi, -1*np.pi)
norm = mpl.colors.Normalize(0, 2*np.pi)
#norm = mpl.colors.Normalize(vmin=-180, vmax = 180)
# norm = mpl.colors.Normalize(vmin=0, vmax=360)
#quant_steps = 2056
cb = mpl.colorbar.ColorbarBase(ax, cmap=cm.get_cmap('spectral'),
                                norm=norm, orientation='horizontal')
# cb.outline.set_visible(False)
# ax.set_axis_off()
ax.set_rlim([-1, 1])


#----------------------------------------
# mapping from 0 to 360 degrees:
#----------------------------------------

fig.add_subplot(2,2,3)
plt.imshow(phase_map, cmap='spectral')
#plt.colorbar()

ax = fig.add_subplot(2,2,4, projection='polar')
ax.set_theta_zero_location('E')
ax._direction = 2*np.pi

#norm = mpl.colors.Normalize(1*np.pi, -1*np.pi)
# norm = mpl.colors.Normalize(vmin=-180, vmax = 180)
norm = mpl.colors.Normalize(vmin=0, vmax=360)
#quant_steps = 2056
cb = mpl.colorbar.ColorbarBase(ax, cmap=cm.get_cmap('spectral'),
                                norm=norm, orientation='horizontal')
# cb.outline.set_visible(False)
# ax.set_axis_off()
ax.set_rlim([-1, 1])



# fig = plt.figure()
# fig.add_subplot(2,2,1)
# plt.imshow(phase_map, cmap='spectral')
# plt.colorbar()

# fig.add_subplot(2,2,2)
# plt.imshow(phase_map_shift, cmap='spectral')
# plt.colorbar()

###########################################


# mag_map = 20*np.log10(np.abs(ftmap[Dkey])) # DECIBELS # not normalized...
# power_map = np.log10(np.abs(ftmap[Dkey])**2)


# ###########################################
# # COMBINE PHASE AND MAG: ??? HSV
# ###########################################

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
#                                 norm=norm, orientation='horizontal')
# # cb.outline.set_visible(False)
# # ax.set_axis_off()
# ax.set_rlim([-1, 1])
# plt.show() # replace with plt.savefig to save


# #plt.suptitle(session_path)
# sessionpath = os.path.split(outdir)[0]
# plt.suptitle(sessionpath)


# imname = which_sesh  + '_combinedHSV' + str(reduce_factor) + '_' + key + '.png'
# plt.savefig(outdirs + '/' + imname)
# print outdirs + '/' + imname






# # PLOT IT ALL:

# fig = plt.figure()

# fig.add_subplot(2,3,1) # GREEN LED image
# plt.imshow(imarray,cmap=cm.Greys_r)

# fig.add_subplot(2,3,2) # heat map of mean intensity
# plt.imshow(D[Dkey]['mean_intensity'], cmap="hot")
# plt.colorbar()
# plt.title("mean intensity")


# fig.add_subplot(2,3,3) # dyn range??
# plt.imshow(D[Dkey]['dynrange'])
# plt.colorbar()
# plt.title("dynamic range")

# fig.add_subplot(2,3,4)
# mag_map = D[Dkey]['mag_map']
# plt.imshow(mag_map, cmap=cm.Greys_r)
# plt.title("magnitude")
# plt.colorbar()


# fig.add_subplot(2,3,5) # ABS PHASE -- azimuth
# plt.imshow(phase_map, cmap="spectral")
# #plt.colorbar()
# plt.title("phase")

# # new subplot for radial color map:
# ax = fig.add_subplot(2,3,6, projection='polar')
# ax.set_theta_zero_location('W')
# ax._direction = 2*np.pi

# norm = mpl.colors.Normalize(1*np.pi, -1*np.pi)
# quant_steps = 2056
# cb = mpl.colorbar.ColorbarBase(ax, cmap=cm.get_cmap('spectral', quant_steps),
# 								norm=norm, orientation='horizontal')
# # cb.outline.set_visible(False)
# # ax.set_axis_off()
# ax.set_rlim([-1, 1])
# plt.show() # replace with plt.savefig to save

# #plt.suptitle(session_path)
# sessionpath = os.path.split(outdir)[0]
# plt.suptitle(sessionpath)



# # # FIND CYCLE STARTS:
# # imdir = os.path.join(os.path.split(outdir)[0], 'stimulus')
# # files = os.listdir(imdir)
# # files = sorted([f for f in files if os.path.splitext(f)[1] == str(im_format)])
# # print len(files)

# # positions = [re.findall("\[([^[\]]*)\]", f) for f in files]
# # plist = list(itertools.chain.from_iterable(positions))
# # positions = [map(float, i.split()) for i in plist] #[map(float, i.split(' ')) for i in plist]

# # thetas = []
# # radii = []
# # for p in positions:
# # 	theta, R = cart2pol(p[0], p[1])
# # 	thetas.append(theta)
# # 	#radii.append(R)

# # radius = R
# # angleradians = [math.radians(d) for d in thetas]
# # del plist, radii



# # SAVE FIG
# outdirs = os.path.join(sessionpath, 'figures')
# which_sesh = os.path.split(sessionpath)[1]
# print outdirs
# if not os.path.exists(outdirs):
# 	os.makedirs(outdirs)

# imname = which_sesh  + '_allmaps_' + str(reduce_factor) + '_' +  key + '.png'
# plt.savefig(outdirs + '/' + imname)
# print outdirs + '/' + imname

# #plt.show()

# # Save as SVG format
# imname = which_sesh  + '_allmaps_' + str(reduce_factor) + '_' +  key + '.svg'
# plt.savefig(outdirs + '/' + imname, format='svg', dpi=1200)
# print outdirs + '/' + imname
# #plt.show()



# # Make a new figure, with HSV coloramp (continuous)

# fig = plt.figure()
# fig.add_subplot(2,2,1) # GREEN LED image
# plt.imshow(imarray,cmap=cm.Greys_r)

# fig.add_subplot(2,2,2)
# mag_map = D[Dkey]['mag_map']
# plt.imshow(mag_map, cmap=cm.Greys_r)
# plt.title("magnitude")
# plt.colorbar()

# fig.add_subplot(2,2,3) # ABS PHASE -- azimuth
# plt.imshow(phase_map, cmap="hsv")
# #plt.colorbar()
# plt.title("phase")

# ax = fig.add_subplot(2,2,4, projection='polar')
# ax.set_theta_zero_location('W')
# ax._direction = 2*np.pi

# norm = mpl.colors.Normalize(1*np.pi, -1*np.pi)
# quant_steps = 2056
# cb = mpl.colorbar.ColorbarBase(ax, cmap=cm.get_cmap('hsv', quant_steps),
# 								norm=norm, orientation='horizontal')
# # cb.outline.set_visible(False)
# # ax.set_axis_off()
# ax.set_rlim([-1, 1])

# plt.show()


# imname = which_sesh  + '_allmaps_HSV' + str(reduce_factor) + '_' + key + '.png'
# plt.savefig(outdirs + '/' + imname)
# print outdirs + '/' + imname
