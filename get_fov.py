#!/usr/bin/env python2

import matplotlib
matplotlib.use('TkAgg')
import pylab as pl
from psychopy import monitors, visual, tools
import os
import numpy as np
from os.path import expanduser
home = expanduser("~")
import shutil
import math
import cPickle as pkl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage.filters import generic_filter as gf
from skimage.measure import block_reduce
import pandas as pd
import matplotlib.gridspec as gridspec


def convert_values(oldval, newmin, newmax, oldmax=phasemax, oldmin=phasemin):
    oldrange = (oldmax - oldmin)  
    newrange = (newmax - newmin)  
    newval = (((oldval - oldmin) * newrange) / oldrange) + newmin
    return newval

source = '/nas/volume1/widefield/data'
# animal = 'JR040W' # 'JR042W'
# session = '20170307' #20170621'
animal = 'JR042W'
session = '20170621'
data_dir = os.path.join(source, animal, session)

az_condtype = 'Left'
el_condtype = 'Top'

threshold = 0.1
radius = 2
interval = 10
short_axis = False

# Contour map params:
linmin = lin_coord_x.min()
linmax = lin_coord_x.max()
cmap='gist_rainbow'
fontsize = 32
linecolor = 'w'
linewidth = 60


# Get monitor calibs:
monitor_dir = '~/Repositories/retinotopy-mapper/protocols/monitors'
psychopy_monitor_dir = '~/.psychopy2/monitors'

if '~' in monitor_dir:
    monitor_dir = monitor_dir.replace('~', home)

calibs = [c for c in os.listdir(monitor_dir) if c.endswith('calib')]
calib_names = [c[:-6] for c in calibs]

if '~' in psychopy_monitor_dir:
   psychopy_monitor_dir = psychopy_monitor_dir.replace('~', home)
# Copy saved monitor calibs to local .psychopy dir:
if not os.path.exists(psychopy_monitor_dir):
    os.makedirs(psychopy_monitor_dir)
existing_calibs = [c for c in os.listdir(psychopy_monitor_dir) if c.endswith('calib')]
missing_calibs = [c for c in calibs if c not in existing_calibs]    
for c in missing_calibs:
    shutil.copyfile(os.path.join(monitor_dir,c), os.path.join(psychopy_monitor_dir,c))

# Select monitor:
for idx,calib in enumerate(calib_names):
    print idx, calib

mon_idx = input('Select IDX of monitor to use: ')
mon = monitors.Monitor(calib_names[mon_idx])

# Get monitor info:
distance = mon.getDistance()
width = mon.getWidth()
resolution = mon.getSizePix()
aspect = float(resolution[0])/float(resolution[1])
height = width * (1./aspect)
pix_cm = float(width)/float(resolution[0])
print "Pixel size (cm):", pix_cm

width_deg = 2*np.arctan((width)/(2*distance)) * (180./math.pi)
height_deg = 2*np.arctan((height)/(2*distance)) * (180./math.pi)

center = [width_deg/2., height_deg/2.]
interval = 10. 

print "Distance (cm):", distance
print "Resolution:", resolution
print "Width, Height (cm):", width, height
print "Width, Height (deg):", width_deg, height_deg

# deg_per_px = degrees(atan2(.5*width, distance)) / (.5*resolution[0])
# width_deg = resolution[0] * deg_per_px
# print width_deg

bar_width = 1.0
print "Bar (deg):", bar_width


# Use Psychopy's unit conversion since that is what we're using in the stimulus protocol:
total_length = np.copy(width)
total_length_deg = tools.monitorunittools.cm2deg(total_length, mon, correctFlat=False) + bar_width
print "Total length (deg):", total_length_deg
print "Distance to center (deg): ", total_length_deg/2.

# Get surface image from selected session:
# Surface image:
from libtiff import TIFF
if 'surface' in os.listdir(data_dir):
    surface_dir = os.path.join(data_dir, 'surface')
else:
    surface_dir = os.path.join(source, animal, 'surface') #, '*%s*.tif' % session)

surface_fn = [f for f in os.listdir(surface_dir) if f.endswith('.tif') and session in f][0]
print surface_fn
surface_impath = os.path.join(surface_dir, surface_fn)

surftiff = TIFF.open(surface_impath, mode='r')
surface = surftiff.read_image().astype('float')
surftiff.close()
pl.imshow(surface, cmap='gray')
print surface.shape

# Get phase maps:

#TODO:  Fix analysis output to save all maps (single runs and combined) to same pkl file

if average is True:
    # ============================================================================
    # COMBO condition runs:
    # ============================================================================
    condinfo_path = os.path.join(data_dir, 'composite', 'figures', 'CONDS.pkl')
    with open(condinfo_path, 'rb') as f:
        conds = pkl.load(f)
    
    if az_condtype=='Right':
        azdict = conds['right']
    elif az_condtype=='Left':
        azdict = conds['left']
    if el_condtype=='Top':
        eldict = conds['top']
    elif el_condtype=='Bottom':
        eldict = conds['bottom']
    print azdict.keys()

    az_phasemap = azdict['phase']
    az_ratiomap = azdict['ratio']

    el_phasemap = eldict['phase']
    el_ratiomap = eldict['ratio']
else:
    # ============================================================================
    # SINGLE condition runs:
    # ============================================================================
 
    fn = '20170621_JR042W_r2_all_struct.pkl'

    with open(os.path.join(data_dir, fn), 'rb') as f:
        structs = pkl.load(f)

    experiments = structs.keys()
    runs = structs[experiments[0]].keys()

    print experiments
    for idx,run in enumerate(runs):
        print idx, run

    curr_experiment = experiments[0]
    az_run = runs[5]
    el_run = runs[1]

    print curr_experiment
    print "AZ:", az_run
    print "EL:", el_run

    structs[curr_experiment][az_run].keys()

    az_phasemap = structs[curr_experiment][az_run]['phase_map']
    az_ratiomap = structs[curr_experiment][az_run]['ratio_map']

    el_phasemap = structs[curr_experiment][el_run]['phase_map']
    el_ratiomap = structs[curr_experiment][el_run]['ratio_map']
    
# Check FFT map dimensions and adjust surface img, if needed:
if not az_phasemap.shape==surface.shape:
    reduce_val = surface.shape[0]/az_phasemap.shape[0]
    print reduce_val
    if reduce_val>1:
        surface = block_reduce(surface, (reduce_val, reduce_val), func=np.mean)
        print surface.shape
    elif reduce_val<1:
        reduce_val = 1./reduce_val
        az_phasemap = block_reduce(az_phasemap, (reduce_val, reduce_val), func=np.mean)
        el_phasemap = block_reduce(el_phasemap, (reduce_val, reduce_val), func=np.mean)
        az_ratiomap = block_reduce(az_ratiomap, (reduce_val, reduce_val), func=np.mean)
        el_ratiomap = block_reduce(el_ratiomap, (reduce_val, reduce_val), func=np.mean)
        
# Convert to continous range:
az_phase = -1 * az_phasemap
az_phase = az_phase % (2*np.pi)

el_phase = -1 * el_phasemap
el_phase = el_phase % (2*np.pi)


# Threshold maps:
min_thr = min([az_ratiomap.max(), el_ratiomap.max()]) * threshold

# Check threshold for AZIMUTH map:
phasemin = 0
phasemax = 2*np.pi

pl.figure(figsize=(15,5))
pl.subplot(1,3,1); pl.imshow(az_phase, cmap=cmap); pl.axis('off')
ax=pl.subplot(1,3,2); im=pl.imshow(az_ratiomap, cmap='hot'); pl.axis('off'); 
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
pl.colorbar(im, cax=cax)

az_phasemap_thr = np.copy(az_phase)
az_phasemap_thr[az_ratiomap<=min_thr] = np.nan
pl.subplot(1,3,3); pl.imshow(az_phasemap_thr, cmap=cmap); pl.axis('off')

# Check threshold for ELEVATION map:
pl.figure(figsize=(15,5))
pl.subplot(1,3,1); pl.imshow(el_phase, cmap=cmap); pl.axis('off')
ax=pl.subplot(1,3,2); im=pl.imshow(el_ratiomap, cmap='hot'); pl.axis('off'); 
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
pl.colorbar(im, cax=cax)

el_phasemap_thr = np.copy(el_phase)
el_phasemap_thr[el_ratiomap<=min_thr] = np.nan
pl.subplot(1,3,3); pl.imshow(el_phasemap_thr, cmap=cmap); pl.axis('off')


# Low-pass filter phase map w/ uniform circular kernel:
kernel = np.zeros((2*radius+1, 2*radius+1))
y,x = np.ogrid[-radius:radius+1, -radius:radius+1]
mask = x**2 + y**2 <= radius**2
kernel[mask] = 1
az_phasemap_filt = gf(az_phase, np.min, footprint=kernel)
el_phasemap_filt = gf(el_phase, np.min, footprint=kernel)

# Threshold smoothed maps:
az_phasemap_thresh = np.copy(az_phasemap_filt)
az_phasemap_thresh[az_ratiomap<=min_thr] = np.nan

el_phasemap_thresh = np.copy(el_phasemap_filt)
el_phasemap_thresh[el_ratiomap<=min_thr] = np.nan

pl.figure(figsize=(10,5))
pl.subplot(1,2,1); pl.imshow(az_phasemap_thresh, cmap='spectral'); pl.axis('off')
pl.subplot(1,2,2); pl.imshow(el_phasemap_thresh, cmap='spectral'); pl.axis('off')


# MONITOR COORDINATE SPACE:

# Convert degs to centimeters:
C2A_cm = width/2.
C2T_cm = height/2.
C2P_cm = width/2.
C2B_cm = height/2.
print "center 2 Top/Anterior:", C2T_cm, C2A_cm

mapx = np.linspace(-1*C2A_cm, C2P_cm, resolution[0])
mapy = np.linspace(C2T_cm, -1*C2B_cm, resolution[1])

lin_coord_x, lin_coord_y = np.meshgrid(mapx, mapy, sparse=False)

mapcorX, mapcorY = np.meshgrid(range(resolution[0]), range(resolution[1]))

linminW = lin_coord_x.min()
linmaxW = lin_coord_x.max()

linminH = lin_coord_y.min()
linmaxH = lin_coord_y.max()



# LEGENDS:
f1 = pl.figure(figsize=(15,5))
pl.subplot(1,2,1)
currfig = pl.imshow(lin_coord_x, vmin=linminW, vmax=linmaxW,  cmap=cmap)
levels1 = range(int(np.floor(lin_coord_x.min() / interval) * interval), 
                int((np.ceil(lin_coord_x.max() / interval) + 1) * interval), interval)
im1 = pl.contour(mapcorX, mapcorY, lin_coord_x, levels1, colors='k', linewidth=2)
pl.clabel(im1, levels1, fontsize=8, inline=3, fmt='%1.1f') #, inline_spacing=2, manual=label_pos)
#f1.colorbar(currfig, ticks=levels1)
pl.axis('off')

pl.subplot(1,2,2)
if short_axis is True:
    curr_fig = pl.imshow(lin_coord_y, vmin=linminH, vmax=linmaxH, cmap=cmap) #pl.colorbar()
else:
    curr_fig = pl.imshow(lin_coord_y, vmin=linminW, vmax=linmaxW, cmap=cmap) #pl.colorbar()

levels2 = range(int(np.floor(lin_coord_y.min() / interval) * interval), 
                int((np.ceil(lin_coord_y.max() / interval) + 1) * interval), interval)

im2 = pl.contour(mapcorX, mapcorY, lin_coord_y, levels2, colors='k', linewidth=2)
pl.clabel(im2, levels2, fontsize=8, inline=3, fmt='%1.1f') #, inline_spacing=2, manual=label_pos)
pl.axis('off')

# CONVERT MAP VALUES TO COORDINATE SPACE:
imsize = az_phasemap_thresh.shape
az_phasemap_lincoord = np.copy(az_phasemap_thresh)
el_phasemap_lincoord = np.copy(el_phasemap_thresh)

for x in range(az_phasemap_thresh.shape[0]):
    for y in range(az_phasemap_thresh.shape[1]):
        if not np.isnan(az_phasemap_thresh[x,y]):
            az_phasemap_lincoord[x,y] = convert_values(az_phasemap_thresh[x,y], linminW, linmaxW)
            if short_axis is True:
                el_phasemap_lincoord[x,y] = convert_values(el_phasemap_thresh[x,y], linminH, linmaxH)
            else:
                el_phasemap_lincoord[x,y] = convert_values(el_phasemap_thresh[x,y], linminW, linmaxW)


# PLOT ISOAZIMUTH:
imgX, imgY = np.meshgrid(range(imsize[1]), range(imsize[0]))

pl.figure(figsize=(20,20))
pl.imshow(surface, cmap='gray')
#pl.subplot(1,2,1); #pl.imshow(img, cmap='gray')
#pl.imshow(az_phasemap_thr, cmap=cmap, vmin=-1*math.pi, vmax=math.pi); pl.axis('off'); #pl.colorbar()

currfig = pl.imshow(az_phasemap_thr, vmin=phasemin, vmax=phasemax,  cmap=cmap, alpha=0.3)
levels1 = range(int(np.floor(lin_coord_x.min() / interval) * interval), 
                int((np.ceil(lin_coord_x.max() / interval) + 1) * interval), interval)


im1 = pl.contour(imgX, imgY, az_phasemap_lincoord, levels1, colors='w', linewidth=linewidth)
pl.clabel(im1, levels1, fontsize=fontsize, inline=3, fmt='%1.1f') #, inline_spacing=2, manual=label_pos)
pl.axis('off')
pl.title('Azimuth')

# PLOT ISOELEVATION:
imgX, imgY = np.meshgrid(range(imsize[1]), range(imsize[0]))

pl.figure(figsize=(20,20))
pl.imshow(surface, cmap='gray')
#pl.subplot(1,2,1); #pl.imshow(img, cmap='gray')
#pl.imshow(az_phasemap_thr, cmap=cmap, vmin=-1*math.pi, vmax=math.pi); pl.axis('off'); #pl.colorbar()

currfig = pl.imshow(el_phasemap_thr, vmin=phasemin, vmax=phasemax,  cmap=cmap, alpha=0.3)
levels2 = range(int(np.floor(lin_coord_y.min() / interval) * interval), 
                int((np.ceil(lin_coord_y.max() / interval) + 1) * interval), interval)


im2 = pl.contour(imgX, imgY, el_phasemap_lincoord, levels2, colors='w', linewidth=linewidth)
pl.clabel(im2, levels2, fontsize=fontsize, inline=3, fmt='%1.1f') #, inline_spacing=2, manual=label_pos)
pl.axis('off')
pl.title('Elevation')


# CONTOURS ONLY overlaid on surface:
pl.figure(figsize=(15,10))
pl.imshow(surface, cmap='gray')
im1 = pl.contour(imgX, imgY, az_phasemap_lincoord, levels1, colors='c', linewidth=4)
pl.clabel(im1, levels1, fontsize=fontsize, inline=3, fmt='%1.1f') #, inline_spacing=2, manual=label_pos)
im2 = pl.contour(imgX, imgY, el_phasemap_lincoord, levels2, colors='m', linewidth=4)
pl.clabel(im2, levels2, fontsize=fontsize, inline=3, fmt='%1.1f') #, inline_spacing=2, manual=label_pos)
pl.axis('off')


             