#!/usr/bin/env python2
# coding: utf-8

# FROM plot_absolute_maps_GCaMP.ipnb (JNB)

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
import pandas as pd
import matplotlib.pyplot as plt
import copy
import colorsys

import math

import copy
from scipy.ndimage import filters
from mpl_toolkits.axes_grid1 import make_axes_locatable

# get_ipython().magic(u'matplotlib inline')

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]


parser = optparse.OptionParser()
parser.add_option('--headless', action="store_true", dest="headless", default=False, help="run in headless mode, no figs")
parser.add_option('--reduce', action="store", dest="reduce_val", default="1", help="block_reduce value")
parser.add_option('--path', action="store", dest="path", default="", help="path to data directory")
parser.add_option('-t', '--thresh', action="store", dest="threshold", default=0.5, help="cutoff threshold value")
parser.add_option('-r', '--run', action="store", dest="run", default=1, help="cutoff threshold value")
parser.add_option('--append', action="store", dest="append", default="", help="appended label for analysis structs")
parser.add_option('--mask', action="store", dest="mask", type="choice", choices=['DC', 'blank', 'magmax'], default='DC', help="mag map to use for thresholding: DC | blank | magmax [default: DC]")
parser.add_option('--cmap', action="store", dest="cmap", default='spectral', help="colormap for summary figures [default: spectral]")
parser.add_option('--use-norm', action="store_true", dest="use_norm", default=False, help="compare normalized blank to condition")
parser.add_option('--use-mask', action="store_true", dest="use_mask", default=False, help="use masked phase to calc gradient")
parser.add_option('--shift', action="store_true", dest="shift_phase", default=False, help="shift phase for monotonically increasing vals")

(options, args) = parser.parse_args()

use_norm = options.use_norm
use_mask = options.use_mask
shift_phase = options.shift_phase

headless = options.headless
reduce_factor = (int(options.reduce_val), int(options.reduce_val))
if reduce_factor[0] > 1:
    reduceit=1
else:
    reduceit=0
if headless:
    mpl.use('Agg')

colormap = options.cmap

threshold_type = options.mask #'blank'
threshold = float(options.threshold)
outdir = options.path
run_num = options.run

exptdir = os.path.split(outdir)[0]
sessiondir = os.path.split(exptdir)[0]
print "EXPT: ", exptdir
print "SESSION: ", sessiondir

savedir = os.path.split(outdir)[0]
figdir = os.path.join(savedir, 'figures')
if not os.path.exists(figdir):
    os.makedirs(figdir)


#################################################################################
# GET BLOOD VESSEL IMAGE:
#################################################################################
folders = os.listdir(sessiondir)
figpath = [f for f in folders if f == 'surface'][0]
# figpath = [f for f in folders if f == 'figures'][0]
# print "EXPT: ", exptdir
# print "SESSION: ", sessiondir
print "path to surface: ", figpath

if figpath:
    # figdir = figpath[0]
    tmp_ims = os.listdir(os.path.join(sessiondir, figpath))
    surface_words = ['surface', 'GREEN', 'green', 'Surface', 'Surf']
    ims = [i for i in tmp_ims if any([word in i for word in surface_words])]
    ims = [i for i in ims if '_' in i]
    print ims
    if ims:
        impath = os.path.join(sessiondir, figpath, ims[0])
        # image = Image.open(impath) #.convert('L')
        # imarray = np.asarray(image)
        print os.path.splitext(impath)[1]
        if os.path.splitext(impath)[1] == '.tif':
            tiff = TIFF.open(impath, mode='r')
            surface = tiff.read_image().astype('float')
            tiff.close()
            plt.imshow(surface)
        else:
            image = Image.open(impath) #.convert('L')
            surface = np.asarray(image)
    else:
        surface = np.zeros([200,300])

else: # NO BLOOD VESSEL IMAGE...
    surface = np.zeros([200,300])

if reduceit:
    surface = block_reduce(surface, reduce_factor, func=np.mean)

plt.imshow(surface, cmap='gray')



#################################################################################
# GET DATA STRUCT FILES:
#################################################################################

append = options.append

files = os.listdir(outdir)
files = [f for f in files if os.path.splitext(f)[1] == '.pkl']
# dstructs = [f for f in files if 'D_target_FFT' in f and str(reduce_factor) in f]
# if not dstructs:
#     dstructs = [f for f in files if 'D_' in f and str(reduce_factor) in f] # address older analysis formats


# dstructs = [f for f in files if 'Target_fft' in f]
dstructs = [f for f in files if 'Target_fft' in f and str(reduce_factor) and append in f]
print dstructs
D = dict()
for f in dstructs:
    outfile = os.path.join(outdir, f)
    with open(outfile,'rb') as fp:
        D[f] = pkl.load(fp)

astructs = [f for f in files if 'Amplitude' in f and str(reduce_factor) and append in f]
print astructs
A = dict()
for f in astructs:
    outfile = os.path.join(outdir, f)
    with open(outfile,'rb') as fp:
        A[f] = pkl.load(fp)


# Get specific keys:

bottomkeys = [k for k in D.keys() if 'Bottom' in k or 'Up' in k]
topkeys = [k for k in D.keys() if 'Top' in k or 'Down' in k]

leftkeys = [k for k in D.keys() if 'Left' in k]
rightkeys = [k for k in D.keys() if 'Right' in k]
if threshold_type=='blank':
    blank_key = [k for k in dstructs if 'blank_' in k and append in k][0]
    print "BLANK: ", blank_key

el_keys = [topkeys, bottomkeys]
az_keys = [leftkeys, rightkeys]

print "COND KEYS: "
print "AZ keys: ", az_keys
print "EL keys: ", el_keys


# grab legends:

legend_dir = '/home/juliana/Repositories/retinotopy-mapper/tests/simulation'

# MAKE LEGENDS:

winsize = [1920, 1200]
screen_size = [int(i*0.25) for i in winsize]
print screen_size

create_legend = 1

if create_legend:
    V_left_legend = np.zeros((screen_size[1], screen_size[0]))
    # First, set half the screen width (0 to 239 = to 0 to -pi)
    nspaces_start = np.linspace(0, -1*math.pi, screen_size[0]/2)
    for i in range(screen_size[1]):
        V_left_legend[i][0:screen_size[0]/2] = nspaces_start

    # Then, set right side of screen (240 to end = to pi to 0)
    nspaces_end = np.linspace(1*math.pi, 0, screen_size[0]/2)
    for i in range(screen_size[1]):
        V_left_legend[i][screen_size[0]/2:] = nspaces_end
        
else:
    legend_name = 'V-Left_legend.tif'
    V_left_legend = imread(os.path.join(legend_dir, legend_name))

    
if create_legend:
    V_right_legend = np.zeros((screen_size[1], screen_size[0]))
    # First, set half the screen width (0 to 239 = to 0 to -pi)
    nspaces_start = np.linspace(0, 1*math.pi, screen_size[0]/2)
    for i in range(screen_size[1]):
        V_right_legend[i][0:screen_size[0]/2] = nspaces_start
    # Then, set right side of screen (240 to end = to pi to 0)
    nspaces_end = np.linspace(-1*math.pi, 0, screen_size[0]/2)
    for i in range(screen_size[1]):
        V_right_legend[i][screen_size[0]/2:] = nspaces_end
        
else:
    legend_name = 'V-Right_legend.tif'
    V_right_legend = imread(os.path.join(legend_dir, legend_name))

    
if create_legend:        
    H_down_legend = np.zeros((screen_size[1], screen_size[0]))
    # First, set half the screen width (0 to 239 = to 0 to -pi)
    nspaces_start = np.linspace(0, -1*math.pi, screen_size[1]/2)
    for i in range(screen_size[0]):
        H_down_legend[0:screen_size[1]/2, i] = nspaces_start

    # Then, set right side of screen (240 to end = to pi to 0)
    nspaces_end = np.linspace(1*math.pi, 0, screen_size[1]/2)
    for i in range(screen_size[0]):
        H_down_legend[screen_size[1]/2:, i] = nspaces_end
        
else:
    legend_name = 'H-Down_legend.tif'
    H_down_legend = imread(os.path.join(legend_dir, legend_name))


if create_legend:
    H_up_legend = np.zeros((screen_size[1], screen_size[0]))
    # First, set half the screen width (0 to 239 = to 0 to -pi)
    nspaces_start = np.linspace(0, 1*math.pi, screen_size[1]/2)
    for i in range(screen_size[0]):
        H_up_legend[0:screen_size[1]/2, i] = nspaces_start

    # Then, set right side of screen (240 to end = to pi to 0)
    nspaces_end = np.linspace(-1*math.pi, 0, screen_size[1]/2)
    for i in range(screen_size[0]):
        H_up_legend[screen_size[1]/2:, i] = nspaces_end
else:
    legend_name = 'H-Up_legend.tif'
    H_up_legend = imread(os.path.join(legend_dir, legend_name))

# plt.imshow(V_left_legend, cmap='hsv')
# plt.colorbar()

date = os.path.split(os.path.split(os.path.split(outdir)[0])[0])[1]
experiment = os.path.split(os.path.split(outdir)[0])[1]

colormap = options.cmap


cond_types = ['Left', 'Right', 'Top', 'Bottom']
for cond in cond_types:

    print "RUNNING cond: ", cond
    # run_conds = [cond, str(run_num)+'_', str(reduce_factor), append]
    run_conds = [cond, str(reduce_factor), append]

    if cond=='Left':
        tmp_keys = [k for k in leftkeys if all([c in k for c in run_conds])] #[0]
        legend = V_left_legend

    elif cond=='Right':
        tmp_keys = [k for k in rightkeys if all([c in k for c in run_conds])] #[0]
        legend = V_right_legend

    elif cond=='Top':
        tmp_keys = [k for k in topkeys if all([c in k for c in run_conds])] #[0]
        legend = H_down_legend 

    elif cond=='Bottom':
        tmp_keys = [k for k in bottomkeys if all([c in k for c in run_conds])] #[0]
        legend = H_up_legend


    if tmp_keys==[]:
        print "No matches found from list: %s", cond
    else:
        for curr_key in tmp_keys:
            print "Curr key is: ", curr_key
            curr_amp_key = curr_key.split('Target_fft')[1]
            print "Corresponding AMP key is: ", curr_amp_key

            if 'Right' in curr_key or 'Left' in curr_key:
                imname = 'grad_AZ_%s' % curr_key
            else:
                imname = 'grad_EL_%s' % curr_key  


            curr_map = D[curr_key]['ft']
            Ny = len(D[curr_key]['freqs'])/2.

            curr_phase_map = D[curr_key]['phase_map'] #np.angle(curr_map)
            cont_phase_map = copy.deepcopy(curr_phase_map)
            for x in range(curr_phase_map.shape[0]):
                for y in range(curr_phase_map.shape[1]):
                    if cont_phase_map[x, y] < 0:
                         cont_phase_map[x, y] += 2*math.pi
                            
            cont_legend = copy.deepcopy(legend)
            for x in range(legend.shape[0]):
                for y in range(legend.shape[1]):
                    if cont_legend[x, y] < 0:
                         cont_legend[x, y] += 2*math.pi

            curr_mag_map = D[curr_key]['mag_map']/Ny #np.abs(curr_map)
            DC_mag_map = D[curr_key]['DC_mag']/Ny

            # MAKE AND SAVE FIGURE:
                    
            fig = plt.figure(figsize=(20,10))

            # 1.  SURFACE
            # -----------------------------------
            fig.add_subplot(2,4,1)
            plt.imshow(surface, cmap='gray')
            plt.axis('off')

            # 2. MAG MAP:
            # -----------------------------------
            fig.add_subplot(2,4,2)
            plt.title('magnitude map')
            ax = plt.gca()
            # im = ax.imshow(mag_map, cmap='gray')
            # plt.colorbar()
            # plt.axis('off')
            # plt.title('magnitude')

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            im = ax.imshow(curr_mag_map, cmap='gray')
            plt.colorbar(im, cax=cax, cmap='gray')


            # 3.  MASKED PHASE MAP
            # -----------------------------------
            # fig.add_subplot(2,4,3)
            # plt.axis('off')
            # plt.title(tit)

            # Assign mask to use for thresholding:
            if threshold_type=='DC':
                thresh_map = DC_mag_map
            elif threshold_type=='magmax' or threshold_type=='logmax':
                thresh_map = curr_mag_map
            elif threshold_type=='blank': # 07-27-2016:  this doesnt exist yet!
                blank_mag_map = D[blank_key]['mag_map']/Ny
                thresh_map = blank_mag_map

            # normalize threshold_map to do comparison against 0 map:
            old_min = curr_mag_map.min()
            old_max = curr_mag_map.max()
            new_min = 0
            new_max = 1
            normed_mag_map = np.zeros(curr_mag_map.shape)
            for x in range(curr_mag_map.shape[0]):
                for y in range(curr_mag_map.shape[1]):
                    old_val = curr_mag_map[x, y]
                    normed_mag_map[x, y] = (((old_val - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min
            
            # normalize threshold_map to do comparison against 0 map:
            old_min = thresh_map.min()
            old_max = thresh_map.max()
            new_min = 0
            new_max = 1
            normed_thresh_map = np.zeros(thresh_map.shape)
            for x in range(curr_mag_map.shape[0]):
                for y in range(thresh_map.shape[1]):
                    old_val = thresh_map[x, y]
                    normed_thresh_map[x, y] = (((old_val - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min


            if use_norm is True:
                [mx, my] = np.where(normed_mag_map >= threshold*normed_thresh_map)
                tit = 'Threshold, %.2f of normed %s magnitude' % (threshold, threshold_type)
            else:
                [mx, my] = np.where(curr_mag_map >= threshold*thresh_map)
                tit = 'Threshold, %.2f of %s magnitude' % (threshold, threshold_type)


            phase_mask = np.ones(curr_mag_map.shape) * 100
            if shift_phase:
                phase_map = cont_phase_map
            else:
                phase_map = curr_phase_map

            phase_mask[mx, my] = phase_map[mx, my]
            [nullx, nully] = np.where(phase_mask == 100)
            phase_mask[nullx, nully] = np.nan
            phase_mask = np.ma.array(phase_mask)

            fig.add_subplot(2,4,3)
            plt.axis('off')
            plt.title(tit)
            ax = plt.gca()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)

            im = ax.imshow(surface, cmap='gray')
            im = ax.imshow(phase_mask, cmap=colormap)
            plt.colorbar(im, cax=cax, cmap=colormap)


            # 4. LEGEND
            # # -----------------------------------
            ax = fig.add_subplot(2,4,4)
            plt.imshow(cont_legend, cmap=colormap)
            plt.axis('off')


            # 5. MEAN INTENSITY:
            # -----------------------------------
            fig.add_subplot(2,4,5)
            plt.title('mean intensity')
            plt.axis('off')

            mean_intensity = D[curr_key]['mean_intensity']
            ax = plt.gca()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            im = ax.imshow(mean_intensity, cmap='hot')
            plt.colorbar(im, cax=cax, cmap='hot')



            # -----------------------------------------------
            # 6_END:  GET GRADIENT INFO -- Use SOBEL FILTERS.
            # -----------------------------------------------
            if use_mask:
                IMG = phase_mask #copy.deepcopy(p)
            else:
                IMG = phase_map

            imx = np.zeros(IMG.shape)
            filters.sobel(IMG, 1, imx)
            imy = np.zeros(IMG.shape)
            filters.sobel(IMG, 0, imy)
            sob = np.hypot(imx, imy)

            grad_magnitude = np.sqrt(imx**2+imy**2)

            valmin = -.5
            valmax = .5

            # 6. X-GRADIENT 
            # -----------------------------------
            plt.subplot(2,4,6)
            plt.title('Sobel X')
            ax = plt.gca()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            im = ax.imshow(imx, vmin=valmin, vmax=valmax)
            # plt.colorbar()
            plt.colorbar(im, cax=cax, cmap='gray')

            # 6. Y-GRADIENT
            # -----------------------------------
            plt.subplot(2,4,7)
            plt.title('Sobel Y')
            ax = plt.gca()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            im = ax.imshow(imy, vmin=valmin, vmax=valmax)
            # plt.colorbar()
            plt.colorbar(im, cax=cax, cmap='gray')


            # 6. Y-GRADIENT
            # -----------------------------------
            plt.subplot(2,4,8)
            plt.title('sobel mag')
            ax = plt.gca()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            # im = ax.imshow(grad_magnitude)
            im = ax.imshow(sob, vmin=-1, vmax=1)
            # plt.colorbar()
            plt.colorbar(im, cax=cax, cmap='hot')


            plt.suptitle(['GRADIENT: ', date, experiment, curr_key])
            

            if use_norm:
                norm_flag = 'normed_'
            else:
                norm_flag = 'actual_'

            if shift_phase:
                shift_flag = 'continuous_'
            else:
                shift_flag = 'negpipospi_'


            if use_mask:
                mask_flag = 'masked_'
            else:
                mask_flag = 'nomask_'

            impath = os.path.join(figdir, colormap+'_GRAD_'+mask_flag+shift_flag+norm_flag+'summary_'+curr_key+'.png')
            plt.savefig(impath, format='png')
            print impath

            plt.show()

