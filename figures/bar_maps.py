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
#import matplotlib.pyplot as plt
import copy
import colorsys

import math
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
parser.add_option('--mask', action="store", dest="mask", type="choice", choices=['DC', 'blank', 'magmax', 'ratio'], default='DC', help="mag map to use for thresholding: DC | blank | magmax [default: DC]")
parser.add_option('--cmap', action="store", dest="cmap", default='spectral', help="colormap for summary figures [default: spectral]")
#parser.add_option('--use-norm', action="store_true", dest="use_norm", default=False, help="compare normalized blank to condition")
parser.add_option('--smooth', action="store_true", dest="smooth", default=False, help="smooth? (default sig = 2)")
parser.add_option('--sigma', action="store", dest="sigma_val", default=2, help="sigma for gaussian smoothing")

parser.add_option('--contour', action="store_true", dest="contour", default=False, help="show contour lines for phase map")
parser.add_option('--power', action='store_true', dest='use_power', default=False, help="use POWER or just magnitude?")
parser.add_option('--ratio', action='store_true', dest='use_ratio', default=False, help="use RATIO or just magnitude?")

parser.add_option('--short-axis', action="store_false", dest="use_long_axis", default=True, help="Used short-axis instead of long?")

(options, args) = parser.parse_args()
use_long_axis = options.use_long_axis


use_power = options.use_power
use_ratio = options.use_ratio

#use_norm = options.use_norm
smooth = options.smooth
sigma_val_num = options.sigma_val
sigma_val = (int(sigma_val_num), int(sigma_val_num))

contour = options.contour


headless = options.headless
reduce_factor = (int(options.reduce_val), int(options.reduce_val))
if reduce_factor[0] > 1:
    reduceit=1
else:
    reduceit=0
if headless:
    import matplotlib as mpl
    mpl.use('Agg')

import matplotlib.pyplot as plt

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
figpath = [f for f in folders if f == 'surface']
# figpath = [f for f in folders if f == 'figures'][0]
# print "EXPT: ", exptdir
# print "SESSION: ", sessiondir
print "path to surface: ", figpath

if figpath:
    # figdir = figpath[0]
    figpath=figpath[0]
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

#plt.imshow(surface, cmap='gray')



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

# astructs = [f for f in files if 'Amplitude' in f and str(reduce_factor) and append in f]
# print astructs
# A = dict()
# for f in astructs:
#     outfile = os.path.join(outdir, f)
#     with open(outfile,'rb') as fp:
#         A[f] = pkl.load(fp)

# if not A:
#     if threshold_type=='ratio' # trying to use ratio-map, but can't
#         print "No amplitude struct found. Use DC or blank:"
#         threshold_type=''
#         user_input=raw_input("\nChoose different threshold map, DC [0] or blank [1]:\n")
#         if int(user_input)==0:
#             threshold_type = 'DC'
#         elif int(user_input)==1:
#             threshold_type = 'blank'

#if threshold_type=='blank':
blank_keys = [k for k in dstructs if 'blank_' in k or 'Blank_' in k] #[0]
if not blank_keys:
    print "Blank condition not found. Using DC."
    threshold_type = 'DC'
else:
    blank_key = blank_keys[0]
    print "Using BLANK key: ", blank_key

# Get specific keys:

bottomkeys = [k for k in D.keys() if 'Bottom' in k or 'Up' in k]
topkeys = [k for k in D.keys() if 'Top' in k or 'Down' in k]
if len([i for i in bottomkeys if 'Up' in i])>0:
    oldflag = True
else:
    oldflag = False

leftkeys = [k for k in D.keys() if 'Left' in k]
rightkeys = [k for k in D.keys() if 'Right' in k]

el_keys = [topkeys, bottomkeys]
az_keys = [leftkeys, rightkeys]

print "COND KEYS: "
print "AZ keys: ", az_keys
print "EL keys: ", el_keys



# grab legends:
use_corrected_screen = True
# legend_dir = '/home/juliana/Repositories/retinotopy-mapper/tests/simulation'

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

# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
# FIX THIS:
# ----------------------------------------------------------------------------------------------
# This adjustment needs to be fixed for cases of using the older Samsung monitor (smaller)
# Also, any scripts in which horizontal condition started at the edge of the screen, rather than
# being centered around the screen middle.

ratio_factor = .5458049 # This is true / hardcoded only for AQUOS monitor.
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------

if (use_corrected_screen is True) and (use_long_axis is True):
    screen_edge = math.pi - (math.pi*ratio_factor)
else:
    screen_edge = 0
    
if create_legend:        
    H_down_legend = np.zeros((screen_size[1], screen_size[0]))
    # First, set half the screen width (0 to 239 = to 0 to -pi)
    # If CORRECTING for true physical screen, start  after 0 (~1.43):
    nspaces_start = np.linspace(-1*screen_edge, -1*math.pi, screen_size[1]/2)
    for i in range(screen_size[0]):
        H_down_legend[0:screen_size[1]/2, i] = nspaces_start
    # Then, set right side of screen (240 to end = to pi to 0)
    nspaces_end = np.linspace(1*math.pi, screen_edge, screen_size[1]/2)
    for i in range(screen_size[0]):
        H_down_legend[screen_size[1]/2:, i] = nspaces_end
else:
    legend_name = 'H-Down_legend.tif'
    H_down_legend = imread(os.path.join(legend_dir, legend_name))

if create_legend:
    H_up_legend = np.zeros((screen_size[1], screen_size[0]))
    # First, set half the screen width (0 to 239 = to 0 to -pi)
    # If CORRECTING for true physical screen, start  after 0 (~1.43):
    nspaces_start = np.linspace(screen_edge, 1*math.pi, screen_size[1]/2)
    for i in range(screen_size[0]):
        H_up_legend[0:screen_size[1]/2, i] = nspaces_start
    # Then, set right side of screen (240 to end = to pi to 0)
    nspaces_end = np.linspace(-1*math.pi, -1*screen_edge, screen_size[1]/2)
    for i in range(screen_size[0]):
        H_up_legend[screen_size[1]/2:, i] = nspaces_end
else:
    legend_name = 'H-Up_legend.tif'
    H_up_legend = imread(os.path.join(legend_dir, legend_name))



# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
# Cycle through ALL conditions, plot:
# ----------------------------------------------------------------------------------------------
# 1.  2x2 :  surface, magnitude, phase, legend
# 2.  2x3 :  

# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------


cond_types = ['Left', 'Right', 'Top', 'Bottom']
for cond in cond_types:
    print "RUNNING cond: ", cond
    # run_conds = [cond, str(run_num)+'_', str(reduce_factor), append]
    if oldflag is True and (cond=='Top' or cond=='Bottom'):
        print "Using old EL condition names..."
        if cond=='Top':
            run_conds = ['Down', str(reduce_factor), append]
        if cond=='Bottom':
            run_conds = ['Up', str(reduce_factor), append]
    else:
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

            # if use_ratio is True:
            #     curr_amp_key_suffix = curr_key.split('Target_fft')[1]
            #     curr_amp_key = [a for a in A.keys() if curr_amp_key_suffix in a][0]
            #     print "Corresponding AMP key is: ", curr_amp_key

            #     ratio_map = A[curr_amp_key]['ratio_map']

            curr_map = D[curr_key]['ft']
            Ny = len(D[curr_key]['freqs'])/2.

            fig = plt.figure()
            plt.subplot(2,2,3)
            plt.imshow(np.angle(curr_map), cmap='spectral', vmin=-1*math.pi, vmax=1*math.pi)
            # plt.title('AZ: left')
            plt.axis('off')


            ax = fig.add_subplot(2,2,4)
            plt.imshow(legend, cmap='spectral')
            plt.axis('off')


            fig.add_subplot(2,2,1)
            plt.imshow(surface, cmap='gray')
            plt.axis('off')

            fig.add_subplot(2,2,2)
            # plt.imshow(D[curr_key]['mag_map']/Ny, cmap='hot')
            plt.imshow(D[curr_key]['ratio_map'], cmap='hot') 
	    plt.axis('off')
            plt.colorbar()


            plt.tight_layout()
            plt.suptitle(curr_key)


            # SAVE FIG:

            # plt.imshow(np.angle(currmap), cmap='spectral', vmin=-1*math.pi, vmax=1*math.pi)
            # # plt.title('AZ: right to left')
            # plt.axis('off')
            print figdir

            impath = os.path.join(figdir, curr_key+'.png')
            plt.savefig(impath, format='png')
            print impath



            # --------------------------------------------------------------------------------------
            # --------------------------------------------------------------------------------------
            # PLOT IT ALL: 
            # --------------------------------------------------------------------------------------
            # --------------------------------------------------------------------------------------
            
            date = os.path.split(os.path.split(os.path.split(outdir)[0])[0])[1]
            experiment = os.path.split(os.path.split(outdir)[0])[1]

            colormap = options.cmap

            print "CURR KEY : ", curr_key

            mag_map = D[curr_key]['mag_map']/Ny
            power_map = mag_map**2 #D[curr_key]['mag_map']**2
            ratio_map = D[curr_key]['ratio_map']
            DC_mag_map = D[curr_key]['DC_mag']/Ny
            
            # ------------------------------------------------------------------
            # Not all structs will have this... need to run get_fft_bar.py with:
            # options:  --rolling --interpolate 
            # True as of 09/15/2016 --------------------------------------------
            ratio_map = D[curr_key]['ratio_map']


            phase_map = D[curr_key]['phase_map']


            if smooth is True:
                # sigma_val = sigma_val
                phase_map[phase_map<0]=2*math.pi+phase_map[phase_map<0]
                phase_map = ndimage.gaussian_filter(phase_map, sigma=sigma_val, order=0)
                vmin_val = 0
                vmax_val = 2*math.pi
                legend[legend<0]=2*math.pi+legend[legend<0]
            else:
                vmin_val = -1*math.pi
                vmax_val = 1*math.pi

            if contour is True:
                levels = np.arange(vmin_val, vmax_val, .25)  # Boost the upper limit to avoid truncation errors.


            # MAKE AND SAVE FIGURE:

            if 'Left' in curr_key or 'Right' in curr_key:
                imname = 'AZ_HSV_%s' % curr_key
                # if 'Left' in curr_key:
                #     legend = V_left_legend
                # else:
                #     legend = V_right_legend
            else:
                imname = 'EL_HSV_%s' % curr_key  
                # if 'Top' in curr_key or 'Down' in curr_key:
                #     legend = H_down_legend
                # else:
                #     legend = H_up_legend
                    
                    
            fig = plt.figure(figsize=(20,10))

            # 1.  SURFACE
            # -----------------------------------
            fig.add_subplot(2,3,1)
            plt.imshow(surface, cmap='gray')
            plt.axis('off')

            # 2.  PHASE MAP
            # -----------------------------------
            fig.add_subplot(2,3,2)
            if contour is True:
                plt.contour(phase_map, levels, origin='upper', cmap=colormap, linewidths=2)
            else:
                plt.imshow(phase_map, cmap=colormap, vmin=vmin_val, vmax=vmax_val)
            plt.axis('off')
            plt.title('phase')

            # 3. PHASE MASKED BY MAG, OVERRLAY:
            # -----------------------------------
            fig.add_subplot(2,3,3)

            # Assign mask to use for thresholding:
            if threshold_type=='DC':
                thresh_map = copy.deepcopy(DC_mag_map)
            elif threshold_type=='magmax' or threshold_type=='logmax':
                thresh_map = copy.deepcopy(mag_map)
            elif threshold_type=='blank': # 07-27-2016:  this doesnt exist yet!
                blank_mag_map = D[blank_key]['mag_map']/Ny
                thresh_map = copy.deepcopy(blank_mag_map)
            elif threshold_type=='ratio':
                thresh_map = copy.deepcopy(ratio_map)

            # normalize threshold_map to do comparison against 0 map:
            # old_min = mag_map.min()
            # old_max = mag_map.max()
            # new_min = 0
            # new_max = 1
            # normed_mag_map = np.zeros(mag_map.shape)
            # for x in range(mag_map.shape[0]):
            #     for y in range(mag_map.shape[1]):
            #         old_val = mag_map[x, y]
            #         normed_mag_map[x, y] = (((old_val - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min
            
            # # normalize threshold_map to do comparison against 0 map:
            # old_min = thresh_map.min()
            # old_max = thresh_map.max()
            # new_min = 0
            # new_max = 1
            # normed_thresh_map = np.zeros(thresh_map.shape)
            # for x in range(mag_map.shape[0]):
            #     for y in range(thresh_map.shape[1]):
            #         old_val = thresh_map[x, y]
            #         normed_thresh_map[x, y] = (((old_val - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min

            # if threshold_type=='logmax':
            #     [mx, my] = np.where(np.log(mag_map) >= log_thresh*np.log(mag_map.max()))
            #     plot_title = 'masked, >= %s of log mag max' % str(log_thresh)
                
            # elif threshold_type=='DC':
            #     [mx, my] = np.where(mag_map >= thresh*(DC_mag_map+0.001))
            # #     [mx, my] = np.where(0.5*mag_map >= DC_mag_map)
            #     plot_title = 'masked, >= %s of DC mag' % str(thresh)
                
            # elif threshold_type=='magmax':
            #     [mx, my] = np.where(mag_map >= thresh*mag_map.max())
            #     plot_title = 'masked, >= %s of mag max' % str(thresh)

            # if use_norm is True:
            #     [mx, my] = np.where(normed_mag_map >= threshold*normed_thresh_map)
            #     tit = 'Threshold, %.2f of normed %s magnitude' % (threshold, threshold_type)
            # else:
            #     [mx, my] = np.where(mag_map >= threshold*thresh_map)
            #     tit = 'Threshold, %.2f of %s magnitude' % (threshold, threshold_type)


            # phase_mask = np.ones(mag_map.shape) * 100
            # phase_mask[mx, my] = phase_map[mx, my]
            # # tit = 'Threshold, %.2f of %s magnitude' % (threshold, threshold_type)

            # [nullx, nully] = np.where(phase_mask == 100)
            # # print len(mx)
            # phase_mask[nullx, nully] = np.nan

            phase_mask = copy.deepcopy(phase_map)

            if not threshold_type=='ratio':
                [mx, my] = np.where(mag_map < threshold*thresh_map)
                phase_mask[mx, my] = np.nan
                mask_title = 'Masked with %s, each pixel > threshold %s' % (threshold_type, str(threshold))
            else:
                phase_mask[thresh_map < (threshold*thresh_map.max())] = np.nan
                mask_title = 'Masked with ratio, %s of max' % str(threshold)

            phase_mask = np.ma.array(phase_mask)
            plt.imshow(surface, cmap='gray')
            plt.imshow(phase_mask, cmap=colormap, vmin=vmin_val, vmax=vmax_val)
            plt.axis('off')
            plt.title(mask_title)

            # 4. MEAN INTENSITY:
            # -----------------------------------
            fig.add_subplot(2,3,4)
            mean_intensity = D[curr_key]['mean_intensity']
            plt.imshow(mean_intensity, cmap='hot')
            plt.axis('off')
            plt.colorbar()
            plt.title('mean intensity')

            # 5. MAG MAP:
            # -----------------------------------
            fig.add_subplot(2,3,5)
            #power_map = mag_map**2
            if use_power is True:
                plt.imshow(power_map, cmap='hot', vmin=0, vmax=200) #, vmax=15) #, vmin=0) #, vmax=250.0)
                plt.title('power')
                plt.colorbar()
                plt.axis('off')
            elif use_ratio is True:
                plt.imshow(ratio_map, cmap='hot') #, vmin=0, vmax=200) #, vmax=15) #, vmin=0) #, vmax=250.0)
                plt.title('mag ratio')
                plt.colorbar()
                plt.axis('off')
            else:
                plt.imshow(mag_map, cmap='hot')
                plt.title('magnitude')
                plt.colorbar()
                plt.axis('off')
            # plt.title('power')

            # 6. LEGEND
            ax = fig.add_subplot(2,3,6)
            plt.imshow(legend, cmap=colormap)
            plt.axis('off')

            plt.suptitle([date, experiment, curr_key])
            

            # if use_norm:
            #     norm_flag = 'normed_'
            # else:
            #     norm_flag = 'actual_'

            impath = os.path.join(figdir, 'summary_'+colormap+'_'+curr_key+'.png')
            plt.savefig(impath, format='png')
            print impath

            plt.show()

            # ------------------------------------------------------------------
            # HSV PLOT
            # ------------------------------------------------------------------

            # THIS SEEMS TO WORK BETTER (FROM BOTTOM OF CIRC jnb)


            # Ny = len(D[curr_key]['freqs'])/2.

            # fig = plt.figure()
            # mag_map = D[curr_key]['mag_map'] / Ny
            # phase_map = D[curr_key]['phase_map']

            # DC_map = D[curr_key]['DC_mag']/Ny
            # # blank_map = D[blank_key]['mag_map']/Ny

            # print "mag range: ", mag_map.min(), mag_map.max()
            # print "phase range: ", phase_map.min(), phase_map.max()


            # Get normed PHASE map for stimulation condN for HSV composite:

            old_min = -math.pi #phase_map.min()
            old_max = math.pi #phase_map.max()
            new_min = 0
            new_max = 1
            normed_phase_map = np.zeros(phase_map.shape)
            for x in range(phase_map.shape[0]):
                for y in range(phase_map.shape[1]):
                    old_val = phase_map[x, y]
                    normed_phase_map[x, y] = (((old_val - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min


            # Get normed MAGNITUDE map for stimulation condN for HSV composite:
        if use_power is True:
            mag_map = power_map #mag_map**2
            old_min = mag_map.min()
            old_max = mag_map.max()
            new_min = 0
            new_max = 1
            normed_mag_map = np.zeros(mag_map.shape)
            for x in range(mag_map.shape[0]):
                for y in range(mag_map.shape[1]):
                    old_val = mag_map[x, y]
                    normed_mag_map[x, y] = (((old_val - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min

                    
            # CHOOSE MAG METHOD TO THRESHOLD OFF OF:

            # if thresh_method=='magmax':
            #     thresh_map = copy.deepcopy(mag_map)
            #     plot_title = 'masked, >= %s of mag max' % str(cutoff_val)
            # elif thresh_method=='DC':
            #     thresh_map = copy.deepcopy(DC_map)
            #     plot_title = 'masked, >= %s of DC mag' % str(cutoff_val)
            # elif thresh_method=='blank':
            #     thresh_map = copy.deepcopy(blank_map)
            #     plot_title = 'masked, >= %s of blank mag' % str(cutoff_val)

                
            # # NORMALIZE THERSHOLD MAP???
            # old_min = thresh_map.min()
            # old_max = thresh_map.max()
            # new_min = 0
            # new_max = 1
            # normed_thresh_map = np.zeros(thresh_map.shape)
            # for x in range(thresh_map.shape[0]):
            #     for y in range(thresh_map.shape[1]):
            #         old_val = thresh_map[x, y]
            #         normed_thresh_map[x, y] = (((old_val - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min


            hue = normed_phase_map
            sat = np.ones(hue.shape)
            val = normed_mag_map

            print "HUE range: ", hue.min(), hue.max()
            print "VAL range: ", val.min(), val.max()

            HSV = np.ones(val.shape + (3,))
            HSV[...,0] = hue
            HSV[...,2] = sat * 1
            HSV[...,1] = val

            # plt.imshow(normed_phase_map, cmap='hsv')
            # plt.colorbar()

            print normed_mag_map.min()
            print normed_mag_map.max()


            ## REMOVE BELOW THRESH:

            print "Cutoff at: ", threshold

            nons = []
            for x in range(mag_map.shape[0]):
                for y in range(mag_map.shape[1]):
                    # if use_norm is True:
                    #     if normed_mag_map[x, y] < normed_thresh_map[x, y]*threshold:
                    #         nons.append([x,y])
                    # else:
                    if not threshold_type=='ratio':
                        # i.e,. if thresholding using magnitude map of condN:
                        if mag_map[x, y] < thresh_map[x, y]*threshold:
                            nons.append([x,y])
                    else:
                        # Or, can threshold using ratio-map of magnitudes:
                        if thresh_map[x,y] < thresh_map.max()*threshold:
                            nons.append([x, y])
                    
            # NOTE ON THRESHOLDING:
            # If use normed-mag against normed-threshold-map, get good removal of baddies.
            # BUT, if use actual mag-map values against actual blank/DC map conditions, too much stuff gets included...

                            
            print "N pixels below threshold:  ", len(nons)

            ##
            # HSV TO RGB:

            # import colorsys
            convmap = np.empty(HSV.shape)

            for i in range(HSV.shape[0]):
                for j in range(HSV.shape[1]):

                        convmap[i, j, :] = colorsys.hsv_to_rgb(HSV[i,j,:][0], HSV[i,j,:][1], HSV[i,j,:][2])
                        
            print "HSV range: ", HSV.min(), HSV.max()
            print convmap[i,j,:]
            print convmap.min()

            ##
            # MASK:

            alpha_channel = np.ones(convmap[:,:,1].shape)
            print alpha_channel.shape
            for i in nons:
                alpha_channel[i[0], i[1]] = 0

            composite = np.empty((alpha_channel.shape[0], alpha_channel.shape[1], 4))
            composite[:,:,0:3] = convmap[:,:,:]

            composite[:,:,3] = alpha_channel


            # PLOT:


            # MAKE AND SAVE FIGURE:

            if 'Left' in curr_key or 'Right' in curr_key:
                imname = 'AZ_HSV_%s' % curr_key
                if 'Left' in curr_key:
                    print "left"
                    legend = V_left_legend
                else:
                    legend = V_right_legend #V_right_legend
            else:
                imname = 'EL_HSV_%s' % curr_key  
                if 'Top' in curr_key or 'Down' in curr_key:
                    legend = H_down_legend
                else:
                    legend = H_up_legend

            date = os.path.split(os.path.split(os.path.split(outdir)[0])[0])[1]
            experiment = os.path.split(os.path.split(outdir)[0])[1]
                    
            plt.figure(figsize=(10,10))

            # plt.subplot(1,3,1)
            # plt.imshow(surface, 'gray')
            # plt.axis('off')
            plt.subplot(2,3,1)
            plt.imshow(ratio_map, 'hot')
            plt.title("ratio")
            plt.colorbar()
            plt.axis('off')

            
            if blank_key:
                blank_mag_map = D[blank_key]['mag_map']/Ny
                blank_intensity = D[blank_key]['mean_intensity']

            deltaF_mag = (mag_map - blank_mag_map) / blank_mag_map

            deltaF_intensity = (mean_intensity - blank_intensity) / blank_intensity

            plt.subplot(2,3,2)
            plt.imshow(deltaF_mag, 'hot')
            plt.title("mag rel. to blank")
            plt.colorbar()
            plt.axis('off')

            plt.subplot(2,3,3)
            plt.imshow(deltaF_intensity, 'hot')
            plt.title("intensity rel. to blank")
            plt.colorbar()
            plt.axis('off')

            # plt.subplot(1,3,2)
            plt.subplot(2,3,4)
            plt.imshow(surface, 'gray')
            plt.imshow(composite, 'hsv')
            plt.axis('off')
            plt.title(cond)
            # plt.colorbar()

            plt.subplot(2,3,5)
            plt.imshow(legend, cmap='hsv')
            plt.axis('off')

            plt.suptitle([date, experiment, curr_key])

            plt.tight_layout()
                
            # impath = os.path.join(outdir, imname+'.svg')
            # plt.savefig(impath, format='svg', dpi=1200)

            
            # if use_norm:
            #     norm_flag = 'normed_'
            # else:
            #     norm_flag = 'actual_'
        
            if use_power is True:
              power_flag = 'power_'
            elif use_ratio is True:
              power_flag = 'ratio'
            else:
              power_flag = 'magnitude'

            impath = os.path.join(figdir, imname+'_'+power_flag+threshold_type+'.png')
            plt.savefig(impath, format='png')
            print impath

            plt.show()



            # ----------------------------------------------
            # OVERLAY surface w/ composite phase map
            # ----------------------------------------------
            plt.figure()
            plt.subplot(1,2,1)
            
            plt.imshow(composite, 'hsv', alpha=1)
            plt.imshow(surface, 'gray', alpha=0.3)
            plt.axis('off')

            plt.subplot(1,2,2)
            plt.imshow(legend, cmap='hsv', alpha=1)
            plt.axis('off')
            
            plt.title('overlay '+cond)

            impath = os.path.join(figdir, cond+'_OVERLAY_'+power_flag+threshold_type+'.png')
            plt.savefig(impath, format='png')
            print impath
            plt.show()
            # In[1376]:

            # Checkout the threshold map...

            # plt.subplot(2,2,1)
            # plt.imshow(mag_map, vmin=0, vmax=mag_map.max()) 
            # plt.title('magnitude at target freq')
            # plt.colorbar()

            # plt.subplot(2,2,2)
            # plt.imshow(normed_mag_map, vmin=0, vmax=1)
            # plt.title('normed magnitude')
            # plt.colorbar()

            # plt.subplot(2,2,3)
            # plt.imshow(thresh_map,  vmin=0, vmax=mag_map.max())
            # if threshold_type=='DC':
            #     plt.title('magnitude at DC')
            # # elif thresh_method=='magmax'
            # #     plt.
            # plt.colorbar()

            # plt.subplot(2,2,4)
            # plt.imshow(normed_thresh_map, vmin=0, vmax=1)
            # plt.title('normed mag at DC)')
            # plt.colorbar()


            # imname = "magnitude_%s_thresh%s%%_%s" % (threshold_type, str(int(threshold*100)), os.path.splitext(curr_key)[0])

            # savedir = os.path.split(outdir)[0]
            # figdir = os.path.join(savedir, 'figures')
            # if not os.path.exists(figdir):
            #     os.makedirs(figdir)
                
            # impath = os.path.join(figdir, imname+'.png')
            # plt.savefig(impath, format='png')
            # print impath


            #plt.show()


            # --------------------------------------------------------------------------------------
            # --------------------------------------------------------------------------------------
            # PLOT IT ALL 2: 
            # --------------------------------------------------------------------------------------
            # --------------------------------------------------------------------------------------

#             # DELTA MAOPS!

#             fig = plt.figure(figsize=(20,10))

#             # 1.  SURFACE
#             # -----------------------------------
#             fig.add_subplot(2,4,1)
#             plt.imshow(surface, cmap='gray')
#             plt.axis('off')

#             # 2.  PHASE MAP
#             # -----------------------------------
#             fig.add_subplot(2,4,2)
#             if contour is True:
#                 plt.contour(phase_map, levels, origin='upper', cmap=colormap, linewidths=2)
#             else:
#                 plt.imshow(phase_map, cmap=colormap, vmin=vmin_val, vmax=vmax_val)
#             plt.axis('off')
#             plt.title('phase')

#             # 3. PHASE MASKED BY MAG, OVERRLAY:
#             # -----------------------------------
#             fig.add_subplot(2,4,3)

#             # Assign mask to use for thresholding:
#             if threshold_type=='DC':
#                 thresh_map = DC_mag_map
#             elif threshold_type=='magmax' or threshold_type=='logmax':
#                 thresh_map = mag_map
#             elif threshold_type=='blank': # 07-27-2016:  this doesnt exist yet!
#                 blank_mag_map = D[blank_key]['mag_map']/Ny
#                 thresh_map = blank_mag_map

#             # normalize threshold_map to do comparison against 0 map:
#             old_min = mag_map.min()
#             old_max = mag_map.max()
#             new_min = 0
#             new_max = 1
#             # normed_mag_map = np.zeros(mag_map.shape)
#             # for x in range(mag_map.shape[0]):
#             #     for y in range(mag_map.shape[1]):
#             #         old_val = mag_map[x, y]
#             #         normed_mag_map[x, y] = (((old_val - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min
            
#             # normalize threshold_map to do comparison against 0 map:
#             # old_min = thresh_map.min()
#             # old_max = thresh_map.max()
#             # new_min = 0
#             # new_max = 1
#             # normed_thresh_map = np.zeros(thresh_map.shape)
#             # for x in range(mag_map.shape[0]):
#             #     for y in range(thresh_map.shape[1]):
#             #         old_val = thresh_map[x, y]
#             #         normed_thresh_map[x, y] = (((old_val - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min

#             # if use_norm is True:
#             #     [mx, my] = np.where(normed_mag_map >= threshold*normed_thresh_map)
#             #     tit = 'Threshold, %.2f of normed %s magnitude' % (threshold, threshold_type)
#             # else:
#             #     [mx, my] = np.where(mag_map >= threshold*thresh_map)
#             #     tit = 'Threshold, %.2f of %s magnitude' % (threshold, threshold_type)

#             phase_mask = np.ones(mag_map.shape) * 100
#             phase_mask[mx, my] = phase_map[mx, my]
#             # tit = 'Threshold, %.2f of %s magnitude' % (threshold, threshold_type)

#             [nullx, nully] = np.where(phase_mask == 100)
#             # print len(mx)
#             phase_mask[nullx, nully] = np.nan
#             phase_mask = np.ma.array(phase_mask)
#             plt.imshow(surface, cmap='gray')
#             plt.imshow(phase_mask, cmap=colormap, vmin=vmin_val, vmax=vmax_val)
#             plt.axis('off')
#             plt.title(tit)

#             # 4. MEAN INTENSITY:
#             # -----------------------------------
#             fig.add_subplot(2,4,5)
#             mean_intensity = D[curr_key]['mean_intensity']
#             plt.imshow(mean_intensity, cmap='hot')
#             plt.axis('off')
#             plt.colorbar()
#             plt.title('mean intensity')

#             # 5. MAG MAP:
#             # -----------------------------------
#             fig.add_subplot(2,4,6)
#             plt.imshow(mag_map, cmap='gray')
#             plt.colorbar()
#             plt.axis('off')
#             plt.title('magnitude')


#             # 5b. DELTA MAG MAP:
#             # -----------------------------------
#             delta_mag = (mag_map-thresh_map)/thresh_map

#             fig.add_subplot(2,4,7)
#             plt.imshow(delta_mag, cmap='gray', vmax=20)
#             plt.colorbar()
#             plt.axis('off')
#             plt.title('perc. change over %s' % threshold_type)


#             # 6. LEGEND
#             ax = fig.add_subplot(2,4,8)
#             plt.imshow(legend, cmap=colormap)
#             plt.axis('off')

#             plt.suptitle([date, experiment, curr_key])
            

#             if use_norm:
#                 norm_flag = 'normed_'
#             else:
#                 norm_flag = 'actual_'

#             impath = os.path.join(figdir, 'DELTA_'+colormap+'_'+norm_flag+'summary_'+curr_key+'.png')
#             plt.savefig(impath, format='png')
#             print impath

# #            plt.show()



# # MERGE??

# I_az = np.zeros((leftmap.shape[0], leftmap.shape[1], 3))
# I_el = np.zeros((topmap.shape[0], topmap.shape[1], 3))

# I = np.zeros((leftmap.shape[0], leftmap.shape[1], 3))
# I[:,:,0] = np.angle(leftmap)
# I[:,:,1] = np.angle(topmap)


# # # define the colormap
# # cmap = plt.cm.jet
# # # extract all colors from the .jet map
# # # cmaplist = [cmap(i) for i in range(cmap.N)]
# # step = 100
# # cmaplist = [cmap(i) for i in np.arange(0, 255+step, step)]
# # # force the first color entry to be grey
# # cmaplist[0] = (.5,.5,.5,1.0)
# # # create the new map
# # cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

# plt.imshow(I)

# # plt.subplot(1,2,1)
# # plt.contour(np.angle(leftmap), cmap='bwr_r', alpha=1)
# # # plt.contour(np.angle(topmap), cmap='YlGn', alpha=1)
# # plt.axis('off')

# # plt.subplot(1,2,2)
# # plt.contour(V_left_legend, cmap='bwr_r', alpha=1)
# # print V_left_legend.shape
# # # plt.contour(H_up_legend, cmap='YlGn', alpha=1)
# # print H_up_legend.shape

# # plt.axis('off')
# # plt.colorbar()


# # In[1309]:

# # CONVER TO QUADS?

# # quad_left = np.angle(leftmap)<0
# # quad_right = np.angle(leftmap)>0

# quad_left = np.angle(rightmap)>0
# quad_right = np.angle(rightmap)<0


# # quad_upper = np.angle(topmap)<0 
# # quad_lower = np.angle(topmap)>0 # bottom half is POSITIVE for downward-movnig bar (TOP or "bottom")

# quad_upper = np.angle(bottommap)>0 
# quad_lower = np.angle(bottommap)<0 # bottom half is POSITIVE for downward-movnig bar (TOP or "bottom")

# upper_left = quad_left&quad_upper
# upper_right = quad_right&quad_upper
# lower_left = quad_left&quad_lower
# lower_right = quad_right&quad_lower

# I = np.zeros(quad_left.shape)
# I[upper_left] = 0
# I[upper_right] = 1
# I[lower_left] = 2
# I[lower_right] = 3

# colormap = np.array(['r','b','y','g'])

# print I


# # In[1310]:

# import matplotlib.pyplot as plt
# from matplotlib import colors

# # np.random.seed(101)
# # zvals = np.random.rand(100, 100) * 10

# # make a color map of fixed colors
# # cmap = colors.ListedColormap(['white', 'red'])
# cmap = colors.ListedColormap(['red', 'blue', 'yellow', 'green'], 'indexed')
# bounds=[0,1,2,3,4]
# norm = colors.BoundaryNorm(bounds, cmap.N)



# # bounds=[0,5,10]
# # norm = colors.BoundaryNorm(bounds, cmap.N)

# # # tell imshow about color map so that only set colors are used
# # img = plt.imshow(zvals, interpolation='nearest', origin='lower',
# #                     cmap=cmap, norm=norm)

# # make a color bar
# # plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, ticks=[0, 5, 10])

# # plt.savefig('redwhite.png')
# # plt.show()


# # colormap = np.array(['r', 'g', 'b'])
# # plt.scatter(a[0], a[1], s=50, c=colormap[categories])


# plt.subplot(1,2,1)
# # plt.imshow(I,  interpolation='none', cmap=cmap) #, cmap=colormap) #, cmap=colormap[I[upper_left]], alpha=1)
# img = plt.imshow(I, interpolation='none', origin='lower', cmap=cmap, norm=norm)
# # plt.colorbar()
# plt.axis('off')


# plt.subplot(1,2,2)
# # make a color bar
# quad_legend = [[0,0,0,0,1,1,1,1],[0,0,0,0,1,1,1,1],[0,0,0,0,1,1,1,1],[2,2,2,2,3,3,3,3],[2,2,2,2,3,3,3,3],[2,2,2,2,3,3,3,3]]
# # plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds)

# plt.imshow(quad_legend,interpolation='none', cmap=cmap)
# plt.axis('off')
# # plt.imshow(upper_right, cmap='gray', alpha=0.5)

# # plt.subplot(1,2,2)
# # plt.imshow(upper_right)

# plt.suptitle('COMBINE LEFT AND TOP')
# plt.tight_layout()


# # In[1311]:

# # MASK!!!

# # V-LEFT:
# # MASK WITH MAGNITUDE:
# # LEFT

# log_thresh = 0.8
# thresh = 0.5
# use_log = 1

# curr_key = leftkey

# fig = plt.figure()
# mag_map = D[curr_key]['mag_map']
# phase_map = D[curr_key]['phase_map']

# ###################################
# fig.add_subplot(2,2,1)

# plt.imshow(mag_map, cmap='gray')
# # plt.colorbar()

# ###################################
# fig.add_subplot(2,2,2)
# if use_log:
#     [mx, my] = np.where(np.log(mag_map) >= log_thresh*np.log(mag_map.max()))
# else:
#     [mx, my] = np.where(mag_map >= thresh*mag_map.max())

# mask = np.ones(mag_map.shape) * 100
# mask[mx, my] = I[mx, my]
# [nullx, nully] = np.where(mask == 100)
# mask[nullx, nully] = np.nan
# mask = np.ma.array(mask)

# plt.imshow(mask, cmap='gray')
# # plt.colorbar()

# ###################################
# fig.add_subplot(2,2,3)

# phase_mask = np.ones(mag_map.shape) * 100
# phase_mask[mx, my] = I[mx, my]
# phase_mask[nullx, nully] = np.nan
# phase_mask = np.ma.array(phase_mask)
# # plt.imshow(phase_mask, cmap=cmap) #, vmin=-1*math.pi, vmax=math.pi)

# plt.imshow(phase_mask, interpolation='none', origin='lower', cmap=cmap, norm=norm)

# ###################################
# fig.add_subplot(2,2,4)
# plt.imshow(quad_legend,interpolation='none', cmap=cmap)
# plt.axis('off')







# In[ ]:




# In[1362]:


# # THIS SEEMS TO WORK BETTER (FROM BOTTOM OF CIRC jnb)

# use_mag_max = 0
# use_DC = 1

# curr_key = leftkey

# Ny = len(D[curr_key]['freqs'])/2.
# fig = plt.figure()
# mag_map = D[curr_key]['mag_map'] / Ny
# phase_map = D[curr_key]['phase_map']

# DC_map = D[curr_key]['DC_mag']/Ny


# print "mag range: ", mag_map.min(), mag_map.max()
# print "phase range: ", phase_map.min(), phase_map.max()

# old_min = -math.pi #phase_map.min()
# old_max = math.pi #phase_map.max()
# new_min = 0
# new_max = 1
# normed_phase_map = np.zeros(phase_map.shape)
# for x in range(phase_map.shape[0]):
#     for y in range(phase_map.shape[1]):
#         old_val = phase_map[x, y]
#         normed_phase_map[x, y] = (((old_val - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min

# old_min = mag_map.min()
# old_max = mag_map.max()
# new_min = 0
# new_max = 1
# normed_mag_map = np.zeros(mag_map.shape)
# for x in range(mag_map.shape[0]):
#     for y in range(mag_map.shape[1]):
#         old_val = mag_map[x, y]
#         normed_mag_map[x, y] = (((old_val - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min


# hue = normed_phase_map
# sat = np.ones(hue.shape)
# val = normed_mag_map
# print "HUE range: ", hue.min(), hue.max()
# print "VAL range: ", val.min(), val.max()

# HSV = np.ones(val.shape + (3,))
# HSV[...,0] = hue
# HSV[...,2] = sat * 1
# HSV[...,1] = val

# # plt.imshow(normed_phase_map, cmap='hsv')
# # plt.colorbar()

# print normed_mag_map.min()
# print normed_mag_map.max()


# ## REMOVE BELOW THRESH:

# cutoff_val = 0.5

# old_min = DC_map.min()
# old_max = DC_map.max()
# new_min = 0
# new_max = 1
# normed_DC_map = np.zeros(DC_map.shape)
# for x in range(DC_map.shape[0]):
#     for y in range(DC_map.shape[1]):
#         old_val = DC_map[x, y]
#         normed_DC_map[x, y] = (((old_val - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min


# import copy
# if use_mag_max:
#     cutoff = cutoff_val * max(normed_mag_map.ravel())
#     plot_title = 'masked, >= %s of mag max' % str(cutoff_val)
# elif use_DC:
#     cutoff = cutoff_val * max(normed_DC_map.ravel())
#     plot_title = 'masked, >= %s of DC mag' % str(cutoff_val)
    
# print "Cutoff at: ", cutoff
# thresh_val = copy.deepcopy(val)

# nons = []
# for x in range(thresh_val.shape[0]):
#     for y in range(thresh_val.shape[1]):

# #         if val[x, y] < cutoff:
#         if val[x, y] < normed_DC_map[x, y]*cutoff_val:
            
#             nons.append([x,y])
# print "N pixels below threshold:  ", len(nons)


# # if use_log:
# #     [mx, my] = np.where(np.log(mag_map) >= log_thresh*np.log(mag_map.max()))
# #     plot_title = 'masked, >= %s of log mag max' % str(log_thresh)
    
# # elif use_DC:
# #     [mx, my] = np.where(mag_map >= thresh*(DC_mag_map+0.001))
# # #     [mx, my] = np.where(0.5*mag_map >= DC_mag_map)
# #     plot_title = 'masked, >= %s of DC mag' % str(thresh)
    
# # else:
# #     [mx, my] = np.where(mag_map >= thresh*mag_map.max())
# #     plot_title = 'masked, >= %s of mag max' % str(thresh)

# # mask = np.ones(mag_map.shape) * 100
# # mask[mx, my] = phase_map[mx, my]

# # [nullx, nully] = np.where(mask == 100)
# # print len(mx)
# # mask[nullx, nully] = np.nan
# # mask = np.ma.array(mask)
# # plt.imshow(surface, cmap='gray')
# # plt.imshow(mask, cmap=colormap)
# # plt.axis('off')
# # plt.title(plot_title)

# # nons = [[x, y] for x,y in zip(mx,my)]


# ##
# # HSV TO RGB:

# import colorsys
# convmap = np.empty(HSV.shape)

# for i in range(HSV.shape[0]):
#     for j in range(HSV.shape[1]):

#             convmap[i, j, :] = colorsys.hsv_to_rgb(HSV[i,j,:][0], HSV[i,j,:][1], HSV[i,j,:][2])
# print "HSV range: ", HSV.min(), HSV.max()
# print convmap[i,j,:]
# print convmap.min()

# ##
# # MASK:

# alpha_channel = np.ones(convmap[:,:,1].shape)
# print alpha_channel.shape
# for i in nons:
#     alpha_channel[i[0], i[1]] = 0

# composite = np.empty((alpha_channel.shape[0], alpha_channel.shape[1], 4))
# composite[:,:,0:3] = convmap[:,:,:]

# composite[:,:,3] = alpha_channel





# # In[734]:

# ######################################################################
# # ABSOLUTE AZIMUTH
# ######################################################################
# # For LEFTMAP - RIGHTMAP, color legend follows LEFT-only map direction:

# azimuth_phase = ( np.angle(leftmap) - np.angle(rightmap) ) / 2.

# # SEE ABOVE COLORBAR
# fig = plt.figure()
# ax = fig.add_subplot(1,2,1)
# plt.imshow(V_left_legend, cmap='spectral')
# ax.axes.get_xaxis().set_visible(False)
# ax.axes.get_yaxis().set_visible(False)

# plt.title('AZ: absolute')
# fig.add_subplot(1,2,2)
# plt.imshow(azimuth_phase, cmap='spectral', vmin=-1*math.pi, vmax=1*math.pi)
# plt.title('V-Left - V-Right')


# # In[735]:

# ######################################################################
# # ABSOLUTE ELEVATION
# ######################################################################

# elevation_phase = ( np.angle(topmap) - np.angle(bottommap) ) / 2.

# fig = plt.figure()
# ax = fig.add_subplot(1,2,1)
# plt.imshow(H_down_legend, cmap='spectral')
# ax.axes.get_xaxis().set_visible(False)
# ax.axes.get_yaxis().set_visible(False)

# plt.title('EL: absolute')
# fig.add_subplot(1,2,2)
# plt.imshow(elevation_phase, cmap='spectral',  vmin=-1*math.pi, vmax=1*math.pi)
# plt.title('H-Down - H-Up')


# # In[736]:

# ######################################################################
# # DELAY VERT:
# ######################################################################

# fig = plt.figure()

# colormap = 'spectral'

# # 1. Delay map
# fig.add_subplot(3,3,1)
# # delay_vert = (np.angle(leftmap) + np.angle(rightmap)) / 2.
# delay_vert = np.angle(leftmap * rightmap) / 2.
# plt.imshow(delay_vert, cmap=colormap,  vmin=-1*math.pi, vmax=1*math.pi)
# plt.axis('off')
# plt.title("Delay")

# # 2. blank

# # 3. LEFT-map shifted
# fig.add_subplot(3,3,4)
# # shift_left = np.angle(leftmap.conjugate()) - delay_vert
# shift_left = np.angle(leftmap) - delay_vert
# plt.imshow(shift_left, cmap=colormap, vmin=-1*math.pi, vmax=1*math.pi)
# plt.axis('off')
# plt.title('left map SHIFTED')

# # 4. LEFT-map relative
# ax = fig.add_subplot(3,3,5)
# plt.imshow(np.angle(leftmap), cmap=colormap, vmin=-1*math.pi, vmax=1*math.pi)
# plt.axis('off')
# plt.title('left, unshifted')

# # 5.  LEFT-map LEGEND
# ax = fig.add_subplot(3,3,6)
# plt.imshow(V_left_legend, cmap=colormap, vmin=-1*math.pi, vmax=1*math.pi)
# plt.axis('off')

# # 6. RIGHT-map shifted
# fig.add_subplot(3,3,7)
# # shift_right = np.angle(rightmap) - delay_vert
# shift_right = np.angle(rightmap.conjugate()) - delay_vert
# plt.imshow(shift_right, cmap=colormap, vmin=-1*math.pi, vmax=1*math.pi)
# plt.axis('off')
# plt.title('right map, SHIFTED')

# # 7. RIGHT-map relative
# ax = fig.add_subplot(3,3,8)
# plt.imshow(np.angle(rightmap), cmap=colormap, vmin=-1*math.pi, vmax=1*math.pi)
# plt.axis('off')
# plt.title('right, unshifted')

# # 8. RIGHT-map LEGEND
# ax = fig.add_subplot(3,3,9)
# plt.imshow(V_right_legend, cmap=colormap, vmin=-1*math.pi, vmax=1*math.pi)
# plt.axis('off')


# # In[737]:

# # azimuth_phase = ( np.angle(leftmap) - np.angle(rightmap) ) / 2.
# azimuth_phase = ( np.angle(leftmap / rightmap) ) / 2.

# plt.imshow(azimuth_phase, cmap='spectral', vmin=-1*math.pi, vmax=1*math.pi)
# plt.title('ELEVATION: absolute by subtraction')


# # In[738]:

# # azimuth_phase = ( np.angle(leftmap) - np.angle(rightmap) ) / 2.
# fig = plt.figure()
# fig.add_subplot(1,2,1)
# azimuth_phase = np.angle(leftmap / rightmap) 
# # azimuth_phase = np.angle(rightmap / leftmap) 

# plt.imshow(azimuth_phase, cmap='spectral', vmin=-1*math.pi, vmax=1*math.pi)
# plt.title('ELEVATION: absolute w/ doubled-map')

# fig.add_subplot(1,2,2)
# plt.imshow(double_left_legend, cmap='spectral', vmin=-1*math.pi, vmax=1*math.pi)


# # In[739]:

# ######################################################################
# # DELAY HORIZ:
# ######################################################################

# fig = plt.figure()

# # 1. Delay map
# fig.add_subplot(3,3,1)
# delay_horz = np.angle(topmap * bottommap) / 2.
# plt.imshow(delay_horz, cmap='spectral',  vmin=-1*math.pi, vmax=1*math.pi)
# plt.axis('off')
# plt.title("Delay")

# # 2. blank

# # 3. DOWN-map shifted
# fig.add_subplot(3,3,4)
# # shift_down = np.angle(downmap.conjugate()) - delay_horz
# shift_top = np.angle(topmap) - delay_horz
# plt.imshow(shift_top, cmap='spectral', vmin=-1*math.pi, vmax=1*math.pi)
# plt.axis('off')
# plt.title('DOWN map shifted')

# # 4. DOWN-map relative
# ax = fig.add_subplot(3,3,5)
# plt.imshow(np.angle(topmap), cmap='spectral', vmin=-1*math.pi, vmax=1*math.pi)
# plt.axis('off')
# plt.title('DOWN map')

# # 5.  Down-map LEGEND
# ax = fig.add_subplot(3,3,6)
# plt.imshow(H_down_legend, cmap='spectral', vmin=-1*math.pi, vmax=1*math.pi)
# plt.axis('off')

# # 6. UP-map shifted
# fig.add_subplot(3,3,7)
# shift_bottom = np.angle(bottommap.conjugate()) - delay_horz
# # shift_up = delay_horz - np.angle(upmap.conjugate())
# plt.imshow(shift_bottom, cmap='spectral', vmin=-1*math.pi, vmax=1*math.pi)
# plt.axis('off')
# plt.title('UP map shifted')

# # 7. UP-map relative
# ax = fig.add_subplot(3,3,8)
# plt.imshow(np.angle(bottommap), cmap='spectral', vmin=-1*math.pi, vmax=1*math.pi)
# plt.axis('off')
# plt.title('UP map')

# # 8. UP-map LEGEND
# ax = fig.add_subplot(3,3,9)
# plt.imshow(H_up_legend, cmap='spectral', vmin=-1*math.pi, vmax=1*math.pi)
# plt.axis('off')


# # In[740]:

# # elevation_phase = ( np.angle(downmap) - np.angle(upmap) ) / 2.
# elevation_phase = ( np.angle(topmap / bottommap) ) / 2.

# plt.imshow(elevation_phase, cmap='spectral', vmin=-1*math.pi, vmax=1*math.pi)
# plt.title('ELEVATION: absolute by subtraction')


# # In[741]:

# # elevation_phase = ( np.angle(downmap) - np.angle(upmap) ) / 2.
# elevation_phase = np.angle(topmap / bottommap)

# fig = plt.figure()
# fig.add_subplot(1,2,1)
# plt.imshow(elevation_phase, cmap='spectral', vmin=-1*math.pi, vmax=1*math.pi)
# plt.title('ELEVATION: absolute w/ DOUBLED map')

# fig.add_subplot(1,2,2)
# plt.imshow(double_down_legend, cmap='spectral', vmin=-1*math.pi, vmax=1*math.pi)


# # In[742]:

# #######################################
# # Set THRESHOLD params:
# #######################################

# thresh = 0.3
# log_thresh = 0.7
# use_log = 1


# # In[743]:


# # V-LEFT:
# # MASK WITH MAGNITUDE:
# # LEFT

# log_thresh = 0.8
# thresh = 0.5
# use_log = 1

# curr_key = leftkey

# fig = plt.figure()
# mag_map = D[curr_key]['mag_map']
# phase_map = D[curr_key]['phase_map']

# ###################################
# fig.add_subplot(1,3,1)

# plt.imshow(mag_map, cmap='gray')
# # plt.colorbar()

# ###################################
# fig.add_subplot(1,3,2)
# if use_log:
#     [mx, my] = np.where(np.log(mag_map) >= log_thresh*np.log(mag_map.max()))
# else:
#     [mx, my] = np.where(mag_map >= thresh*mag_map.max())

# mask = np.ones(mag_map.shape) * 100
# mask[mx, my] = mag_map[mx, my]
# [nullx, nully] = np.where(mask == 100)
# mask[nullx, nully] = np.nan
# mask = np.ma.array(mask)

# plt.imshow(mask, cmap='gray')
# # plt.colorbar()

# ###################################
# fig.add_subplot(1,3,3)

# phase_mask = np.ones(mag_map.shape) * 100
# phase_mask[mx, my] = phase_map[mx, my]
# phase_mask[nullx, nully] = np.nan
# phase_mask = np.ma.array(phase_mask)
# plt.imshow(phase_mask, cmap='spectral', vmin=-1*math.pi, vmax=math.pi)


# # In[744]:

# # V-RIGHT:  MASK WITH MAGNITUDE:
# # RIGHT 

# log_thresh = 0.8
# thresh = 0.5
# use_log = 1

# curr_key = rightkey

# fig = plt.figure()
# mag_map = D[curr_key]['mag_map']
# phase_map = D[curr_key]['phase_map']

# ###################################
# fig.add_subplot(1,3,1)

# plt.imshow(mag_map, cmap='gray')
# # plt.colorbar()

# ###################################
# fig.add_subplot(1,3,2)
# if use_log:
#     [mx, my] = np.where(np.log(mag_map) >= log_thresh*np.log(mag_map.max()))
# else:
#     [mx, my] = np.where(mag_map >= thresh*mag_map.max())

# mask = np.ones(mag_map.shape) * 100
# mask[mx, my] = mag_map[mx, my]
# [nullx, nully] = np.where(mask == 100)
# mask[nullx, nully] = np.nan
# mask = np.ma.array(mask)

# plt.imshow(mask, cmap='gray')
# # plt.colorbar()

# ###################################
# fig.add_subplot(1,3,3)

# phase_mask = np.ones(mag_map.shape) * 100
# phase_mask[mx, my] = phase_map[mx, my]
# phase_mask[nullx, nully] = np.nan
# phase_mask = np.ma.array(phase_mask)
# plt.imshow(phase_mask, cmap='spectral', vmin=-1*math.pi, vmax=math.pi)


# # In[745]:

# # MASK WITH MAGNITUDE:
# # H-DOWN: 
# log_thresh = 0.8
# thresh = 0.5
# use_log = 1

# curr_key = topkey

# fig = plt.figure()
# mag_map = D[curr_key]['mag_map']
# phase_map = D[curr_key]['phase_map']

# ###################################
# fig.add_subplot(1,3,1)

# plt.imshow(mag_map, cmap='gray')
# # plt.colorbar()

# ###################################
# fig.add_subplot(1,3,2)
# if use_log:
#     [mx, my] = np.where(np.log(mag_map) >= log_thresh*np.log(mag_map.max()))
# else:
#     [mx, my] = np.where(mag_map >= thresh*mag_map.max())

# mask = np.ones(mag_map.shape) * 100
# mask[mx, my] = mag_map[mx, my]
# [nullx, nully] = np.where(mask == 100)
# mask[nullx, nully] = np.nan
# mask = np.ma.array(mask)

# plt.imshow(mask, cmap='gray')
# # plt.colorbar()

# ###################################
# fig.add_subplot(1,3,3)

# phase_mask = np.ones(mag_map.shape) * 100
# phase_mask[mx, my] = phase_map[mx, my]
# phase_mask[nullx, nully] = np.nan
# phase_mask = np.ma.array(phase_mask)
# plt.imshow(phase_mask, cmap='spectral', vmin=-1*math.pi, vmax=math.pi)


# # In[746]:

# # MASK WITH MAGNITUDE:
# # H-UP: 

# log_thresh = 0.8
# thresh = 0.5
# use_log = 1

# curr_key = bottomkey

# fig = plt.figure()
# mag_map = D[curr_key]['mag_map']
# phase_map = D[curr_key]['phase_map']

# ###################################
# fig.add_subplot(1,3,1)

# plt.imshow(mag_map, cmap='gray')
# # plt.colorbar()

# ###################################
# fig.add_subplot(1,3,2)
# if use_log:
#     [mx, my] = np.where(np.log(mag_map) >= log_thresh*np.log(mag_map.max()))
# else:
#     [mx, my] = np.where(mag_map >= thresh*mag_map.max())

# mask = np.ones(mag_map.shape) * 100
# mask[mx, my] = mag_map[mx, my]
# [nullx, nully] = np.where(mask == 100)
# mask[nullx, nully] = np.nan
# mask = np.ma.array(mask)

# plt.imshow(mask, cmap='gray')
# # plt.colorbar()

# ###################################
# fig.add_subplot(1,3,3)

# phase_mask = np.ones(mag_map.shape) * 100
# phase_mask[mx, my] = phase_map[mx, my]
# phase_mask[nullx, nully] = np.nan
# phase_mask = np.ma.array(phase_mask)
# plt.imshow(phase_mask, cmap='spectral', vmin=-1*math.pi, vmax=math.pi)


# # In[ ]:




# # In[747]:

# ######################################################################
# # ABSOLUTE AZIMUTH -- SHIFT NEGATIVE VALS
# ######################################################################

# phase_left = np.angle(leftmap)
# phase_right = np.angle(rightmap)

# for x in range(phase_left.shape[0]):
#     for y in range(phase_left.shape[1]):
#         if phase_left[x,y] < 0:
#             phase_left[x,y] += 2*math.pi

# for x in range(phase_right.shape[0]):
#     for y in range(phase_right.shape[1]):
#         if phase_right[x,y] < 0:
#             phase_right[x,y] += 2*math.pi

        
# # plt.imshow(phase_left, cmap='spectral', vmin=-1*math.pi, vmax=1*math.pi)
# # phase_left

# # plt.subplot(1,2,2)
# # plt.imshow(phase_right, cmap='spectral')
# # plt.colorbar()
        
# az = (phase_left - phase_right) / 2.
# # plt.imshow(az, cmap='spectral') #, vmin=-1*math.pi, vmax=1*math.pi)
# plt.imshow(az, cmap='spectral') #, vmin=0, vmax=2*math.pi)
# plt.colorbar()
# # plt.colorbar()
# # x = np.where(phase_left<0)
# # len(x[1])
# # phase_left.max()
# az.min()


# # In[748]:

# # GCaMP6f map - Vertically-moving right looks dec:

# use_log = 1
# log_thresh = 0.7
# thresh = 0.3

# curr_key = rightkey

# fig = plt.figure()

# mag_map = D[curr_key]['mag_map']
# phase_map = D[curr_key]['phase_map']

# fig.add_subplot(1,2,1)

# plt.imshow(surface, cmap='gray')

# if use_log:
#     [mx, my] = np.where(np.log(mag_map) >= log_thresh*np.log(mag_map.max()))
# else:
#     [mx, my] = np.where(mag_map >= thresh*mag_map.max())

# mask = np.ones(mag_map.shape) * 100
# mask[mx, my] = mag_map[mx, my]
# [nullx, nully] = np.where(mask == 100)
# mask[nullx, nully] = np.nan
# mask = np.ma.array(mask)

# plt.imshow(mask, cmap='gray')
# # plt.colorbar()

# phase_mask = np.ones(mag_map.shape) * 100
# phase_mask[mx, my] = phase_map[mx, my]
# phase_mask[nullx, nully] = np.nan
# phase_mask = np.ma.array(phase_mask)
# plt.imshow(phase_mask, cmap='spectral', vmin=-1*math.pi, vmax=math.pi)
# plt.axis('off')


# ax = fig.add_subplot(1,2,2)
# plt.imshow(V_right_legend, cmap='spectral')
# ax.axes.get_xaxis().set_visible(False)
# ax.axes.get_yaxis().set_visible(False)
# plt.axis('off')

# imname = 'AZ_right_phase_overlay_withkey'
# impath = os.path.join(outdir, imname+'.svg')
# plt.savefig(impath, format='svg', dpi=1200)

# impath = os.path.join(outdir, imname+'.jpg')
# plt.savefig(impath, format='jpg')


# print impath


# # In[749]:

# # fig = plt.figure()

# curr_key = topkey

# fig = plt.figure()

# mag_map = D[curr_key]['mag_map']
# phase_map = D[curr_key]['phase_map']


# ###################################

# use_log = 1
# log_thresh = 0.7
# thresh = 0.5

# fig.add_subplot(1,2,1)

# plt.imshow(surface, cmap='gray')

# if use_log:
#     [mx, my] = np.where(np.log(mag_map) >= log_thresh*np.log(mag_map.max()))
# else:
#     [mx, my] = np.where(mag_map >= thresh*mag_map.max())

# mask = np.ones(mag_map.shape) * 100
# mask[mx, my] = mag_map[mx, my]
# [nullx, nully] = np.where(mask == 100)
# mask[nullx, nully] = np.nan
# mask = np.ma.array(mask)

# plt.imshow(mask, cmap='gray')
# # plt.colorbar()

# phase_mask = np.ones(mag_map.shape) * 100
# phase_mask[mx, my] = phase_map[mx, my]
# phase_mask[nullx, nully] = np.nan
# phase_mask = np.ma.array(phase_mask)
# plt.imshow(phase_mask, cmap='spectral', vmin=-1*math.pi, vmax=math.pi)
# plt.axis('off')


# ax = fig.add_subplot(1,2,2)
# plt.imshow(H_up_legend, cmap='spectral')
# ax.axes.get_xaxis().set_visible(False)
# ax.axes.get_yaxis().set_visible(False)
# plt.axis('off')

# imname = 'EL_bottom_phase_overlay_withkey'
# impath = os.path.join(outdir, imname+'.svg')
# plt.savefig(impath, format='svg', dpi=1200)

# impath = os.path.join(outdir, imname+'.jpg')
# plt.savefig(impath, format='jpg')


# print impath


# # In[750]:

# # fig = plt.figure()

# use_log = 1
# log_thresh = 0.8
# # thresh = 0.2

# curr_key = topkey
# print curr_key

# fig = plt.figure()

# mag_map = D[curr_key]['mag_map']
# phase_map = D[curr_key]['phase_map']


# ###################################

# # use_log = 1
# # log_thresh = 0.7

# fig.add_subplot(1,2,1)

# plt.imshow(surface, cmap='gray')

# if use_log:
#     [mx, my] = np.where(np.log(mag_map) >= log_thresh*np.log(mag_map.max()))
# else:
#     [mx, my] = np.where(mag_map >= thresh*mag_map.max())

# mask = np.ones(mag_map.shape) * 100
# mask[mx, my] = mag_map[mx, my]
# [nullx, nully] = np.where(mask == 100)
# mask[nullx, nully] = np.nan
# mask = np.ma.array(mask)

# plt.imshow(mask, cmap='gray')
# # plt.colorbar()

# phase_mask = np.ones(mag_map.shape) * 100
# phase_mask[mx, my] = phase_map[mx, my]
# phase_mask[nullx, nully] = np.nan
# phase_mask = np.ma.array(phase_mask)
# plt.imshow(phase_mask, cmap='spectral', vmin=-1*math.pi, vmax=math.pi)
# plt.axis('off')


# ax = fig.add_subplot(1,2,2)
# plt.imshow(H_down_legend, cmap='spectral')
# ax.axes.get_xaxis().set_visible(False)
# ax.axes.get_yaxis().set_visible(False)
# plt.axis('off')

# imname = 'EL_top_phase_overlay_withkey'
# impath = os.path.join(outdir, imname+'.svg')
# plt.savefig(impath, format='svg', dpi=1200)

# impath = os.path.join(outdir, imname+'.jpg')
# plt.savefig(impath, format='jpg')


# print impath


# # In[981]:

# # fig = plt.figure()

# curr_key = leftkey

# fig = plt.figure()

# mag_map = D[curr_key]['mag_map']
# phase_map = D[curr_key]['phase_map']


# ###################################

# use_log = 1
# log_thresh = 0.8
# # thresh = 0.3

# fig.add_subplot(1,2,1)

# plt.imshow(surface, cmap='gray')

# if use_log:
#     [mx, my] = np.where(np.log(mag_map) >= log_thresh*np.log(mag_map.max()))
# else:
#     [mx, my] = np.where(mag_map >= thresh*mag_map.max())

# mask = np.ones(mag_map.shape) * 100
# mask[mx, my] = mag_map[mx, my]
# [nullx, nully] = np.where(mask == 100)
# mask[nullx, nully] = np.nan
# mask = np.ma.array(mask)

# plt.imshow(mask, cmap='gray')
# # plt.colorbar()

# phase_mask = np.ones(mag_map.shape) * 100
# phase_mask[mx, my] = phase_map[mx, my]
# phase_mask[nullx, nully] = np.nan
# phase_mask = np.ma.array(phase_mask)
# plt.imshow(phase_mask, cmap='spectral', vmin=-1*math.pi, vmax=math.pi)
# plt.axis('off')


# ax = fig.add_subplot(1,2,2)
# plt.imshow(V_left_legend, cmap='spectral')
# ax.axes.get_xaxis().set_visible(False)
# ax.axes.get_yaxis().set_visible(False)
# plt.axis('off')

# imname = 'AZ_left_phase_overlay_withkey'
# impath = os.path.join(outdir, imname+'.svg')
# plt.savefig(impath, format='svg', dpi=1200)

# impath = os.path.join(outdir, imname+'.jpg')
# plt.savefig(impath, format='jpg')


# print impath


# # # print D.keys()

# # In[ ]:



