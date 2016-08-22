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
parser.add_option('--mask', action="store", dest="mask", type="choice", choices=['DC', 'blank', 'magmax'], default='DC', help="mag map to use for thresholding: DC | blank | magmax [default: DC]")
parser.add_option('--cmap', action="store", dest="cmap", default='spectral', help="colormap for summary figures [default: spectral]")
parser.add_option('--use-norm', action="store_true", dest="use_norm", default=False, help="compare normalized blank to condition")
parser.add_option('--smooth', action="store_true", dest="smooth", default=False, help="smooth? (default sig = 2)")
parser.add_option('--sigma', action="store", dest="sigma_val", default=2, help="sigma for gaussian smoothing")

parser.add_option('--contour', action="store_true", dest="contour", default=False, help="show contour lines for phase map")
parser.add_option('--power', action='store_true', dest='use_power', default=False, help="use power or just magnitude?")


(options, args) = parser.parse_args()

use_power = options.use_power

use_norm = options.use_norm
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
if len([i for i in bottomkeys if 'Up' in i])>0:
    oldflag = True
else:
    oldflag = False

leftkeys = [k for k in D.keys() if 'Left' in k]
rightkeys = [k for k in D.keys() if 'Right' in k]
if threshold_type=='blank':
    blank_key = [k for k in dstructs if 'blank_' in k][0]
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
screen_size = [int(i*0.1) for i in winsize]
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
            curr_amp_key = curr_key.split('Target_fft')[1]
            print "Corresponding AMP key is: ", curr_amp_key

            curr_map = D[curr_key]['ft']
            Ny = len(D[curr_key]['freqs'])/2.

            fig = plt.figure()
            # plt.subplot(2,2,3)
            # plt.imshow(np.angle(curr_map), cmap='spectral', vmin=-1*math.pi, vmax=1*math.pi)
            # # plt.title('AZ: left')
            # plt.axis('off')


            # ax = fig.add_subplot(2,2,4)
            # plt.imshow(legend, cmap='spectral')
            # plt.axis('off')

            # fig.add_subplot(2,2,1)
            # plt.imshow(surface, cmap='gray')
            # plt.axis('off')

            # fig.add_subplot(2,2,2)
            # plt.imshow(D[curr_key]['mag_map']/Ny, cmap='hot')
            # plt.axis('off')
            # plt.colorbar()

            # plt.tight_layout()
            # plt.suptitle(curr_key)


            # SAVE FIG:

            # plt.imshow(np.angle(currmap), cmap='spectral', vmin=-1*math.pi, vmax=1*math.pi)
            # # plt.title('AZ: right to left')
            # # plt.axis('off')
            # print figdir

            # impath = os.path.join(figdir, curr_key+'.png')
            # plt.savefig(impath, format='png')
            # print impath


            # --------------------------------------------------------------------------------------
            # --------------------------------------------------------------------------------------
            # PLOT IT ALL: 
            # --------------------------------------------------------------------------------------
            # --------------------------------------------------------------------------------------

            # In[1319]:

            # curr_key = leftkey

            print "CURR KEY : ", curr_key

            mag_map = D[curr_key]['mag_map']/Ny
            phase_map = D[curr_key]['phase_map']
            power_map = mag_map**2 #D[curr_key]['mag_map']**2

            vmin_val = -1*math.pi
            vmax_val = 1*math.pi
            levels = np.arange(vmin_val, vmax_val, .75) 


            date = os.path.split(os.path.split(os.path.split(outdir)[0])[0])[1]
            experiment = os.path.split(os.path.split(outdir)[0])[1]

            colormap = options.cmap

            # 1.  SURFACE
            # -----------------------------------
            fig.add_subplot(1,3,1)
            plt.imshow(surface, cmap='gray', alpha=0)
            plt.axis('off')
            plt.contour(phase_map, levels, origin='lower', cmap=colormap, linewidths=.7)


            # MAKE AND SAVE FIGURE:

            if 'Left' in curr_key or 'Right' in curr_key:
                imname = 'contour_AZ_HSV_%s' % curr_key
                if 'Left' in curr_key:
                    legend = V_left_legend
                else:
                    legend = V_right_legend
            else:
                imname = 'contour_EL_HSV_%s' % curr_key  
                if 'Top' in curr_key or 'Down' in curr_key:
                    legend = H_down_legend
                else:
                    legend = H_up_legend
                    
                    
            # fig = plt.figure(figsize=(20,10))


            # 5. MAG MAP:
            # -----------------------------------
            fig.add_subplot(1,3,2)
            #power_map = mag_map**2
            if use_power is True:
                plt.imshow(power_map, cmap='hot', vmin=0, vmax=200) #, vmax=15) #, vmin=0) #, vmax=250.0)
    		plt.title('power')
            else:
    		plt.imshow(mag_map, cmap='hot')
    		plt.title('magnitude')

            plt.colorbar()
            plt.axis('off')
            # plt.title('power')

            # 6. LEGEND
            ax = fig.add_subplot(1,3,3)
	    plt.imshow(surface, cmap='gray', alpha=0)
	    plt.axis('off')
            plt.contour(legend, levels, origin='lower', cmap=colormap, linewidths=1)
            plt.axis('off')

            plt.suptitle([date, experiment, curr_key])

            if use_power is True:
    		  power_flag = 'power_'
    	    else:
    		  power_flag = ''
            impath = os.path.join(figdir, 'contour_'+power_flag+imname+'.eps')
            plt.savefig(impath, format='eps')
            print impath

            
