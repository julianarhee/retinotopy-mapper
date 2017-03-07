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

def get_ratio_mask(phase_map, ratio_map, threshold):
    phase_mask = copy.deepcopy(phase_map)
    phase_mask[ratio_map < (threshold*ratio_map.max())] = np.nan
    phase_mask = np.ma.array(phase_mask)
    return phase_mask


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

import matplotlib.cm as cm
import numpy.linalg as la
import scipy.ndimage as ndimage
 

parser = optparse.OptionParser()
parser.add_option('--headless', action="store_true", dest="headless", default=False, help="run in headless mode, no figs")
parser.add_option('--reduce', action="store", dest="reduce_val", default="1", help="block_reduce value")
parser.add_option('--path', action="store", dest="path", default="", help="path to data directory")
# parser.add_option('-t', '--thresh', action="store", dest="threshold", default=0.5, help="cutoff threshold value")
parser.add_option('-r', '--run', action="store", dest="run", default=1, help="cutoff threshold value")
parser.add_option('--append', action="store", dest="append", default="", help="appended label for analysis structs")
parser.add_option('--mask-type', action="store", dest="mask_type", type="choice", choices=['DC', 'blank', 'magmax', 'ratio'], default='DC', help="mag map to use for thresholding: DC | blank | magmax [default: DC]")
parser.add_option('--cmap', action="store", dest="cmap", default='spectral', help="colormap for summary figures [default: spectral]")
parser.add_option('--smooth', action="store_true", dest="smooth", default=False, help="smooth? (default sig = 2)")
parser.add_option('--sigma', action="store", dest="sigma_val", default=2, help="sigma for gaussian smoothing")

# parser.add_option('--contour', action="store_true", dest="contour", default=False, help="Show contour lines for phase map")
# parser.add_option('--power', action='store_true', dest='use_power', default=False, help="Use power or just magnitude?")

parser.add_option('--right', action='store_false', dest='use_left', default=True, help="Use left-bottom or right-top config?")
parser.add_option('--noclean', action='store_false', dest='get_clean', default=True, help="Save borderless, clean maps for COREG")

parser.add_option('--avg', action='store_true', dest='use_avg', default=False, help="Use averaged maps or single runs?")
parser.add_option('--mask', action='store_true', dest='use_mask', default=False, help="Use masked phase maps")
parser.add_option('--threshold', action="store", dest="threshold", default=0.01, help="Threshold (max of ratio map)")
parser.add_option('--alpha', action="store", dest="alpha_val", default=0.5, help="Alpha value for overlays")

parser.add_option('--short-axis', action="store_false", dest="use_long_axis", default=True, help="Used short-axis instead of long?")
parser.add_option('--average', action="store_true", dest="show_avg", default=False, help="Show averaged maps or only save single runs?")
parser.add_option('--new', action="store_true", dest="create_new", default=False, help="Create new map struct or no?")

(options, args) = parser.parse_args()

create_new = options.create_new
show_avg = options.show_avg
use_avg = options.use_avg
use_left = options.use_left
get_clean = options.get_clean
use_mask = options.use_mask

smooth = options.smooth
sigma_val_num = options.sigma_val
sigma_val = (int(sigma_val_num), int(sigma_val_num))

# contour = options.contour
# use_power = options.use_power

headless = options.headless
reduce_val = int(options.reduce_val)
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

threshold_type = options.mask_type #'blank'
threshold = float(options.threshold)
outdir = options.path
run_num = options.run

subject = os.path.split(os.path.split(outdir)[0])[1]
date = os.path.split(outdir)[1]
conditions = os.listdir(outdir)
conditions = [i for i in conditions if subject in i and 'Hz' in i]

print "EXPT: ", date
print "CONDITIONS: ", conditions

composite_dir = os.path.join(outdir, 'composite')
fig_dir = os.path.join(composite_dir, 'figures')
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

#date = os.path.split(os.path.split(os.path.split(outdir)[0])[0])[1]
#experiment = os.path.split(os.path.split(outdir)[0])[1]

# --------------------------------------------------------------------
# Get blood vessel image:
# --------------------------------------------------------------------

folders = os.listdir(outdir)
figpath = [f for f in folders if f == 'surface']
if not figpath:
    figpath = [f for f in folders if f == 'figures']
print "path to surface: ", figpath

if figpath:
    # figdir = figpath[0]
    figpath=figpath[0]
    tmp_ims = os.listdir(os.path.join(outdir, figpath))
    surface_words = ['surface', 'GREEN', 'green', 'Surface', 'Surf']
    ims = [i for i in tmp_ims if any([word in i for word in surface_words])]
    ims = [i for i in ims if date in i]
    print ims
    if ims:
        impath = os.path.join(outdir, figpath, ims[0])
        print os.path.splitext(impath)[1]
        if os.path.splitext(impath)[1] == '.tif':
            tiff = TIFF.open(impath, mode='r')
            surface = tiff.read_image().astype('float')
            tiff.close()
            #plt.imshow(surface)
        else:
            image = Image.open(impath) #.convert('L')
            surface = np.asarray(image)
    else:
        surface = np.zeros([200,300])
    print surface.shape
else: # NO BLOOD VESSEL IMAGE...
    surface = np.zeros([200,300])
    print "No blood vessel image found. Using empty."

if reduceit:
    surface = block_reduce(surface, reduce_factor, func=np.mean)
    print "Reduced image to: ", surface.shape


# --------------------------------------------------------------------
# GET DATA STRUCT FILES:
# --------------------------------------------------------------------

append = options.append

struct_fns = os.listdir(outdir)
struct_fns = [f for f in struct_fns if os.path.splitext(f)[1] == '.pkl' and 'r'+str(reduce_val) in f]

if len(struct_fns) > 1:
    if len(struct_fns) == 1: # composite struct exists
        composite_struct_fn = os.path.join(outdir, struct_fns[0])

    elif len(struct_fns) > 1:
        print "Found more than 1 composite struct file for session %s: " % date
        for struct_idx, struct_fn in enumerate(struct_fns):
            print struct_idx, struct_fn
        user_input=raw_input("\nChoose one [0,1...]:\n")
        if user_input=='':
            composite_struct_fn = struct_fns[0] # just take first cond key 
        else:
            composite_struct_fn = struct_fns[int(user_input)]

    with open(composite_struct_fn, 'rb') as rf:
        D = pkl.load(rf)

elif len(struct_fns)==0 or create_new is True:
    print "No composite struct found. Creating new."
    composite_struct_fn = '{date}_{animal}_r{reduceval}_struct.pkl'.format(date=date, animal=subject, reduceval=reduce_val)
    print "New struct name is: %s" % composite_struct_fn

    D = dict()
    for condition in conditions:
        condition_dir = os.path.join(outdir, condition, 'structs')
	if not os.path.exists(condition_dir):
	    continue
        condition_structs = os.listdir(condition_dir)
        condition_structs = [f for f in condition_structs if '.pkl' in f and 'fft' in f and append in f]
        D[condition] = dict()
        for condition_struct in condition_structs:
            curr_condition_struct = os.path.join(condition_dir, condition_struct)
            curr_cond_key = condition_struct.split('Target_fft_')[1].split('_.pkl')[0]
            with open(curr_condition_struct, 'rb') as f:
                D[condition][curr_cond_key] = pkl.load(f)

    path_to_composite_struct_fn = os.path.join(outdir, composite_struct_fn)
    with open(path_to_composite_struct_fn, 'wb') as wf:
        pkl.dump(D, wf, protocol=pkl.HIGHEST_PROTOCOL)
    print "New struct name is: %s" % path_to_composite_struct_fn



CONDS = dict()
condition_keys = D.keys()
condition_types = ['Left', 'Right', 'Top', 'Bottom']

for condition_type in condition_types:
    condkey = condition_type.lower()
    if 'Left' in condition_type or 'Right' in condition_type:
	direction = 'AZIMUTH'
    else:
        direction = 'ELEVATION'

    print "Select session for %s maps (%s):" % (direction, condition_type)

    for cond_idx, cond_fn in enumerate(condition_keys):
	print cond_idx, cond_fn
    user_input=raw_input("\nChoose a session [0,1...]:\n")
    selected_condition = condition_keys[int(user_input)]

    run_keys = D[selected_condition].keys()
    run_keys = [r for r in run_keys if condition_type in r and str(reduce_factor) in r]
    for run_idx, run_fn in enumerate(run_keys):
	print run_idx, run_fn
    using_average = False
    user_input=raw_input("\nChoose %s run [0,1...]:\n" % condition_type)
    if len(user_input)==1:
	selected_run = run_keys[int(user_input)]
    elif len(user_input)>1:
	using_average = True
	run_idxs = [int(r) for r in user_input]
	runs_to_use = [run_keys[r] for r in run_idxs]
    elif user_input=='':
	using_average = True
	runs_to_use = copy.copy(run_keys)
    
    if using_average is False:
	CONDS[condkey] = D[selected_condition][selected_run]
    else:
        CONDS[condkey] = dict()
	sample = D[selected_condition][runs_to_use[0]]['phase_map']
	combined_phase = np.zeros((sample.shape[0], sample.shape[1], len(runs_to_use)))
        combined_ratio = np.zeros(combined_phase.shape)
	for ridx,curr_runkey in enumerate(runs_to_use):
	    combined_phase[:,:,ridx] = D[selected_condition][curr_runkey]['phase_map']
	    combined_ratio[:,:,ridx] = D[selected_condition][curr_runkey]['ratio_map']
	combined_phase_x = np.sum(np.cos(combined_phase), 2)
	combined_phase_y = np.sum(np.sin(combined_phase), 2)
	CONDS[condkey]['phase'] = np.arctan2(combined_phase_y, combined_phase_x)
	CONDS[condkey]['ratio'] = np.mean(combined_ratio, 2)
	CONDS[condkey]['averaging'] = True
	CONDS[condkey]['runs_used'] = runs_to_use

#
#print "Select session for AZIMUTH maps (RIGHT):"
#for cond_idx, cond_fn in enumerate(condition_keys):
#    print cond_idx, cond_fn
#user_input=raw_input("\nChoose a session [0,1...]:\n")
#selected_right_condition = condition_keys[int(user_input)]
#
#run_keys = D[selected_right_condition].keys()
#run_keys = [r for r in run_keys if 'Right' in r and str(reduce_factor) in r]
#for run_idx, run_fn in enumerate(run_keys):
#    print run_idx, run_fn
#user_input=raw_input("\nChoose RIGHT run [0,1...]:\n")
#
#if user_input=='':
#    # average all found runs
#    if len(run_keys) > 1:
#        CONDS['right'] = dict()
#        sample = D[selected_right_condition][run_keys[0]]['phase_map']
#        combined_right = np.zeros((sample.shape[0], sample.shape[1], len(run_keys)))
#        combined_ratio = np.zeros((sample.shape[0], sample.shape[1], len(run_keys)))
#        for ridx,run_key in enumerate(run_keys):
#            combined_right[:,:,ridx] = D[selected_right_condition][run_key]['phase_map']
#            combined_ratio[:,:,ridx] = D[selected_right_condition][run_key]['ratio_map']
#        combined_right_x = np.sum(np.cos(combined_right), 2)
#        combined_right_y = np.sum(np.sin(combined_right), 2)
#        CONDS['right']['phase'] = np.arctan2(combined_right_y, combined_right_x)
#        CONDS['right']['ratio'] = np.mean(combined_ratio, 2)    
#        CONDS['right']['averaging'] = True
#    else:
#        CONDS['right'] = D[selected_right_condition][run_keys[0]]
#else:
#    selected_right_run = run_keys[int(user_input)]
#    CONDS['right'] = D[selected_right_condition][selected_right_run]
#
#
## EL = dict()
#
#print "Select session for ELEVATION maps (TOP):"
#for cond_idx, cond_fn in enumerate(condition_keys):
#    print cond_idx, cond_fn
#user_input=raw_input("\nChoose a session [0,1...]:\n")
#
#selected_top_condition = condition_keys[int(user_input)]
#
#run_keys = D[selected_top_condition].keys()
#run_keys = [r for r in run_keys if 'Top' in r and str(reduce_factor) in r]
#for run_idx, run_fn in enumerate(run_keys):
#    print run_idx, run_fn
#user_input=raw_input("\nChoose TOP run [0,1...]:\n")
#
#if user_input=='':
#    # average all found runs
#    if len(run_keys) > 1:
#        CONDS['top'] = dict()
#        sample = D[selected_top_condition][run_keys[0]]['phase_map']
#        combined_top = np.zeros((sample.shape[0], sample.shape[1], len(run_keys)))
#        combined_ratio = np.zeros((sample.shape[0], sample.shape[1], len(run_keys)))
#        for ridx,run_key in enumerate(run_keys):
#            combined_top[:,:,ridx] = D[selected_top_condition][run_key]['phase_map']
#            combined_ratio[:,:,ridx] = D[selected_top_condition][run_key]['ratio_map']
#        combined_top_x = np.sum(np.cos(combined_top), 2)
#        combined_top_y = np.sum(np.sin(combined_top), 2)
#        CONDS['top']['phase'] = np.arctan2(combined_top_y, combined_top_x)
#        CONDS['top']['ratio'] = np.mean(combined_ratio, 2)
#        CONDS['top']['averaging'] = True
#    else:
#        CONDS['top'] = D[selected_top_condition][run_keys[0]]
#else:
#    selected_top_run = run_keys[int(user_input)]
#    CONDS['top'] = D[selected_top_condition][selected_top_run]
#
#print "Select session for ELEVATION maps (BOTTOM):"
#for cond_idx, cond_fn in enumerate(condition_keys):
#    print cond_idx, cond_fn
#user_input=raw_input("\nChoose a session [0,1...]:\n")
#selected_bottom_condition = condition_keys[int(user_input)]
#
#run_keys = D[selected_bottom_condition].keys()
#run_keys = [r for r in run_keys if 'Bottom' in r and str(reduce_factor) in r]
#for run_idx, run_fn in enumerate(run_keys):
#    print run_idx, run_fn
#user_input=raw_input("\nChoose BOTTOM run [0,1...]:\n")
#
#if user_input=='':
#    # average all found runs
#    if len(run_keys) > 1:
#        CONDS['bottom'] = dict()
#        sample = D[selected_bottom_condition][run_keys[0]]['phase_map']
#        combined_bottom = np.zeros((sample.shape[0], sample.shape[1], len(run_keys)))
#        combined_ratio = np.zeros((sample.shape[0], sample.shape[1], len(run_keys)))
#        for ridx,run_key in enumerate(run_keys):
#            combined_bottom[:,:,ridx] = D[selected_bottom_condition][run_key]['phase_map']
#            combined_ratio[:,:,ridx] = D[selected_bottom_condition][run_key]['ratio_map']
#        combined_bottom_x = np.sum(np.cos(combined_bottom), 2)
#        combined_bottom_y = np.sum(np.sin(combined_bottom), 2)
#        CONDS['bottom']['phase'] = np.arctan2(combined_bottom_y, combined_bottom_x)
#        CONDS['bottom']['ratio'] = np.mean(combined_ratio, 2)
#        CONDS['bottom']['averaging'] = True
#    else:
#        CONDS['bottom'] = D[selected_bottom_condition][run_keys[0]]
#else:
#    selected_bottom_run = run_keys[int(user_input)]
#    CONDS['bottom'] = D[selected_bottom_condition][selected_bottom_run]
#
#
# --------------------------------------------------------------------
# Make legends:
# --------------------------------------------------------------------

use_corrected_screen = False
# legend_dir = '/home/juliana/Repositories/retinotopy-mapper/tests/simulation'
winsize = [1920, 1200]
screen_size = [int(i*0.25) for i in winsize]
print screen_size

create_legend = 1 # don't use saved legends, use new corrected

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
if use_corrected_screen is True:
    screen_edge = math.pi - (math.pi*ratio_factor)
else:
    screen_edge = 0
    
if create_legend:        
    H_top_legend = np.zeros((screen_size[1], screen_size[0]))
    # First, set half the screen width (0 to 239 = to 0 to -pi)
    # If CORRECTING for true physical screen, start  after 0 (~1.43):
    nspaces_start = np.linspace(-1*screen_edge, -1*math.pi, screen_size[1]/2)
    for i in range(screen_size[0]):
        H_top_legend[0:screen_size[1]/2, i] = nspaces_start
    # Then, set right side of screen (240 to end = to pi to 0)
    nspaces_end = np.linspace(1*math.pi, screen_edge, screen_size[1]/2)
    for i in range(screen_size[0]):
        H_top_legend[screen_size[1]/2:, i] = nspaces_end
else:
    legend_name = 'H-Down_legend.tif'
    H_top_legend = imread(os.path.join(legend_dir, legend_name))

if create_legend:
    H_bottom_legend = np.zeros((screen_size[1], screen_size[0]))
    # First, set half the screen width (0 to 239 = to 0 to -pi)
    # If CORRECTING for true physical screen, start  after 0 (~1.43):
    nspaces_start = np.linspace(screen_edge, 1*math.pi, screen_size[1]/2)
    for i in range(screen_size[0]):
        H_bottom_legend[0:screen_size[1]/2, i] = nspaces_start
    # Then, set right side of screen (240 to end = to pi to 0)
    nspaces_end = np.linspace(-1*math.pi, -1*screen_edge, screen_size[1]/2)
    for i in range(screen_size[0]):
        H_bottom_legend[screen_size[1]/2:, i] = nspaces_end
else:
    legend_name = 'H-Up_legend.tif'
    H_bottom_legend = imread(os.path.join(legend_dir, legend_name))
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# GET CONDS:
# -------------------------------------------------------------------------
maps = dict()
ratios = dict()
legends = dict()
for condkey in CONDS.keys():
    if 'averaging' in CONDS[condkey].keys():
        maps[condkey] = CONDS[condkey]['phase']
        ratios[condkey] = CONDS[condkey]['ratio']
    else:
        maps[condkey] = np.angle(CONDS[condkey]['ft'])
        ratios[condkey] = CONDS[condkey]['ratio_map']

    if 'left' in condkey:
        legends[condkey] = V_left_legend
    elif 'right' in condkey:
        legends[condkey] = V_right_legend
    elif 'top' in condkey:
        legends[condkey] = H_top_legend
    else:
        legends[condkey] = H_bottom_legend

# Fix legends:
for condkey in CONDS.keys():
    if 'averaging' in CONDS[condkey].keys():
	tmp_leg = np.dstack((legends[condkey], legends[condkey]))
	tmp_leg_x = np.sum(np.cos(tmp_leg), 2)
	tmp_leg_y = np.sum(np.sin(tmp_leg), 2)
	legends[condkey] = np.arctan2(tmp_leg_y, tmp_leg_x)
   
tmaps = dict()
nconds = len(maps.keys())
threshold = float(options.threshold)
alpha_val = float(options.alpha_val)
plt.figure()
for cidx,condkey in enumerate(maps.keys()):
    tmaps[condkey] = copy.copy(maps[condkey])
    tmaps[condkey][np.where(ratios[condkey] < threshold)] = np.nan

    plt.subplot(2,nconds,cidx+1)
    plt.title(condkey)
    plt.imshow(surface, cmap='gray')
    plt.imshow(tmaps[condkey], cmap=colormap, alpha=alpha_val)
    plt.axis('off')
    plt.subplot(2,nconds, cidx+nconds+1)
    plt.imshow(legends[condkey], cmap=colormap, alpha=alpha_val)
    plt.axis('off')

plt.tight_layout()
plt.suptitle([date, subject])

imname = 'thresholded_maps_thresh%0.4f_%s' % (threshold, colormap)
imname = imname.replace('.', 'x')

impath = os.path.join(fig_dir, imname+'.png')
plt.savefig(impath, bbox_inches='tight', pad_inches = 0)

plt.show()

print impath

if using_average is False:
    iminfo_fn = impath.replace('.png', '.txt')
    iminfo = open(iminfo_fn, 'w')
    iminfo.write('LeftCond\t LeftRun\t RightCond\t RightRun\t TopCond\t TopRun\t BottomCond\t BottomRun\t Threshold\t Colormap\n')
    iminfo.write('%s\t %s\t %s\t %s\t %s\t %s\t %s\t %s\t %f\t %s\n' % (selected_left_condition, selected_left_run, selected_right_condition, selected_right_run, selected_top_condition, selected_top_run, selected_bottom_condition, selected_bottom_run, threshold, colormap))
    iminfo.close()

# -------------------------------------------------------
# Get phase maps, shift pi for averaging.  Then, AVERAGE.
# -------------------------------------------------------
vmin_val = -1*math.pi # 0
vmax_val = math.pi

# shift_left_phase = np.angle(left_map)
# shift_right_phase = np.angle(right_map.conjugate())
# shift_left_phase[shift_left_phase<0] += 2*math.pi
# shift_right_phase[shift_right_phase<0] += 2*math.pi
# shift_az_legend = copy.deepcopy(V_left_legend)
# shift_az_legend[shift_az_legend<0] += 2*math.pi
# shift_other_az_legend = copy.deepcopy(V_right_legend)
# shift_other_az_legend[shift_other_az_legend<0] += 2*math.pi

# shift_top_phase = np.angle(top_map) 
# shift_bottom_phase = np.angle(bottom_map.conjugate())
# shift_top_phase[shift_top_phase<0] += 2*math.pi
# shift_bottom_phase[shift_bottom_phase<0] += 2*math.pi
# shift_el_legend = copy.deepcopy(H_top_legend)
# shift_el_legend[shift_el_legend<0] += 2*math.pi
# shift_other_el_legend = copy.deepcopy(H_bottom_legend)
# shift_other_el_legend[shift_other_el_legend<0] += 2*math.pi

#avg_az_phase = (shift_left_phase + shift_right_phase) * 0.5
#avg_el_phase = (shift_top_phase + shift_bottom_phase) * 0.5

#------ average phases correctly ------------
if using_average is False:
    left_phase = np.angle(left_map)
    right_phase = np.angle(right_map.conjugate())
    tmp_az_combined = np.dstack((left_phase, right_phase))
    tmp_az_cos = np.sum(np.cos(tmp_az_combined), 2)
    tmp_az_sin = np.sum(np.sin(tmp_az_combined), 2)
    avg_az_phase = np.arctan2(tmp_az_sin, tmp_az_cos)
    print avg_az_phase.shape

    top_phase = np.angle(top_map)
    bottom_phase = np.angle(bottom_map.conjugate())
    tmp_el_combined = np.dstack((top_phase, bottom_phase))
    tmp_el_x = np.sum(np.cos(tmp_el_combined), 2)
    tmp_el_y = np.sum(np.sin(tmp_el_combined), 2)
    avg_el_phase = np.arctan2(tmp_el_y, tmp_el_x)

    ratio_avg_az = (ratio_left + ratio_right) / 2.
    thresh_avg_az_phase = copy.copy(avg_az_phase)
    thresh_avg_az_phase[np.where(ratio_avg_az < threshold)] = np.nan

    ratio_avg_el = (ratio_top + ratio_bottom) / 2.
    thresh_avg_el_phase = copy.copy(avg_el_phase)
    thresh_avg_el_phase[np.where(ratio_avg_el < threshold)] = np.nan



    plt.figure()

    if show_avg is True:
        plt.subplot(2,2,1)
        plt.imshow(surface, cmap='gray')
        plt.imshow(thresh_avg_az_phase, cmap=colormap, vmin=vmin_val, vmax=vmax_val, alpha=alpha_val)
        plt.axis('off')
        plt.subplot(2,2,3)
        plt.imshow(V_left_legend, cmap=colormap, vmin=vmin_val, vmax=vmax_val, alpha=alpha_val)
        plt.axis('off')

        plt.subplot(2,2,2)
        plt.imshow(surface, cmap='gray')
        plt.imshow(thresh_avg_el_phase, cmap=colormap, vmin=vmin_val, vmax=vmax_val, alpha=alpha_val)
        plt.axis('off')
        plt.subplot(2,2,4)
        plt.imshow(H_top_legend, cmap=colormap, vmin=vmin_val, vmax=vmax_val, alpha=alpha_val)
        plt.axis('off')

        plt.tight_layout()

        avg_figname = 'AVG, thr: %0.4f' % threshold
        plt.suptitle([avg_figname, date, subject])

        imname = 'avg_phases_%s' % colormap

        impath = os.path.join(fig_dir, imname+'.png')
        plt.savefig(impath, bbox_inches='tight', pad_inches = 0)

        plt.show()



    maps = dict()
    cond_names = ['left', 'right', 'top', 'bottom']
    for cond_name in cond_names:
        maps[cond_name] = dict()

    maps['left']['threshold_phase'] = thresh_left_phase
    maps['right']['threshold_phase'] = thresh_right_phase
    maps['top']['threshold_phase'] = thresh_top_phase
    maps['bottom']['threshold_phase'] = thresh_bottom_phase

    maps['left']['legend'] = V_left_legend
    maps['right']['legend'] = V_right_legend
    maps['top']['legend'] = H_top_legend
    maps['bottom']['legend'] = H_bottom_legend


    maps['left']['shift_phase'] = shift_left_phase
    maps['right']['shift_phase'] = shift_right_phase
    maps['top']['shift_phase'] = shift_top_phase
    maps['bottom']['shift_phase'] = shift_bottom_phase

    maps['threshold'] = threshold
    maps['shift_az_legend'] = shift_az_legend
    maps['shift_other_az_legend'] = shift_other_az_legend
    maps['shift_el_legend'] = shift_el_legend
    maps['shift_other_el_legend'] = shift_other_el_legend


    path_to_map_struct = os.path.join(composite_dir, 'maps%i.pkl' % reduce_val)
    with open(path_to_map_struct, 'wb') as wm:
        pkl.dump(maps, wm, protocol=pkl.HIGHEST_PROTOCOL)

# # smooth = True
# # sigma_val = (3,3)
# if smooth is True:
#     left_phase = ndimage.gaussian_filter(left_phase, sigma=sigma_val, order=0)
#     right_phase = ndimage.gaussian_filter(right_phase, sigma=sigma_val, order=0)
    
# vmin_val = 0 #-1*math.pi # 0
# vmax_val = 2*math.pi


# # smooth = True
# # sigma_val = (3,3)
# # smooth = True
# if smooth is True:
#     top_phase = ndimage.gaussian_filter(top_phase, sigma=sigma_val, order=0)
#     bottom_phase = ndimage.gaussian_filter(bottom_phase, sigma=sigma_val, order=0)


# --------------------------------------------------------------------------------------
# GET JUST THE PHASE MAP FOR COREG:
# --------------------------------------------------------------------------------------
# This format saves png/fig without any borders:
# Need this type of data-only image for COREG, for example.

    if get_clean is True:

        # SURFACE 
        # --------------------------------------------------------------------------------------
        fig = plt.imshow(surface, cmap='gray')
        plt.imshow(thresh_avg_az_phase, cmap='hsv', vmin=vmin_val, vmax=vmax_val, alpha=alpha_val)
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        imname = 'overlay_avg_phase_AZ_HSV_SURFACE'

        impath = os.path.join(fig_dir, imname+'.png')
        plt.savefig(impath, bbox_inches='tight', pad_inches = 0)

        plt.show()

        # AZ average 
        # --------------------------------------------------------------------------------------
        fig = plt.imshow(surface, cmap='gray')
        plt.imshow(thresh_avg_el_phase, cmap='hsv', vmin=vmin_val, vmax=vmax_val, alpha=0.5)
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        imname = 'overlay_avg_phase_EL_HSV_PHASE'

        impath = os.path.join(fig_dir, imname+'.png')
        plt.savefig(impath, bbox_inches='tight', pad_inches = 0)

        plt.show()

        if use_left is False:
            # plt.imshow(np.angle(rightmap), cmap=colormap)
            fig = plt.imshow(np.angle(rightmap), cmap='hsv', vmin=vmin_val, vmax=vmax_val)
            plt.axis('off')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)

            imname = 'right_phase_AZ_HSV_PHASE'

            impath = os.path.join(figdir, imname+'.png')
            plt.savefig(impath, bbox_inches='tight', pad_inches = 0)

            plt.show()

            fig = plt.imshow(AZ_legend, cmap='hsv', vmin=vmin_val, vmax=vmax_val)
            plt.axis('off')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)

            imname = 'right_phase_AZ_HSV_PHASE_LEGEND'

            impath = os.path.join(fig_dir, imname+'.png')
            plt.savefig(impath, bbox_inches='tight', pad_inches = 0)

            plt.show()


        fig = plt.imshow(AZ_legend, cmap='hsv', vmin=vmin_val, vmax=vmax_val)
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        imname = 'avg_phase_AZ_HSV_PHASE_LEGEND'

        impath = os.path.join(fig_dir, imname+'.png')
        plt.savefig(impath, bbox_inches='tight', pad_inches = 0)

        plt.show()
        # EL average 
        # --------------------------------------------------------------------------------------

        fig = plt.imshow(el_avg, cmap='hsv', vmin=vmin_val, vmax=vmax_val)
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        imname = 'avg_phase_EL_HSV_PHASE'

        impath = os.path.join(fig_dir, imname+'.png')
        plt.savefig(impath, bbox_inches='tight', pad_inches = 0)

        plt.show()


    I = dict()
    I['az_phase'] = az_avg
    I['vmin'] = vmin_val
    I['vmax'] = vmax_val
    I['az_legend'] = AZ_legend
    I['surface'] = surface

    fext = 'clean_fig_info.pkl'
    fname = os.path.join(fig_dir, fext)
    with open(fname, 'wb') as f:
        # protocol=pkl.HIGHEST_PROTOCOL)
        pkl.dump(I, f, protocol=pkl.HIGHEST_PROTOCOL)


    # mat_fn = 'temp2sample'+'.pkl'
    # # scipy.io.savemat(os.path.join(source_dir, condition, tif_fn), mdict=pydict)

    # import scipy.io
    # scipy.io.savemat(os.path.join(out_path, mat_fn), mdict=T)
    # print os.path.join(out_path, 'mw_data', mat_fn)
