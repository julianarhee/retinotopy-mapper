#!/usr/bin/env python2
# coding: utf-8

# FROM plot_absolute_maps_GCaMP.ipnb (JNB)
import json
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

def get_thresholded_maps(phase_map, ratio_map, threshold):
    phase_mask = copy.deepcopy(phase_map)
    phase_mask[ratio_map < threshold] = np.nan
    phase_mask = np.ma.array(phase_mask)
    return phase_mask


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

import matplotlib.cm as cm
import numpy.linalg as la
import scipy.ndimage as ndimage
#import json

from json import dumps, loads, JSONEncoder, JSONDecoder
import pickle

class PythonObjectEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (list, dict, str, unicode, int, float, bool, type(None))):
            return JSONEncoder.default(self, obj)
        return {'_python_object': pickle.dumps(obj)}

def as_python_object(dct):
    if '_python_object' in dct:
        return pickle.loads(str(dct['_python_object']))
    return dct


parser = optparse.OptionParser()
parser.add_option('--headless', action="store_true", dest="headless", default=False, help="run in headless mode, no figs")
parser.add_option('--reduce', action="store", dest="reduce_val", default="2", help="block_reduce value")
parser.add_option('--path', action="store", dest="path", default="", help="path to data directory")
# parser.add_option('-t', '--thresh', action="store", dest="threshold", default=0.5, help="cutoff threshold value")
parser.add_option('-r', '--run', action="store", dest="run", default=1, help="cutoff threshold value")
parser.add_option('--append', action="store", dest="append", default="", help="appended label for analysis structs")
parser.add_option('-M', '--map', action="store", dest="maptype", type="choice", choices=['absolute', 'combo'], default='absolute', help="Method for creating phase maps [absolute | combo]")
parser.add_option('-C', '--cmap', action="store", dest="cmap", default='spectral', help="colormap for summary figures [default: spectral]")
parser.add_option('--smooth', action="store_true", dest="smooth", default=False, help="smooth? (default sig = 2)")
parser.add_option('--sigma', action="store", dest="sigma_val", default=2, help="sigma for gaussian smoothing")

# parser.add_option('--contour', action="store_true", dest="contour", default=False, help="Show contour lines for phase map")
# parser.add_option('--power', action='store_true', dest='use_power', default=False, help="Use power or just magnitude?")


parser.add_option('--noclean', action='store_false', dest='get_clean', default=True, help="Save borderless, clean maps for COREG")

parser.add_option('--threshold', action="store", dest="threshold", default=0.01, help="Threshold (max of ratio map)")
parser.add_option('--alpha', action="store", dest="alpha_val", default=0.5, help="Alpha value for overlays")

parser.add_option('--short-axis', action="store_false", dest="use_long_axis", default=True, help="Used short-axis instead of long?")
parser.add_option('--combo', action="store_true", dest="show_combo", default=False, help="Show combined maps or only save single runs?")
parser.add_option('--new', action="store_true", dest="create_new", default=False, help="Create new map struct or no?")

parser.add_option('--exclude', action="store", dest="exclude", default='avg', help='string to exclude from append variable')
parser.add_option('--correct-monitor', action="store", dest="correct_travel_distance", default=True, help='correct pi range for travel distance if bar off screen')
parser.add_option('--vertical', action="store_true", dest="vertical_only", default=False, help='only look at vertical conditions')



(options, args) = parser.parse_args()
vertical_only = options.vertical_only
maptype = options.maptype
exclude = options.exclude
create_new = options.create_new
show_combo = options.show_combo
get_clean = options.get_clean

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

threshold = float(options.threshold)
outdir = options.path
run_num = options.run

append = options.append


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

animaldir = os.path.split(outdir)[0]
print "Animal dir: ", animaldir
animal = os.path.split(animaldir)[1]
session = date
#print "animal dir: ", animaldir
#folders = os.listdir(sessiondir)
folders = os.listdir(animaldir)
figpath = [f for f in folders if f == 'surface']
# figpath = [f for f in folders if f == 'figures'][0]
# print "EXPT: ", exptdir
# print "SESSION: ", sessiondir
print "path to surface: ", figpath

if figpath:
    # figdir = figpath[0]
    figpath=figpath[0]
    
    #tmp_ims = os.listdir(os.path.join(sessiondir, figpath))
    #surface_words = ['surface', 'GREEN', 'green', 'Surface', 'Surf']
    #ims = [i for i in tmp_ims if any([word in i for word in surface_words])]
    #ims = [i for i in ims if '_' in i]
    tmp_ims = os.listdir(os.path.join(animaldir, figpath))
    ims = [i for i in tmp_ims if session in i]
    print ims
    if ims:
        #impath = os.path.join(sessiondir, figpath, ims[0])
        impath = os.path.join(animaldir, figpath, ims[0])
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

# folders = os.listdir(outdir)
# figpath = [f for f in folders if f == 'surface']
# if not figpath:
#     figpath = [f for f in folders if f == 'figures']
# print "path to surface: ", figpath
# 
# if figpath:
#     # figdir = figpath[0]
#     figpath=figpath[0]
#     tmp_ims = os.listdir(os.path.join(outdir, figpath))
#     surface_words = ['surface', 'GREEN', 'green', 'Surface', 'Surf']
#     ims = [i for i in tmp_ims if any([word in i for word in surface_words])]
#     ims = [i for i in ims if date in i]
#     # print ims
#     if ims:
#         impath = os.path.join(outdir, figpath, ims[0])
#         print os.path.splitext(impath)[1]
#         if os.path.splitext(impath)[1] == '.tif':
#             tiff = TIFF.open(impath, mode='r')
#             surface = tiff.read_image().astype('float')
#             tiff.close()
#         else:
#             image = Image.open(impath) #.convert('L')
#             surface = np.asarray(image)
#     else:
#         surface = np.zeros([200,300])
#     print surface.shape
# else: # NO BLOOD VESSEL IMAGE...
#     surface = np.zeros([200,300])
#     print "No blood vessel image found. Using empty."
# 
if reduceit:
    surface = block_reduce(surface, reduce_factor, func=np.mean)
    print "Reduced image to: ", surface.shape


# --------------------------------------------------------------------
# GET DATA STRUCT FILES:
# --------------------------------------------------------------------

#append = options.append

struct_fns = os.listdir(outdir)
struct_fns = [f for f in struct_fns if os.path.splitext(f)[1] == '.pkl' and 'r'+str(reduce_val) in f]

if len(struct_fns) > 1 and create_new is False:
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
    uname = raw_input("Enter new stuct name: ")
    composite_struct_fn = '{date}_{animal}_r{reduceval}_{custom}_struct.pkl'.format(date=date, animal=subject, reduceval=reduce_val, custom=uname)
    #composite_struct_fn = '{date}_{animal}_r{reduceval}_{custom}_struct.json'.format(date=date, animal=subject, reduceval=reduce_val, custom=uname)

    print "New struct name is: %s" % composite_struct_fn

    D = dict()
    for condition in conditions:
        condition_dir = os.path.join(outdir, condition, 'structs')
	if not os.path.exists(condition_dir):
	    continue
        condition_structs = os.listdir(condition_dir)
        condition_structs = [f for f in condition_structs if '.pkl' in f and 'fft' in f and append in f and exclude not in f]
        print "Found condition structs: ", condition_structs

        D[condition] = dict()
        for condition_struct in condition_structs:
            curr_condition_struct = os.path.join(condition_dir, condition_struct)
            curr_cond_key = condition_struct.split('Target_fft_')[1].split('_.pkl')[0]
            with open(curr_condition_struct, 'rb') as f:
                D[condition][curr_cond_key] = pkl.load(f)

    path_to_composite_struct_fn = os.path.join(outdir, composite_struct_fn)
    with open(path_to_composite_struct_fn, 'wb') as wf:
        pkl.dump(D, wf, protocol=pkl.HIGHEST_PROTOCOL)
    #dlist = [{'condition': condition, 'values': D[condition]} for condition in D.keys()]
#     with open(path_to_composite_struct_fn, 'w') as f:
# 	json.dump(dlist, f)
    print "New struct name is: %s" % path_to_composite_struct_fn



CONDS = dict()
condition_keys = D.keys()
print "Condition keys in D-dict: ", condition_keys
if vertical_only:
    condition_types = ['Left', 'Right']
else:
    condition_types = ['Left', 'Right', 'Top', 'Bottom']

for condition_type in condition_types:
    condkey = condition_type.lower()
    CONDS[condkey] = dict()
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
        runs_to_use = selected_run
    elif len(user_input)>1:
	using_average = True
	run_idxs = [int(r) for r in user_input]
	runs_to_use = [run_keys[r] for r in run_idxs]
    #elif user_input=='':
    else:
	using_average = True
	runs_to_use = copy.copy(run_keys)
    
    print "Runs to use: ", runs_to_use
 
    if using_average is False:
	#CONDS[condkey] = D[selected_condition][selected_run]
        CONDS[condkey]['phase'] = D[selected_condition][selected_run]['phase_map']
        CONDS[condkey]['averaging'] = False
        CONDS[condkey]['ratio'] = D[selected_condition][selected_run]['ratio_map']
        CONDS[condkey]['runs_to_use'] = selected_run
        CONDS[condkey]['condition'] = selected_condition
    else:

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
	CONDS[condkey]['runs_to_use'] = runs_to_use
        CONDS[condkey]['condition'] = selected_condition



# --------------------------------------------------------------------
# Make legends:
# --------------------------------------------------------------------

use_corrected_screen = options.correct_travel_distance #False
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
# CREATE main mapinfo struct:
# -------------------------------------------------------------------------

mapinfo = dict()
for condkey in CONDS.keys():
    mapinfo[condkey] = dict()
    #if 'averaging' in CONDS[condkey].keys():
    mapinfo[condkey]['phase'] = CONDS[condkey]['phase']
    mapinfo[condkey]['ratio'] = CONDS[condkey]['ratio']

   # else:
       # mapinfo[condkey]['phase'] = np.angle(CONDS[condkey]['ft'])
       # mapinfo[condkey]['ratio'] = CONDS[condkey]['ratio_map']

    if 'left' in condkey:
        mapinfo[condkey]['legend'] = V_left_legend
    elif 'right' in condkey:
        mapinfo[condkey]['legend'] = V_right_legend
    elif 'top' in condkey:
        mapinfo[condkey]['legend'] = H_top_legend
    else:
        mapinfo[condkey]['legend'] = H_bottom_legend

# Fix legends:
for condkey in CONDS.keys():
    if CONDS[condkey]['averaging']:
	tmp_leg = np.dstack((mapinfo[condkey]['legend'], mapinfo[condkey]['legend']))
	tmp_leg_x = np.sum(np.cos(tmp_leg), 2)
	tmp_leg_y = np.sum(np.sin(tmp_leg), 2)
	mapinfo[condkey]['legend'] = np.arctan2(tmp_leg_y, tmp_leg_x)



# -----------------------------------------------------------------------
# PLOT each selected condition:
# ----------------------------------------------------------------------   
nconds = len(mapinfo.keys())
threshold = float(options.threshold)
alpha_val = float(options.alpha_val)
plt.figure()
for cidx,condkey in enumerate(mapinfo.keys()):
    mapinfo[condkey]['threshold'] = threshold
    mapinfo[condkey]['alpha'] = alpha_val
    #tmaps[condkey] = copy.copy(maps[condkey])
    #tmaps[condkey][np.where(ratios[condkey] < threshold)] = np.nan
    mapinfo[condkey]['thrphase'] = get_thresholded_maps(mapinfo[condkey]['phase'], mapinfo[condkey]['ratio'], threshold)

    # add run source info to dict:
    mapinfo[condkey]['averaging'] = CONDS[condkey]['averaging']
    mapinfo[condkey]['runs'] = CONDS[condkey]['runs_to_use']
    mapinfo[condkey]['condition'] = CONDS[condkey]['condition']

    plt.subplot(2,nconds,cidx+1)
    plt.title(condkey)
    plt.imshow(surface, cmap='gray')
    plt.imshow(mapinfo[condkey]['thrphase'], cmap=colormap, alpha=alpha_val)
    plt.axis('off')
    plt.subplot(2,nconds, cidx+nconds+1)
    plt.imshow(mapinfo[condkey]['legend'], cmap=colormap, alpha=alpha_val)
    plt.axis('off')

plt.tight_layout()
plt.suptitle([date, subject])

imname = 'thresholded_maps_thresh%0.4f_%s' % (threshold, colormap)
imname = imname.replace('.', 'x')

impath = os.path.join(fig_dir, imname+'.png')
plt.savefig(impath, bbox_inches='tight', pad_inches = 0)

plt.show()

print impath

# Save map info:
#iminfo_fn = impath.replace('.png', '.pkl')
#iminfo = open(iminfo_fn, 'w')
#iminfo.write('LeftCond\t LeftRun\t RightCond\t RightRun\t TopCond\t TopRun\t BottomCond\t BottomRun\t Threshold\t Colormap\n')
#iminfo.write('%s\t %s\t %s\t %s\t %s\t %s\t %s\t %s\t %f\t %s\n' % (selected_left_condition, selected_left_run, selected_right_condition, selected_right_run, selected_top_condition, selected_top_run, selected_bottom_condition, selected_bottom_run, threshold, colormap))
#iminfo.close()

iminfo_fn = impath.replace('.png', '.json')

with open(os.path.join(composite_dir, iminfo_fn), 'w') as f:
    minfo = dumps(mapinfo, cls=PythonObjectEncoder)
    
iminfo_fn = impath.replace('.json', '.pkl')
with open(os.path.join(composite_dir, iminfo_fn), 'wb') as f:
    pkl.dump(mapinfo, f, protocol=pkl.HIGHEST_PROTOCOL)
    
condpath = os.path.join(fig_dir, 'CONDS.pkl')
with open(os.path.join(condpath), 'wb') as f:
    pkl.dump(CONDS, f, protocol=pkl.HIGHEST_PROTOCOL)
    
# To open:
# loads(minfo, object_hook=as_python_object)



# -------------------------------------------------------
# Get phase maps of SINGLE runs, then, COMBINE runs (psuedo-average).
# -------------------------------------------------------
vmin_val = -1*math.pi # 0
vmax_val = math.pi

#------ average phases correctly ------------
#if using_average is False:
#     left_phase = np.angle(left_map)
#     thresh_left_phase = copy.copy(left_phase)
#     thresh_left_phase[np.where(ratio_left < threshold)] = np.nan
# 
#     right_phase = np.angle(right_map.conjugate())
#     thresh_right_phase = copy.copy(right_phase)
#     thresh_right_phase[np.where(ratio_right < threshold)] = np.nan
#     
#     thresh_left_phase = get_thresholded_maps(mapinfo['left']['phase'], mapinfo['left']['ratio'], threshold)
#     thresh_right_phase = get_thresholded_maps(mapinfo['right']['phase'], mapinfo['right']['ratio'], threshold)
#     
# 

# ---------------------------------------------------------------------------
# GET COMBO maps:
# ----------------------------------------------------------------------------

combomaps = dict()
combomaps['comboAZ'] = dict()
combomaps['comboEL'] = dict()
combomaps['absoluteAZ'] = dict()
combomaps['absoluteEL'] = dict()

left_phase = mapinfo['left']['phase']    
if mapinfo['right']['averaging']:
    conj_right_phase = np.angle(D[mapinfo['right']['condition']][mapinfo['right']['runs'][0]]['ft'].conjugate())
else:
    conj_right_phase = np.angle(D[mapinfo['right']['condition']][mapinfo['right']['runs']]['ft'].conjugate())

tmp_az_combined = np.dstack((left_phase, conj_right_phase))
tmp_az_cos = np.sum(np.cos(tmp_az_combined), 2)
tmp_az_sin = np.sum(np.sin(tmp_az_combined), 2)
combomaps['comboAZ']['phase'] = np.arctan2(tmp_az_sin, tmp_az_cos)
#print avg_az_phase.shape

#     top_phase = np.angle(top_map)
#     bottom_phase = np.angle(bottom_map.conjugate())
top_phase = mapinfo['top']['phase']
if mapinfo['bottom']['averaging']:
    conj_bottom_phase = np.angle(D[mapinfo['bottom']['condition']][mapinfo['bottom']['runs'][0]]['ft'].conjugate())
else:
    conj_bottom_phase = np.angle(D[mapinfo['bottom']['condition']][mapinfo['bottom']['runs']]['ft'].conjugate())

#     thresh_top_phase = get_thresholded_maps(mapinfo['top']['phase'], mapinfo['top']['ratio'], threshold)
#     thresh_bottom_phase = get_thresholded_maps(mapinfo['bottom']['phase'], mapinfo['bottom']['ratio'], threshold)
# 
tmp_el_combined = np.dstack((top_phase, conj_bottom_phase))
tmp_el_x = np.sum(np.cos(tmp_el_combined), 2)
tmp_el_y = np.sum(np.sin(tmp_el_combined), 2)
combomaps['comboEL']['phase'] = np.arctan2(tmp_el_y, tmp_el_x)

combomaps['comboAZ']['ratio'] = (mapinfo['left']['ratio'] + mapinfo['right']['ratio'])/2.
combomaps['comboAZ']['thrphase'] = get_thresholded_maps(combomaps['comboAZ']['phase'], combomaps['comboAZ']['ratio'], threshold)
combomaps['comboAZ']['legend'] = V_left_legend

#     thresh_avg_az_phase = copy.copy(avg_az_phase)
#     thresh_avg_az_phase[np.where(ratio_avg_az < threshold)] = np.nan
# 
combomaps['comboEL']['ratio'] = (mapinfo['top']['ratio'] + mapinfo['bottom']['ratio']) / 2.
combomaps['comboEL']['thrphase'] = get_thresholded_maps(combomaps['comboEL']['phase'], combomaps['comboEL']['ratio'], threshold)
combomaps['comboEL']['legend'] = H_top_legend

#     thresh_avg_el_phase = copy.copy(avg_el_phase)
#     thresh_avg_el_phase[np.where(ratio_avg_el < threshold)] = np.nan
# 


plt.figure()

if show_combo is True:
    plt.subplot(2,2,1)
    plt.imshow(surface, cmap='gray')
    plt.imshow(combomaps['comboAZ']['phase'], cmap=colormap, vmin=vmin_val, vmax=vmax_val, alpha=alpha_val)
    plt.axis('off')
    plt.subplot(2,2,3)
    plt.imshow(combomaps['comboAZ']['legend'], cmap=colormap, vmin=vmin_val, vmax=vmax_val, alpha=alpha_val)
    plt.axis('off')

    plt.subplot(2,2,2)
    plt.imshow(surface, cmap='gray')
    plt.imshow(combomaps['comboEL']['phase'], cmap=colormap, vmin=vmin_val, vmax=vmax_val, alpha=alpha_val)
    plt.axis('off')
    plt.subplot(2,2,4)
    plt.imshow(combomaps['comboEL']['legend'], cmap=colormap, vmin=vmin_val, vmax=vmax_val, alpha=alpha_val)
    plt.axis('off')

    plt.tight_layout()

    avg_figname = 'combined, thr: %0.4f' % threshold
    plt.suptitle([avg_figname, date, subject])

    imname = 'combophases_%s' % colormap

    impath = os.path.join(fig_dir, imname+'.png')
    plt.savefig(impath, bbox_inches='tight', pad_inches = 0)

    plt.show()


# 
# maps = dict()
# cond_names = ['left', 'right', 'top', 'bottom']
# for cond_name in cond_names:
#     maps[cond_name] = dict()
# 
# maps['left']['threshold_phase'] = thresh_left_phase
# maps['right']['threshold_phase'] = thresh_right_phase
# maps['top']['threshold_phase'] = thresh_top_phase
# maps['bottom']['threshold_phase'] = thresh_bottom_phase
# 
# maps['left']['legend'] = V_left_legend
# maps['right']['legend'] = V_right_legend
# maps['top']['legend'] = H_top_legend
# maps['bottom']['legend'] = H_bottom_legend
# 
# maps['comboAZ']['threshold_phase'] = thresh_avg_az_phase
# maps['comboAZ']['legend'] = V_left_legend
# maps['comboEL']['threshold_phase'] = thresh_avg_el_phase
# maps['comboEL']['legend'] = H_top_legend
# 
# path_to_map_struct = os.path.join(composite_dir, 'maps%i.pkl' % reduce_val)
# with open(path_to_map_struct, 'wb') as wm:
#     pkl.dump(maps, wm, protocol=pkl.HIGHEST_PROTOCOL)
# 
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
# Get absolute maps:
# --------------------------------------------------------------------------------------

#AZphases = np.dstack((mapinfo['left']['phase'], mapinfo['right']['phase']))

def get_absolute_maps(AZphases):
#AZphases = np.dstack((mapinfo['left']['phase'], mapinfo['right']['phase']))
    AZhalf = np.true_divide(AZphases[:,:,0], 2)
    AZhalf2 = np.true_divide(AZphases[:,:,1], 2)

    # Initial absolute phase map:
    abs_phase_map = AZhalf - AZhalf2

    # correct values where numerical problems are expected to arise(?):
    region2fix1 = np.logical_and(AZphases[:,:,0] <= -np.true_divide(np.pi,2), AZphases[:,:,1] <= -np.true_divide(np.pi,2))
    region2fix2 = np.logical_and(AZphases[:,:,0] >= np.true_divide(np.pi,2), AZphases[:,:,1]>=np.true_divide(np.pi,2))

    region2fix = np.logical_or(region2fix1, region2fix2)

    tmp_phase = np.copy(abs_phase_map)

    logical_idxs1 = np.logical_and(region2fix, abs_phase_map>0)
    tmp_phase[logical_idxs1] = abs_phase_map[logical_idxs1] - (np.pi)

    logical_idxs2 = np.logical_and(region2fix, abs_phase_map<0)
    tmp_phase[logical_idxs2] = abs_phase_map[logical_idxs2] + (np.pi)

    abs_phase_map = tmp_phase

    display_phase_map = np.copy(abs_phase_map)
    display_phase_map[abs_phase_map<0] = -abs_phase_map[abs_phase_map<0]
    display_phase_map[abs_phase_map>0] = (2*np.pi) - abs_phase_map[abs_phase_map>0]

    # Get delay map and correct:
    delay_map = AZhalf + AZhalf2
    delay_map[region2fix1] = np.pi + delay_map[region2fix1]
    delay_map[region2fix2] = delay_map[region2fix2] - np.pi


    return display_phase_map, delay_map

AZphases = np.dstack((mapinfo['left']['phase'], mapinfo['right']['phase']))
az_abs_phase, az_delay = get_absolute_maps(AZphases)
AZlegends = np.dstack((mapinfo['left']['legend'], mapinfo['right']['legend']))
az_legend, az_legend_delay = get_absolute_maps(AZlegends)

ELphases = np.dstack((mapinfo['top']['phase'], mapinfo['bottom']['phase']))
el_abs_phase, el_delay = get_absolute_maps(ELphases)
ELlegends = np.dstack((mapinfo['top']['legend'], mapinfo['bottom']['legend']))
el_legend, el_legend_delay = get_absolute_maps(ELlegends)

combomaps['absoluteAZ']['phase'] = az_abs_phase
combomaps['absoluteAZ']['legend'] =  az_legend
combomaps['absoluteAZ']['delay'] = az_delay


combomaps['absoluteEL']['phase'] = el_abs_phase
combomaps['absoluteEL']['legend'] =  el_legend
combomaps['absoluteEL']['delay'] = el_delay


plt.figure()
plt.subplot(2,2,1)
plt.imshow(az_abs_phase, cmap=colormap)
plt.title('absolute AZ')
plt.colorbar()
plt.subplot(2,2,2)
plt.imshow(az_delay, cmap=colormap)
plt.title('delay')
plt.subplot(2,2,3)
plt.imshow(az_legend, cmap=colormap)
plt.colorbar()
plt.subplot(2,2,4)
plt.imshow(az_legend_delay, cmap=colormap)
plt.axis('off')

imname = 'absolute_delay_maps'
impath = os.path.join(fig_dir, imname+'.png')
plt.savefig(impath, bbox_inches='tight', pad_inches = 0)

plt.show()

fig = plt.imshow(surface, cmap='gray')
#plt.subplot(1,2,1)
plt.imshow(surface, cmap='gray')
#plt.imshow(az_abs_phase, cmap=colormap, alpha=alpha_val)
#plt.title('Absolte AZ phase overlay')
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

imname = 'absoluteAZ_overlay_surface_surface'
impath = os.path.join(fig_dir, imname+'.png')
plt.savefig(impath, bbox_inches='tight', pad_inches = 0)
surface.shape

fig = plt.imshow(surface, cmap='gray')
#plt.subplot(1,2,1)
#plt.imshow(surface, cmap='gray')
plt.imshow(az_abs_phase, cmap=colormap, alpha=alpha_val)
#plt.title('Absolte AZ phase overlay')
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

imname = 'absoluteAZ_overlay_surface'
impath = os.path.join(fig_dir, imname+'.png')
plt.savefig(impath, bbox_inches='tight', pad_inches = 0)

plt.show()

fig = plt.imshow(az_legend, cmap=colormap)
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.axis('off')

imname = 'absoluteAZ_overlay_surface_legend'
impath = os.path.join(fig_dir, imname+'.png')
plt.savefig(impath, bbox_inches='tight', pad_inches = 0)

plt.show()


# --------------------------------------------------------------------------------------
# GET JUST THE PHASE MAP FOR COREG:
# --------------------------------------------------------------------------------------
# This format saves png/fig without any borders:
# Need this type of data-only image for COREG, for example.

if maptype=='absolute':
    AZ_phase = combomaps['absoluteAZ']['phase']
    AZ_legend = combomaps['absoluteAZ']['legend']
    EL_phase = combomaps['absoluteEL']['phase']
    EL_legend = combomaps['absoluteEL']['legend']
    vmin = 0
    vmax_val = 2*math.pi
elif maptype=='combo':
    AZ_phase = combomaps['comboAZ']['thrphase']
    AZ_legend = combomaps['comboAZ']['legend']
    EL_phase = combomaps['comboEL']['thrphase']
    EL_legend = combomaps['comboEL']['legend']


if get_clean is True:

    # SURFACE w/ AVG_EL:
    # --------------------------------------------------------------------------------------
    fig = plt.imshow(surface, cmap='gray')
    plt.imshow(AZ_phase, cmap='hsv', vmin=vmin_val, vmax=vmax_val, alpha=alpha_val)
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    imname = 'overlay_AZphaseHSV_thresh%0.2f' % threshold

    impath = os.path.join(fig_dir, imname+'.png')
    plt.savefig(impath, bbox_inches='tight', pad_inches = 0)

    plt.show()

    # EL average 
    # --------------------------------------------------------------------------------------
    fig = plt.imshow(surface, cmap='gray')
    plt.imshow(EL_phase, cmap='hsv', vmin=vmin_val, vmax=vmax_val, alpha=0.5)
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    imname = 'overlay_ELphaseHSV_threshold%0.2f' % threshold

    impath = os.path.join(fig_dir, imname+'.png')
    plt.savefig(impath, bbox_inches='tight', pad_inches = 0)

    plt.show()

    fig = plt.imshow(AZ_legend, cmap='hsv', vmin=vmin_val, vmax=vmax_val)
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    imname = 'overlay_AZphaseHSV_legend'

    impath = os.path.join(fig_dir, imname+'.png')
    plt.savefig(impath, bbox_inches='tight', pad_inches = 0)

    plt.show()
    
    fig = plt.imshow(EL_legend, cmap='hsv', vmin=vmin_val, vmax=vmax_val)
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    imname = 'overlay_ELphaseHSV_legend'

    impath = os.path.join(fig_dir, imname+'.png')
    plt.savefig(impath, bbox_inches='tight', pad_inches = 0)

    plt.show()


I = dict()
I['azimuth'] = AZ_phase
I['elevation'] = EL_phase
I['vmin'] = vmin_val
I['vmax'] = vmax_val
I['az_legend'] = AZ_legend
I['el_legend'] = EL_legend
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


iminfo_fn = impath.replace('.png', '.json')

with open(os.path.join(composite_dir, iminfo_fn), 'w') as f:
    minfo = dumps(mapinfo, cls=PythonObjectEncoder)

# To open:
# loads(minfo, object_hook=as_python_object)



# ---------------------------------------------------------------------------------
