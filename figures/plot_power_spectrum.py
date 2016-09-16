#!/usr/bin/env python2
# coding: utf-8

import numpy as np
import os
from skimage.measure import block_reduce
#from scipy.misc import imread
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


(options, args) = parser.parse_args()

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
figpath = [f for f in folders if f == 'surface']
# figpath = [f for f in folders if f == 'figures'][0]
# print "EXPT: ", exptdir
# print "SESSION: ", sessiondir
print "path to surface: ", figpath

if figpath:
    # figdir = figpath[0]
    figpath = figpath[0]
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
dstructs = [f for f in files if 'power_' in f and str(reduce_factor) and append in f]
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


# Get specific keys:

bottomkeys = [k for k in D.keys() if 'Bottom' in k or 'Up' in k]
topkeys = [k for k in D.keys() if 'Top' in k or 'Down' in k]

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



cond_types = ['Left', 'Right', 'Top', 'Bottom']
for cond in cond_types:
    print "RUNNING cond: ", cond
    # run_conds = [cond, str(run_num)+'_', str(reduce_factor), append]
    run_conds = [cond, str(reduce_factor), append]

    if cond=='Left':
        tmp_keys = [k for k in leftkeys if all([c in k for c in run_conds])] #[0]
        #legend = V_left_legend
    elif cond=='Right':
        tmp_keys = [k for k in rightkeys if all([c in k for c in run_conds])] #[0]
        #legend = V_right_legend
    elif cond=='Top':
        tmp_keys = [k for k in topkeys if all([c in k for c in run_conds])] #[0]
        #legend = H_down_legend
    elif cond=='Bottom':
        tmp_keys = [k for k in bottomkeys if all([c in k for c in run_conds])] #[0]
        #legend = H_up_legend

    if tmp_keys==[]:
        print "No matches found from list: %s", cond
    else:
        for curr_key in tmp_keys:
            print "Curr key is: ", curr_key
            #curr_amp_key = curr_key.split('power')[1]
            #print "Corresponding AMP key is: ", curr_amp_key

            magnitudes = D[curr_key]['magnitudes']
            freqs = D[curr_key]['freqs']
            Ny = len(freqs)/2.

            target_bin = D[curr_key]['target_bin']
            target = D[curr_key]['target']

            plt.plot(freqs, magnitudes[x*y, target_bin], 'k')
            plt.plot(freqs[target_bin], 0, 'r*')
            title = 'power spectrum @ x=%i, y=%i (target=%s)' % (int(xpix), int(ypix), str(freqs[target_bin]))

            plt.title(title)

            # fig = plt.figure()
            # plt.subplot(2,2,3)
            # plt.imshow(np.angle(curr_map), cmap='spectral', vmin=-1*math.pi, vmax=1*math.pi)
            # # plt.title('AZ: left')
            # plt.axis('off')


