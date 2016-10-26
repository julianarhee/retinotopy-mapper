#!/usr/bin/env python2
# coding: utf-8

# FROM plot_absolute_maps_GCaMP.ipnb (JNB)

import numpy as np
import os
from skimage.measure import block_reduce
# from scipy.misc import imread
from scipy.ndimage import imread
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
 

def py_ang(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'    """
    cosang = np.dot(v1, v2)
    sinang = la.norm(np.cross(v1, v2))
    return np.arctan2(sinang, cosang)
def gradient_phase(f, *varargs, **kwargs):
    """
    Return the gradient of an N-dimensional array.

    The gradient is computed using second order accurate central differences
    in the interior and either first differences or second order accurate
    one-sides (forward or backwards) differences at the boundaries. The
    returned gradient hence has the same shape as the input array.

    Parameters
    ----------
    f : array_like
        An N-dimensional array containing samples of a scalar function.
    varargs : scalar or list of scalar, optional
        N scalars specifying the sample distances for each dimension,
        i.e. `dx`, `dy`, `dz`, ... Default distance: 1.
        single scalar specifies sample distance for all dimensions.
        if `axis` is given, the number of varargs must equal the number of axes.
    edge_order : {1, 2}, optional
        Gradient is calculated using N\ :sup:`th` order accurate differences
        at the boundaries. Default: 1.

        .. versionadded:: 1.9.1

    axis : None or int or tuple of ints, optional
        Gradient is calculated only along the given axis or axes
        The default (axis = None) is to calculate the gradient for all the axes of the input array.
        axis may be negative, in which case it counts from the last to the first axis.

        .. versionadded:: 1.11.0

    Returns
    -------
    gradient : list of ndarray
        Each element of `list` has the same shape as `f` giving the derivative
        of `f` with respect to each dimension.

    Examples
    --------
    >>> x = np.array([1, 2, 4, 7, 11, 16], dtype=np.float)
    >>> np.gradient(x)
    array([ 1. ,  1.5,  2.5,  3.5,  4.5,  5. ])
    >>> np.gradient(x, 2)
    array([ 0.5 ,  0.75,  1.25,  1.75,  2.25,  2.5 ])

    For two dimensional arrays, the return will be two arrays ordered by
    axis. In this example the first array stands for the gradient in
    rows and the second one in columns direction:

    >>> np.gradient(np.array([[1, 2, 6], [3, 4, 5]], dtype=np.float))
    [array([[ 2.,  2., -1.],
            [ 2.,  2., -1.]]), array([[ 1. ,  2.5,  4. ],
            [ 1. ,  1. ,  1. ]])]

    >>> x = np.array([0, 1, 2, 3, 4])
    >>> dx = np.gradient(x)
    >>> y = x**2
    >>> np.gradient(y, dx, edge_order=2)
    array([-0.,  2.,  4.,  6.,  8.])

    The axis keyword can be used to specify a subset of axes of which the gradient is calculated
    >>> np.gradient(np.array([[1, 2, 6], [3, 4, 5]], dtype=np.float), axis=0)
    array([[ 2.,  2., -1.],
           [ 2.,  2., -1.]])
    """
    f = np.asanyarray(f)
    N = len(f.shape)  # number of dimensions

    axes = kwargs.pop('axis', None)
    if axes is None:
        axes = tuple(range(N))
    # check axes to have correct type and no duplicate entries
    if isinstance(axes, int):
        axes = (axes,)
    if not isinstance(axes, tuple):
        raise TypeError("A tuple of integers or a single integer is required")

    # normalize axis values:
    axes = tuple(x + N if x < 0 else x for x in axes)
    if max(axes) >= N or min(axes) < 0:
        raise ValueError("'axis' entry is out of bounds")

    if len(set(axes)) != len(axes):
        raise ValueError("duplicate value in 'axis'")

    n = len(varargs)
    if n == 0:
        dx = [1.0]*N
    elif n == 1:
        dx = [varargs[0]]*N
    elif n == len(axes):
        dx = list(varargs)
    else:
        raise SyntaxError(
            "invalid number of arguments")

    edge_order = kwargs.pop('edge_order', 1)
    if kwargs:
        raise TypeError('"{}" are not valid keyword arguments.'.format(
                                                  '", "'.join(kwargs.keys())))
    if edge_order > 2:
        raise ValueError("'edge_order' greater than 2 not supported")

    # use central differences on interior and one-sided differences on the
    # endpoints. This preserves second order-accuracy over the full domain.

    outvals = []

    # create slice objects --- initially all are [:, :, ..., :]
    slice1 = [slice(None)]*N
    slice2 = [slice(None)]*N
    slice3 = [slice(None)]*N
    slice4 = [slice(None)]*N

    otype = f.dtype.char
    if otype not in ['f', 'd', 'F', 'D', 'm', 'M']:
        otype = 'd'

    # Difference of datetime64 elements results in timedelta64
    if otype == 'M':
        # Need to use the full dtype name because it contains unit information
        otype = f.dtype.name.replace('datetime', 'timedelta')
    elif otype == 'm':
        # Needs to keep the specific units, can't be a general unit
        otype = f.dtype

    # Convert datetime64 data into ints. Make dummy variable `y`
    # that is a view of ints if the data is datetime64, otherwise
    # just set y equal to the array `f`.
    if f.dtype.char in ["M", "m"]:
        y = f.view('int64')
    else:
        y = f

    for i, axis in enumerate(axes):

        if y.shape[axis] < 2:
            raise ValueError(
                "Shape of array too small to calculate a numerical gradient, "
                "at least two elements are required.")
        
        # Numerical differentiation: 1st order edges, 2nd order interior
        if y.shape[axis] == 2 or edge_order == 1:
            
            # Use first order differences for time data
            out = np.empty_like(y, dtype=otype)

            slice1[axis] = slice(1, -1)
            slice2[axis] = slice(2, None)
            slice3[axis] = slice(None, -2)
            # 1D equivalent -- out[1:-1] = (y[2:] - y[:-2])/2.0
            out[slice1] = (y[slice2] - y[slice3])
            out[slice1] = (out[slice1] + math.pi) % (2*math.pi) - math.pi
            out[slice1]=out[slice1]/2.0

            slice1[axis] = 0
            slice2[axis] = 1
            slice3[axis] = 0
            # 1D equivalent -- out[0] = (y[1] - y[0])
            out[slice1] = (y[slice2] - y[slice3])
            out[slice1] = (out[slice1] + math.pi) % (2*math.pi) - math.pi

            slice1[axis] = -1
            slice2[axis] = -1
            slice3[axis] = -2
            # 1D equivalent -- out[-1] = (y[-1] - y[-2])
            out[slice1] = (y[slice2] - y[slice3])
            out[slice1] = (out[slice1] + math.pi) % (2*math.pi) - math.pi

        # Numerical differentiation: 2st order edges, 2nd order interior
        else:
            # Use second order differences where possible
            out = np.empty_like(y, dtype=otype)

            slice1[axis] = slice(1, -1)
            slice2[axis] = slice(2, None)
            slice3[axis] = slice(None, -2)
            # 1D equivalent -- out[1:-1] = (y[2:] - y[:-2])/2.0
            out[slice1] = (y[slice2] - y[slice3])
            out[slice1] = (out[slice1] + math.pi) % (2*math.pi) - math.pi
            out[slice1] = out[slice1]/2

            slice1[axis] = 0
            slice2[axis] = 0
            slice3[axis] = 1
            slice4[axis] = 2
            # 1D equivalent -- out[0] = -(3*y[0] - 4*y[1] + y[2]) / 2.0
            out[slice1] = -(3.0*y[slice2] - 4.0*y[slice3] + y[slice4])
            out[slice1] = (out[slice1] + math.pi) % (2*math.pi) - math.pi
            out[slice1]=out[slice1]/2.0

            slice1[axis] = -1
            slice2[axis] = -1
            slice3[axis] = -2
            slice4[axis] = -3
            # 1D equivalent -- out[-1] = (3*y[-1] - 4*y[-2] + y[-3])
            out[slice1] = (3.0*y[slice2] - 4.0*y[slice3] + y[slice4])
            out[slice1] = (out[slice1] + math.pi) % (2*math.pi) - math.pi
            out[slice1]=out[slice1]/2.0

        # divide by step size
        out /= dx[i]
        outvals.append(out)

        # reset the slice object in this dimension to ":"
        slice1[axis] = slice(None)
        slice2[axis] = slice(None)
        slice3[axis] = slice(None)
        slice4[axis] = slice(None)

    if len(axes) == 1:
        return outvals[0]
    else:
        return outvals



parser = optparse.OptionParser()
parser.add_option('--headless', action="store_true", dest="headless", default=False, help="run in headless mode, no figs")
parser.add_option('--reduce', action="store", dest="reduce_val", default="1", help="block_reduce value")
parser.add_option('--path', action="store", dest="path", default="", help="path to data directory")

# -- specific condition options --
parser.add_option('-r', '--run', action="store", dest="run", default=1, help="cutoff threshold value")
parser.add_option('--append', action="store", dest="append", default="", help="appended label for analysis structs")

# -- visualization and smoothing for maps --
parser.add_option('--cmap', action="store", dest="cmap", default='gist_rainbow', help="colormap for summary figures [default: gist_rainbow]")
parser.add_option('--smooth', action="store_true", dest="smooth", default=False, help="smooth? (default sig = 2)")
parser.add_option('--sigma', action="store", dest="sigma_val", default=2, help="sigma for gaussian smoothing")

# -- opts for checking out phase and mag maps (useful for thresholding) --
# parser.add_option('--contour', action="store_true", dest="contour", default=False, help="Show contour lines for phase map")
# parser.add_option('--power', action='store_true', dest='use_power', default=False, help="Use power or just magnitude?")

# -- whether use right or left just constrains whether signs should be flipped for Blue vs. Red in sign map --
parser.add_option('--right', action='store_false', dest='use_left', default=True, help="Use left-bottom or right-top config?")
parser.add_option('--noclean', action='store_false', dest='get_clean', default=True, help="Save borderless, clean maps for COREG")

# -- phase map starting point options (threshold/mask or not, use average of two maps, etc.) --
parser.add_option('--avg', action='store_true', dest='use_avg', default=False, help="Use averaged maps or single runs?")
parser.add_option('--mask', action='store_true', dest='use_mask', default=False, help="Use masked phase maps")
parser.add_option('--mask-type', action="store", dest="mask_type", type="choice", choices=['DC', 'blank', 'magmax', 'ratio'], default='DC', help="mag map to use for thresholding: DC | blank | magmax [default: DC]")
parser.add_option('--threshold', action="store", dest="threshold", default=0.2, help="Threshold for valid phase results (max of ratio map)")

# -- opts for tweaking how VF sign areas are smoothed/demarcated --
parser.add_option('--std', action="store", dest="std_thresh", default=0.3, help="STD threshold for VF sign map")

parser.add_option('--k1', action="store", dest="k1", default=5, help="kernel size for first opening")
parser.add_option('--k2', action="store", dest="k2", default=5, help="kernel size for morphol. steps")
parser.add_option('--kcanny', action="store", dest="k_canny", default=3, help="kernel size for dumb canny edge detection")


(options, args) = parser.parse_args()

k1 = int(options.k1)
k2 = int(options.k2)
k_canny = int(options.k_canny)

std_thresh = float(options.std_thresh)

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

exptdir = os.path.split(outdir)[0]
sessiondir = os.path.split(exptdir)[0]
print "EXPT: ", exptdir
print "SESSION: ", sessiondir

savedir = os.path.split(outdir)[0]
figdir = os.path.join(savedir, 'figures')
if not os.path.exists(figdir):
    os.makedirs(figdir)

date = os.path.split(os.path.split(os.path.split(outdir)[0])[0])[1]
experiment = os.path.split(os.path.split(outdir)[0])[1]

# --------------------------------------------------------------------
# Get blood vessel image:
# --------------------------------------------------------------------

folders = os.listdir(sessiondir)
figpath = [f for f in folders if f == 'surface']
if not figpath:
    figpath = [f for f in folders if f == 'figures']

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

else: # NO BLOOD VESSEL IMAGE...
    surface = np.zeros([200,300])
    print "No blood vessel image found. Using empty."

if reduceit:
    surface = block_reduce(surface, reduce_factor, func=np.mean)



# --------------------------------------------------------------------
# GET DATA STRUCT FILES:
# --------------------------------------------------------------------

append = options.append

files = os.listdir(outdir)
files = [f for f in files if os.path.splitext(f)[1] == '.pkl']


dstructs = [f for f in files if 'Target_fft' in f and str(reduce_factor) in f and append in f]
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


# --------------------------------------------------------------------
# Get condition key names:
# --------------------------------------------------------------------

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


# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Make legends:
# --------------------------------------------------------------------
# TODO:  fix this -- this is super janky way of getting accurate legends
# Physical screen mapping is dependent on the size of the monitor, and
# where the stimulus begins/ends on the visible screen.
# --------------------------------------------------------------------
# --------------------------------------------------------------------

use_corrected_screen = True
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
# TODO ---- FIX THIS:
# ----------------------------------------------------------------------------------------------
# This adjustment needs to be fixed for cases of using the older Samsung monitor (smaller)
# Also, any scripts in which horizontal condition started at the edge of the screen, rather than
# being centered around the screen middle.

ratio_factor = .5458049 # This is true / hardcoded only for AQUOS monitor.
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




# -------------------------------------------------------------------------
# GET CONDS:
# -------------------------------------------------------------------------

az_conds = ['Left', 'Right']
AZ = dict()
for az_idx,az_key in enumerate(az_keys):
    condkey = az_conds[az_idx]
    if len(az_key) > 1:
        print "Found more than 1 condition each for AZ-%s keys:" % condkey
        for run_idx, run_key in enumerate(az_key):
            print run_idx, run_key
        user_input=raw_input("\nChoose one [0,1...]:\n")
        if user_input=='':
            AZ[condkey] = az_key[0] # just take first cond key 
        else:
            AZ[condkey] = az_key[int(user_input)]
    else:
        AZ[condkey] = az_key[0]

el_conds = ['Top', 'Bottom']
EL = dict()
for el_idx,el_key in enumerate(el_keys):
    condkey = el_conds[el_idx]
    if len(el_key) > 1:
        print "Found more than 1 condition each for EL-%s keys:" % condkey
        for run_idx, run_key in enumerate(el_key):
            print run_idx, run_key
        user_input=raw_input("\nChoose one [0,1...]:\n")
        if user_input=='':
            EL[condkey] = el_key[0] # just take first cond key 
        else:
            EL[condkey] = el_key[int(user_input)]
    else:
        EL[condkey] = el_key[0]

leftmap = D[AZ['Left']]['ft']
rightmap = D[AZ['Right']]['ft']
topmap = D[EL['Top']]['ft']
bottommap = D[EL['Bottom']]['ft']

if threshold_type=='ratio':
    ratio_left = D[AZ['Left']]['ratio_map']
    ratio_right = D[AZ['Right']]['ratio_map']
    ratio_top = D[EL['Top']]['ratio_map']
    ratio_bottom = D[EL['Bottom']]['ratio_map']
# else:


# ----------------------- FIRST FIGURE -----------------------------------
# -------------------------------------------------------------------------
# Quick checkout of each condition:
# -------------------------------------------------------------------------
# colormap = 'gist_rainbow'
plt.subplot(2,4,1)
plt.title('left')
plt.imshow(np.angle(leftmap), cmap=colormap)
plt.axis('off')
plt.subplot(2,4,5)
plt.imshow(V_left_legend, cmap=colormap)
plt.axis('off')
plt.tight_layout()

plt.subplot(2,4,2)
plt.title('right')
plt.imshow(np.angle(rightmap), cmap=colormap)
plt.axis('off')
plt.subplot(2,4,6)
plt.imshow(V_right_legend, cmap=colormap)
plt.axis('off')

plt.subplot(2,4,3)
plt.title('top')
plt.imshow(np.angle(topmap), cmap=colormap)
plt.axis('off')
plt.subplot(2,4,7)
plt.imshow(H_top_legend, cmap=colormap)
plt.axis('off')

plt.subplot(2,4,4)
plt.title('bottom')
plt.imshow(np.angle(bottommap), cmap=colormap)
plt.axis('off')
plt.subplot(2,4,8)
plt.imshow(H_bottom_legend, cmap=colormap)
plt.axis('off')

plt.tight_layout()

plt.suptitle([experiment, date])

plt.show()


# -------------------------------------------------------
# -------------------------------------------------------
# Get phase maps, shift pi for averaging.  Then, AVERAGE.
# -------------------------------------------------------
# -------------------------------------------------------


# 1.  AZIMUTH maps --------------------------------------
# -------------------------------------------------------

# left_phase = np.angle(leftmap) #D[leftkey]['phase_map']
# right_phase = np.angle(rightmap.conjugate()) #D[rightkey]['phase_map']

# left_phase[left_phase<0] += 2*math.pi
# right_phase[right_phase<0] += 2*math.pi
# V_left_legend[V_left_legend<0] += 2*math.pi

if use_left is True:
    # For SIGN-MAP, need to combine left w/ bottom (or reverse sign of 0)
    # or, can combine right w/ top 
    # For LEGEND, need to use legend of map that is NOT conjugated for average.
    left_phase = np.angle(leftmap) #D[leftkey]['phase_map']
    right_phase = np.angle(rightmap.conjugate()) #D[rightkey]['phase_map']
    AZ_legend = copy.deepcopy(V_left_legend)

    top_phase = np.angle(topmap.conjugate()) #D[leftkey]['phase_map']
    bottom_phase = np.angle(bottommap) #D[rightkey]['phase_map']
    EL_legend = copy.deepcopy(H_bottom_legend)

else:
    left_phase = np.angle(leftmap.conjugate())
    right_phase = np.angle(rightmap)
    AZ_legend = copy.deepcopy(V_right_legend)

    top_phase = np.angle(topmap)
    bottom_phase = np.angle(bottommap.conjugate())
    EL_legend = copy.deepcopy(H_top_legend)


left_phase[left_phase<0] += 2*math.pi
right_phase[right_phase<0] += 2*math.pi
AZ_legend[AZ_legend<0] += 2*math.pi


# smooth = True
# sigma_val = (3,3)
if smooth is True:
    left_phase = ndimage.gaussian_filter(left_phase, sigma=sigma_val, order=0)
    right_phase = ndimage.gaussian_filter(right_phase, sigma=sigma_val, order=0)
    
vmin_val = 0
vmax_val = 2*math.pi

az_avg = (left_phase + right_phase) / 2.


# ----------------------- SECOND FIGURE ----------------------------------
# QUICK PLOTS: -----------------------------------------------------------
# ------------------------------------------------------------------------

from mpl_toolkits.axes_grid1 import make_axes_locatable

# surface:
plt.subplot(2,2,1)
plt.imshow(az_avg, cmap=colormap, vmin=vmin_val, vmax=vmax_val)
plt.axis('off')

# ratio of stimulation freq vs. all other:
# threshold = 0.2
if threshold_type=='ratio':
    az_ratio_map = (ratio_left + ratio_right) / 2.
    plt.subplot(2,2,2)
    plt.axis('off')
    ax = plt.gca()
    im = ax.imshow(az_ratio_map, cmap='hot')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    # mask phase with ratio:
    az_phase_mask = get_ratio_mask(az_avg, az_ratio_map, threshold)
    plt.subplot(2,2,3)
    plt.title('Threshold: %s of ratio max' % str(threshold))
    plt.imshow(surface, cmap='gray')
    plt.imshow(az_phase_mask, cmap=colormap, vmin=vmin_val, vmax=vmax_val)
    plt.axis('off')

    # legend
    plt.subplot(2,2,4)
    plt.imshow(AZ_legend, cmap=colormap)
    plt.axis('off')
    plt.suptitle('AZ AVG')
    plt.show()


# 2.  ELEVATION maps --------------------------------------
# ---------------------------------------------------------

# if use_left is True:
#     # For SIGN-MAP, need to combine left w/ bottom (or reverse sign of 0)
#     # or, can combine right w/ top 
#     # For LEGEND, need to use legend of map that is NOT conjugated for average.
#     top_phase = np.angle(topmap.conjugate()) #D[leftkey]['phase_map']
#     bottom_phase = np.angle(bottommap) #D[rightkey]['phase_map']
#     EL_legend = copy.deepcopy(H_bottom_legend)
# else:
#     top_phase = np.angle(topmap)
#     bottom_base = np.angle(bottom_map.conjugate())
#     EL_legend = copy.deepcopy(H_top_legend)



# Do the thing to deal with averaging across -pi and pi:
# Vmap[Vmap<0]=2*math.pi+Vmap[Vmap<0]

top_phase[top_phase<0] += 2*math.pi
bottom_phase[bottom_phase<0] += 2*math.pi
EL_legend[EL_legend<0] += 2*math.pi

vmin_val = 0
vmax_val = 2*math.pi

# smooth = True
# sigma_val = (3,3)
# smooth = True
if smooth is True:
    top_phase = ndimage.gaussian_filter(top_phase, sigma=sigma_val, order=0)
    bottom_phase = ndimage.gaussian_filter(bottom_phase, sigma=sigma_val, order=0)
    
# To calculate average for 
el_avg = (bottom_phase + top_phase) / 2.


# ----------------------- THIRD FIGURE ----------------------------------
# QUICK PLOTS: ----------------------------------------------------------
# -----------------------------------------------------------------------

# surface
plt.subplot(2,2,1)
plt.imshow(el_avg, cmap=colormap, vmin=vmin_val, vmax=vmax_val)
plt.axis('off')

# ratio of stimulation freq vs. all other:
# threshold = 0.2
if threshold_type=='ratio':
    el_ratio_map = (ratio_top + ratio_bottom) / 2.
    plt.subplot(2,2,2)
    plt.axis('off')
    ax = plt.gca()
    im = ax.imshow(el_ratio_map, cmap='hot')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    # mask phase with ratio:
    el_phase_mask = get_ratio_mask(el_avg, el_ratio_map, threshold)

    plt.subplot(2,2,3)
    plt.title('Threshold: %s of ratio max' % str(threshold))
    plt.imshow(surface, cmap='gray')
    plt.imshow(el_phase_mask, cmap=colormap, vmin=vmin_val, vmax=vmax_val)
    plt.axis('off')

    # legend
    plt.subplot(2,2,4)
    plt.imshow(EL_legend, cmap=colormap, vmin=vmin_val, vmax=vmax_val)
    plt.axis('off')
    plt.suptitle('EL AVG')
    plt.show()

# -------------------------------------------------------
# -------------------------------------------------------
# Calculate gradients, make field sign map.
# -------------------------------------------------------
# -------------------------------------------------------

# az_phase_mask = get_ratio_mask(az_avg, az_ratio_map, threshold) # reduce thresh to 0.10
# el_phase_mask = get_ratio_mask(el_avg, el_ratio_map, threshold) # thres 0.15 good


if use_avg is True:
    if use_mask is True:
        az_phase_mask = get_ratio_mask(az_avg, az_ratio_map, threshold) # reduce thresh to 0.10
        el_phase_mask = get_ratio_mask(el_avg, el_ratio_map, threshold) # thres 0.15 good

        Hmap = el_phase_mask
        Vmap = az_phase_mask
    else:
        Hmap = el_avg
        Vmap = az_avg
else:
    if use_left is True:
        if use_mask is True:
            Hmap = get_ratio_mask(bottom_phase, ratio_bottom, threshold)
            Vmap = get_ratio_mask(left_phase, ratio_left, threshold)
        else:
            Hmap = bottom_phase
            Vmap = left_phase
    else:
        if use_mask is True:
            Hmap = get_ratio_mask(top_phase, ratio_top, threshold)
            Vmap = get_ratio_mask(right_phase, ratio_right, threshold)
        else:
            Hmap = top_phase
            Vmap = right_phase



# smooth = True
sigma_val = (3, 3)
if smooth is True:
    Hmap = ndimage.gaussian_filter(Hmap, sigma=sigma_val, order=0)
    Vmap = ndimage.gaussian_filter(Vmap, sigma=sigma_val, order=0)


[Hgy,Hgx]=np.array(gradient_phase(Hmap))

# Hgy =(Hgy + math.pi) % (2*math.pi) - math.pi
# Hgx =(Hgx + math.pi) % (2*math.pi) - math.pi

[Vgy,Vgx]=np.array(gradient_phase(Vmap))

Hgdir=np.arctan2(Hgy,Hgx) # gradient direction
Vgdir=np.arctan2(Vgy,Vgx)
# Hgdir[Hgdir<0]=2*math.pi+Hgdir[Hgdir<0]
# Vgdir[Vgdir<0]=2*math.pi+Vgdir[Vgdir<0]


gdiff = Vgdir-Hgdir
gdiff = (gdiff + math.pi) % (2*math.pi) - math.pi

# O=-1*np.sin(gdiff)
O=np.sin(gdiff) # LEFT goes w/ BOTTOM.  RIGHT goes w/ TOP.
S=np.sign(O)


# PLOT:
# -----------------------------------------------------
# plt.subplot(131)
# plt.imshow(Hgdir,cmap='jet');
# # plt.colorbar();
# plt.axis('off')
# plt.title('Hgdir')

# plt.subplot(132)
# plt.imshow(Vgdir,cmap='jet');
# # plt.colorbar();
# plt.axis('off')
# plt.title('Vgdir')


# plt.subplot(133)
# plt.imshow(S,cmap='jet');
# plt.axis('off')
# plt.title('sign')
# # plt.colorbar()


# ----------------------- FOURTH FIGURE ----------------------------------
# -------------------------------------------------------
# -------------------------------------------------------
# Calculate STD, and threshold to separate areas (simple morph. step)
# -------------------------------------------------------
# -------------------------------------------------------

O_sigma=np.nanstd(O)

S_thresh=np.zeros(np.shape(O))
# std_thresh = float(options.std_thresh) #.2
S_thresh[O>(O_sigma*std_thresh)]=1
S_thresh[O<(-1*O_sigma*std_thresh)]=-1

plt.figure()
plt.subplot(1,3,1)
plt.imshow(surface, cmap='gray')
plt.imshow(S_thresh,cmap='bwr', alpha=0.5);
plt.axis('off')
plt.title('Threshold to separate areas by sign.')
# plt.colorbar();

# -------------------------------------------------------
# -------------------------------------------------------
# MORPHOLOGY STEPS to get borders.
# -------------------------------------------------------
# -------------------------------------------------------

# Processing steps of Criterion 2 (S-thersh) split patches 
# that are identified as being multiple areas...
# Low threshold tends to let through background noise.

# 1.  Doing a simple "opening" eliminates much of this (supposedly):
# Opening is basically erosion + dilation. Good for getting rid of noise.
# This sometimes misses borders.. (Criterion 2) -- see STEP 3. 

import cv2

first_kernel_sz = k1 #options.k1 #10 #10
kernel = np.ones((first_kernel_sz,first_kernel_sz))

first_open = cv2.morphologyEx(S_thresh, cv2.MORPH_OPEN, kernel)
plt.subplot(1,3,2)
plt.title('S_thresh, std: %s' % str(std_thresh))
plt.imshow(S_thresh, cmap='bwr')
plt.axis('off')

plt.subplot(1,3,3)
plt.title('first open, kernel %s' % first_kernel_sz)
plt.imshow(first_open, cmap='bwr')
plt.axis('off')

plt.suptitle('Threshold and open to separate areas.')


# ----------------------- FIFTH FIGURE ----------------------------------

# 2.  Additional morphology steps:
# -------------------------------------------------------
# These steps are to connect all regions that should be connected.
# This can fail, see STEP 3.

plt.figure()

plt.subplot(1,5,1)
plt.title('S_thresh, std: %s' % str(std_thresh))
# plt.imshow(surface, cmap='gray')
plt.imshow(S_thresh, cmap='bwr')
plt.axis('off')

plt.subplot(1,5,2)
plt.title('first open, kernel %s' % first_kernel_sz)
plt.imshow(first_open, cmap='bwr')
plt.axis('off')

# a.  Closing on abs(S-thresh)
# Closing is dilation, then erosion. Good for closing small holes
close_kernel = k2 #10
kernel = np.ones((close_kernel,close_kernel))

# This must be wrong -- abs() gets rid of all borders...
# closing = cv2.morphologyEx(abs(S_thresh), cv2.MORPH_CLOSE, kernel)
closing = cv2.morphologyEx(first_open, cv2.MORPH_CLOSE, kernel)

plt.subplot(1,5,3)
plt.imshow(closing, cmap='bwr')
plt.axis('off')
plt.title('closing, kernel %i' % close_kernel)


open_kernel = k2 #10
kernel = np.ones((close_kernel,close_kernel))
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
plt.subplot(1,5,4)
plt.imshow(opening, cmap='bwr')
plt.axis('off')
plt.title('opening, k %i' % open_kernel)


niter = 1
dilate_kernel = k2 #5
kernel = np.ones((dilate_kernel,dilate_kernel))
dilation = cv2.dilate(opening, kernel, iterations = niter)
plt.subplot(1,5,5)
plt.imshow(dilation, cmap='bwr')
plt.axis('off')
plt.title('dilation, k: %i' % dilate_kernel)

plt.show()


# --------------------------------------------------------------
# --------------------------------------------------------------
# Use space between found regions to demarcate borders.
# Likely, this can be better done with some kind of watershed...
# Possible to do this also in a dumb way with canny edge detection.
# --------------------------------------------------------------
# --------------------------------------------------------------


# ----------------------- SIXTH FIGURE ----------------------------------

# --------------------------------------------------------------
# Next, recompute areal borders using morphological “thinning,” 
# iterating to infinity. Do this until boundaries between areas
# is 1 pixel wide...
# --------------------------------------------------------------

# get canny edges

cn_min = 0
cn_max = 1
dilationU8 = np.uint8(dilation)
canny_edges = cv2.Canny(dilationU8, cn_min, cn_max)
plt.subplot(1,2,1)
plt.imshow(canny_edges, cmap='gray_r')
plt.axis('off')
plt.imshow(dilation, cmap='bwr', alpha=0.5)
plt.axis('off')
plt.title('canny edge detect on dilation')

# DILATE to fill? # This is obviously wrong, 
# but not sure best way to do "thinning"...
# --------------------------------------------------------------
canny_edge_kernel = np.ones((k_canny, k_canny)) #np.ones((3,3))
canny_close_kernel = np.ones((k_canny, k_canny)) #np.ones((3,3))
canny_dilate = cv2.dilate(canny_edges, canny_edge_kernel, iterations = niter)
canny_closed = cv2.morphologyEx(canny_dilate, cv2.MORPH_CLOSE, canny_close_kernel)
plt.subplot(1,2,2)
plt.imshow(canny_closed)
plt.title('morph steps to clean up borders')

plt.show()

# h, w = canny_edges.shape[:2]
# contours0, hierarchy = cv2.findContours( canny_edges.copy(), cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)

# contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours0]
# vis = np.zeros((h, w, 3), np.uint8)
# cv2.drawContours(vis,contours,0,255,-1)
# vis = cv2.bitwise_not(vis)

# plt.subplot(1,2,2)
# plt.imshow(vis)
# plt.axis('off')



# ----------------------- SEVENTH FIGURE ----------------------------------

# -----------------------------------------------------------
# -----------------------------------------------------------
# SUMMARY PLOTS:
# -----------------------------------------------------------
# -----------------------------------------------------------

plt.figure()

plt.subplot(2,3,1)
plt.title('Horizontal')
plt.imshow(surface, cmap='gray')
plt.axis('off')
plt.imshow(Hmap, alpha=0.8, vmin=vmin_val, vmax=vmax_val)
plt.axis('off')


plt.subplot(2,3,2)
plt.title('Vertical')
plt.imshow(surface, cmap='gray')
plt.axis('off')
plt.imshow(Vmap, alpha=0.8, vmin=vmin_val, vmax=vmax_val)
plt.axis('off')

plt.subplot(2,3,3)
plt.title('VF Sign Map')
plt.imshow(surface, cmap='gray')
plt.axis('off')
plt.imshow(O, alpha=0.8)
plt.axis('off')

plt.subplot(2,3,6)
plt.title('VF Patches (post)')
plt.imshow(surface, cmap='gray')
plt.axis('off')
plt.imshow(dilation, cmap='bwr', alpha=0.8)
plt.axis('off')


plt.subplot(2,3,4)
plt.imshow(EL_legend, vmin=vmin_val, vmax=vmax_val)
plt.axis('off')

plt.subplot(2,3,5)
plt.imshow(AZ_legend, vmin=vmin_val, vmax=vmax_val)
plt.axis('off')

im_info = 'std: %s, k1: %i, k2: %i' % (str(std_thresh), first_kernel_sz, close_kernel)
title = [experiment, date, im_info]
plt.suptitle(title)




# ----------------------- EIGHTH FIGURE ----------------------------------

# Contours -- figures made to psuedo-match the original paper
# -----------------------------------------------------------
# Compare to Garrett et al., 2014 FIG.1
# -----------------------------------------------------------

plt.figure()

boost_vmin_val = 2
boost_vmax_val = 5

plt.subplot(2,3,1)
plt.title('Horizontal')
plt.imshow(surface, cmap='gray')
plt.axis('off')
plt.imshow(Hmap, vmin=boost_vmin_val, vmax=boost_vmax_val, alpha=0.8)
plt.axis('off')

plt.subplot(2,3,2)
plt.title('Vertical')
plt.imshow(surface, cmap='gray')
plt.axis('off')
plt.imshow(Vmap, vmin=boost_vmin_val, vmax=boost_vmax_val, alpha=0.8)
plt.axis('off')

plt.subplot(2,3,3)
plt.title('VF Sign Map')
plt.imshow(surface, cmap='gray')
plt.axis('off')
plt.imshow(O, alpha=0.8)
plt.axis('off')

plt.subplot(2,3,6)
plt.title('VF Patches (Sthresh)')
# plt.imshow(S_thresh, cmap='bwr')
plt.imshow(S_thresh, cmap='bwr')
# plt.imshow(dilation, cmap='bwr')
plt.axis('off')


# Overlay contours over found-boundaries:
# ------------------------------------------
# canny_dilate[canny_dilate==0] = np.nan
# canny_dilate = np.ma.array(canny_dilate)

levels = np.arange(boost_vmin_val, boost_vmax_val, .2)  # Boost the upper limit to avoid truncation errors.
# levels = np.arange(vmin_val, vmax_val, .2)  # Boost the upper limit to avoid truncation errors.

plt.subplot(2,3,4)
plt.title('AZ contours')
# plt.imshow(borders)...
# plt.imshow(surface, cmap='gray', alpha=0.2)
plt.contour(Vmap, levels, origin='lower', linewidths=2, alpha=0.8)
plt.axis('off')
plt.imshow(canny_closed, cmap='gray_r')
plt.axis('off')

plt.subplot(2,3,5)
plt.title('EL contours')
# plt.imshow(borders)...
# plt.imshow(surface, cmap='gray', alpha=0.2)
plt.imshow(canny_closed, cmap='gray_r')
plt.contour(Hmap, levels, origin='lower', linewidths=2, alpha=0.8)
plt.axis('off')


im_info = 'phase thresh: %s, std: %s, k1: %i, k2: %i' % (str(threshold), str(std_thresh), first_kernel_sz, close_kernel)
title = [experiment, date, im_info]

plt.suptitle(title)

plt.show()


# # ----------------------------------------------------------------------------------------
# # IMAGE DILATION and etc.....
# # ----------------------------------------------------------------------------------------
# from scipy import ndimage
# # im2 = ndimage.grey_dilation(S_thresh)

# import cv2
# kernel = np.ones((2,2),np.uint8)

# opening = cv2.morphologyEx(S_thresh, cv2.MORPH_OPEN, kernel)

# closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
# opening2 = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

# dilation = cv2.dilate(opening2,kernel,iterations = 1)


# plt.imshow(dilation,cmap='bwr', alpha=0.3);
# plt.axis('off')
# plt.colorbar();

# plt.show()


# # ANOTHER ATTEMPT:
# # ----------------------------------------------------------------------------------------
# from skimage.morphology import erosion, dilation, opening, closing, white_tophat
# from skimage.morphology import black_tophat, skeletonize, convex_hull_image

# from skimage.morphology import disk

# selem = disk(2)

# plt.figure(figsize=(20,10))

# # OPEN:
# plt.subplot(1,4,1)
# opened = opening(abs(S_thresh), selem)
# plt.imshow(surface, cmap='gray')
# plt.imshow(opened,cmap='bwr', alpha=0.5);
# plt.axis('off')
# plt.title('opening')

# # CLOSE:
# plt.subplot(1,4,2)
# # closed = closing(opened, selem)
# closed = closing(opened, selem)
# plt.imshow(surface, cmap='gray')
# plt.imshow(closed,cmap='bwr', alpha=0.5);
# plt.axis('off')
# plt.title('closing')

# # OPEN2:
# plt.subplot(1,4,3)
# opened2 = closing(closed, selem)
# plt.imshow(surface, cmap='gray')
# plt.imshow(opened2,cmap='bwr', alpha=0.5);
# plt.axis('off')
# plt.title('re-opening')

# # DILATE:
# plt.subplot(1,4,4)
# dilated = dilation(opened2, selem)
# skel = skeletonize(abs(dilated))
# plt.imshow(surface, cmap='gray')
# plt.imshow(dilated,cmap='bwr', alpha=0.5);
# plt.imshow(skel,cmap='gray', alpha=0.5);
# plt.axis('off')
# plt.title('dilated')
# # plt.colorbar();

# plt.show()

