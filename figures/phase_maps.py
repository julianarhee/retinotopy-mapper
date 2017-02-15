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
parser.add_option('--threshold', action="store", dest="threshold", default=0.2, help="Threshold (max of ratio map)")

parser.add_option('--short-axis', action="store_false", dest="use_long_axis", default=True, help="Used short-axis instead of long?")


(options, args) = parser.parse_args()

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
    ims = [i for i in ims if exptdate in i]
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
struct_fns = [f for f in struct_fns if os.path.splitext(f)[1] == '.pkl']

if len(files) > 0:
    if len(files) == 1: # composite struct exists
        composite_struct_fn = os.path.join(outdir, struct_fns[0])

    elif len(files) > 1:
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

else:
    print "No composite struct found. Creating new."
    composite_struct_fn = '{date}_{animal}_struct.pkl'.format(date=date, animal=subject)
    print "New struct name is: %s" % composite_struct_fn

    D = dict()
    for condition in conditions:
        condition_dir = os.path.join(outdir, condition, 'structs')
        condition_structs = os.listdir(condition_dir)
        condition_structs = [f for f in condition_structs if '.pkl' in f and 'fft' in f]
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



AZ = dict()
condition_keys = D.keys()

print "Select session for AZIMUTH maps (LEFT):"
for cond_idx, cond_fn in enumerate(condition_keys):
    print cond_idx, cond_fn
user_input=raw_input("\nChoose a session [0,1...]:\n")

selected_left_condition = condition_keys[int(user_input)]

run_keys = D[selected_left_condition].keys()
run_keys = [r for r in run_keys if 'Left' in r]
for run_idx, run_fn in enumerate(run_keys):
    print run_idx, run_fn
user_input=raw_input("\nChoose LEFT run [0,1...]:\n")
selected_left_run = run_keys[int(user_input)]
AZ['left'] = D[selected_left_condition][selected_left_run]


print "Select session for AZIMUTH maps (RIGHT):"
for cond_idx, cond_fn in enumerate(condition_keys):
    print cond_idx, cond_fn
user_input=raw_input("\nChoose a session [0,1...]:\n")
selected_right_condition = condition_keys[int(user_input)]

run_keys = D[selected_right_condition].keys()
run_keys = [r for r in run_keys if 'Right' in r]
for run_idx, run_fn in enumerate(run_keys):
    print run_idx, run_fn
user_input=raw_input("\nChoose RIGHT run [0,1...]:\n")
selected_right_run = run_keys[int(user_input)]
AZ['right'] = D[selected_right_condition][selected_right_run]



EL = dict()

print "Select session for ELEVATION maps (TOP):"
for cond_idx, cond_fn in enumerate(condition_keys):
    print cond_idx, cond_fn
user_input=raw_input("\nChoose a session [0,1...]:\n")

selected_top_condition = condition_keys[int(user_input)]

run_keys = D[selected_top_condition].keys()
run_keys = [r for r in run_keys if 'Top' in r]
for run_idx, run_fn in enumerate(run_keys):
    print run_idx, run_fn
user_input=raw_input("\nChoose TOP run [0,1...]:\n")
selected_top_run = run_keys[int(user_input)]
EL['top'] = D[selected_top_condition][selected_top_run]


print "Select session for ELEVATION maps (BOTTOM):"
for cond_idx, cond_fn in enumerate(condition_keys):
    print cond_idx, cond_fn
user_input=raw_input("\nChoose a session [0,1...]:\n")
selected_bottom_condition = condition_keys[int(user_input)]

run_keys = D[selected_bottom_condition].keys()
run_keys = [r for r in run_keys if 'Bottom' in r]
for run_idx, run_fn in enumerate(run_keys):
    print run_idx, run_fn
user_input=raw_input("\nChoose BOTTOM run [0,1...]:\n")
selected_bottom_run = run_keys[int(user_input)]
EL['bottom'] = D[selected_bottom_condition][selected_bottom_run]




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
left_map = AZ['left']['ft']
right_map = AZ['right']['ft']
top_map = EL['top']['ft']
bottom_map = EL['bottom']['ft']

ratio_left = AZ['left']['ratio_map']
ratio_right = AZ['right']['ratio_map']
ratio_top = EL['top']['ratio_map']
ratio_bottom = EL['bottom']['ratio_map']

threshold = 0.001
thresh_left_phase = np.angle(left_map)
thresh_left_phase[np.where(ratio_left < threshold)] = np.nan

thresh_right_phase = np.angle(right_map)
thresh_right_phase[np.where(ratio_right < threshold)] = np.nan

thresh_top_phase = np.angle(top_map)
thresh_top_phase[np.where(ratio_top < threshold)] = np.nan

thresh_bottom_phase = np.angle(bottom_map)
thresh_bottom_phase[np.where(ratio_bottom < threshold)] = np.nan


# Quick checkout:
#colormap = 'gist_rainbow'

alpha_val = 0.5
plt.subplot(2,4,1)
plt.title('left')
plt.imshow(surface, cmap='gray')
plt.imshow(thresh_left_phase, cmap=colormap, alpha=alpha_val)
plt.axis('off')
plt.subplot(2,4,5)
plt.imshow(V_left_legend, cmap=colormap, alpha=alpha_val)
plt.axis('off')
plt.tight_layout()

plt.subplot(2,4,2)
plt.title('right')
plt.imshow(surface, cmap='gray')
plt.imshow(thresh_right_phase, cmap=colormap, alpha=alpha_val)
plt.axis('off')
plt.subplot(2,4,6)
plt.imshow(V_right_legend, cmap=colormap, alpha=alpha_val)
plt.axis('off')

plt.subplot(2,4,3)
plt.title('top')
plt.imshow(surface, cmap='gray')
plt.imshow(thresh_top_phase, cmap=colormap, alpha=alpha_val)
plt.axis('off')
plt.subplot(2,4,7)
plt.imshow(H_top_legend, cmap=colormap, alpha=alpha_val)
plt.axis('off')

plt.subplot(2,4,4)
plt.title('bottom')
plt.imshow(surface, cmap='gray')
plt.imshow(thresh_bottom_phase, cmap=colormap, alpha=alpha_val)
plt.axis('off')
plt.subplot(2,4,8)
plt.imshow(H_bottom_legend, cmap=colormap, alpha=alpha_val)
plt.axis('off')

plt.tight_layout()

plt.suptitle([date, subject])

plt.show()

imname = 'thresholded_maps_thresh%0.4f' % threshold
imname.replace('.', 'x')

impath = os.path.join(fig_dir, imname+'.png')
plt.savefig(impath, bbox_inches='tight', pad_inches = 0)

plt.show()



# -------------------------------------------------------
# Get phase maps, shift pi for averaging.  Then, AVERAGE.
# -------------------------------------------------------
vmin_val = 0 #-1*math.pi # 0
vmax_val = 2*math.pi

shift_left_phase = np.angle(left_map)
shift_right_phase = np.angle(right_map.conjugate())
shift_left_phase[shift_left_phase<0] += 2*math.pi
shift_right_phase[shift_right_phase<0] += 2*math.pi
shift_az_legend = copy.deepcopy(V_left_legend)
shift_az_legend[shift_az_legend<0] += 2*math.pi
shift_other_az_legend = copy.deepcopy(V_right_legend)
shift_other_az_legend[shift_other_legend<0] += 2*math.pi


shift_top_phase = np.angle(top_map) 
shift_bottom_phase = np.angle(bottom_map.conjugate())
shift_top_phase[shift_top_phase<0] += 2*math.pi
shift_bottom_phase[shift_bottom_phase<0] += 2*math.pi
shift_el_legend = copy.deepcopy(H_top_legend)
shift_el_legend[shift_el_legend<0] += 2*math.pi
shift_other_el_legend = copy.deepcopy(H_bottom_legend)
shift_other_el_legend[shift_other_el_legend<0] += 2*math.pi

avg_az_phase = (shift_left_phase + shift_right_phase) * 0.5
avg_el_phase = (top_phase + bottom_phase) * 0.5

if show_avg is True:
    plt.subplot(2,2,1)
    plt.imshow(avg_az_phase, cmap=colormap, vmin=vmin_val, vmax=vmax_val)
    plt.axis('off')
    plt.subplot(2,2,3)
    plt.imshow(az_legend, cmap=colormap, vmin=vmin_val, vmax=vmax_val)
    plt.axis('off')

    avg_el_phase = (top_phase + bottom_phase) * 0.5
    plt.subplot(2,2,2)
    plt.imshow(avg_el_phase, cmap=colormap, vmin=vmin_val, vmax=vmax_val)
    plt.axis('off')
    plt.subplot(2,2,4)
    plt.imshow(el_legend, cmap=colormap, vmin=vmin_val, vmax=vmax_val)
    plt.axis('off')

    plt.tight_layout()

    plt.suptitle(['AVG', date, subject])

    plt.show()


    imname = 'avg_phases'

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


path_to_map_struct = os.path.join(composite_dir, 'maps.pkl')
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
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    imname = 'avg_phase_AZ_HSV_SURFACE'

    impath = os.path.join(figdir, imname+'.png')
    plt.savefig(impath, bbox_inches='tight', pad_inches = 0)

    plt.show()

    # AZ average 
    # --------------------------------------------------------------------------------------
    fig = plt.imshow(surface, cmap='gray')
    plt.imshow(right_phase, cmap='hsv', vmin=vmin_val, vmax=vmax_val, alpha=0.5)
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    imname = 'overlay_avg_phase_AZ_HSV_PHASE'

    impath = os.path.join(figdir, imname+'.png')
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

        impath = os.path.join(figdir, imname+'.png')
        plt.savefig(impath, bbox_inches='tight', pad_inches = 0)

        plt.show()


    fig = plt.imshow(AZ_legend, cmap='hsv', vmin=vmin_val, vmax=vmax_val)
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    imname = 'avg_phase_AZ_HSV_PHASE_LEGEND'

    impath = os.path.join(figdir, imname+'.png')
    plt.savefig(impath, bbox_inches='tight', pad_inches = 0)

    plt.show()
    # EL average 
    # --------------------------------------------------------------------------------------

    fig = plt.imshow(el_avg, cmap='hsv', vmin=vmin_val, vmax=vmax_val)
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    imname = 'avg_phase_EL_HSV_PHASE'

    impath = os.path.join(figdir, imname+'.png')
    plt.savefig(impath, bbox_inches='tight', pad_inches = 0)

    plt.show()


I = dict()
I['az_phase'] = az_avg
I['vmin'] = vmin_val
I['vmax'] = vmax_val
I['az_legend'] = AZ_legend
I['surface'] = surface

fext = 'clean_fig_info.pkl'
fname = os.path.join(figdir, fext)
with open(fname, 'wb') as f:
    # protocol=pkl.HIGHEST_PROTOCOL)
    pkl.dump(I, f, protocol=pkl.HIGHEST_PROTOCOL)


# mat_fn = 'temp2sample'+'.pkl'
# # scipy.io.savemat(os.path.join(source_dir, condition, tif_fn), mdict=pydict)

# import scipy.io
# scipy.io.savemat(os.path.join(out_path, mat_fn), mdict=T)
# print os.path.join(out_path, 'mw_data', mat_fn)
