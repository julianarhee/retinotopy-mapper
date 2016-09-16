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
from scipy import ndimage
import datetime
import time

def movingaverage(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
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
parser.add_option('--headless', action="store_true", dest="headless",
                  default=False, help="run in headless mode, no figs")
parser.add_option('--freq', action="store", dest="target_freq",
                  default="0.05", help="stimulation frequency")
parser.add_option('--reduce', action="store",
                  dest="reduce_val", default="2", help="block_reduce value")
#parser.add_option('--sigma', action="store", dest="gauss_kernel",
#                  default="0", help="size of Gaussian kernel for smoothing")
parser.add_option('--format', action="store",
                  dest="im_format", default="png", help="saved image format")
parser.add_option('--fps', action="store",
                  dest="sampling_rate", default="60", help="saved image format")
parser.add_option('--append', action="store",
                  dest="append_name", default="", help="append string to saved file name")
parser.add_option('--circle', action="store_true", dest="circle",
                  default=False, help="is the stimulus a circle or bar?")
parser.add_option('--CW', action="store_true", dest="CW",
                  default=False, help="circle stim ONLY: CW or not?")
parser.add_option('--average', action="store_true", dest="get_average_cycle",
                  default=False, help="average cycles or no?")
parser.add_option('--detrend-first', action="store_true", dest="detrend_first",
                  default=False, help="detrend first, then mean subtract?")
parser.add_option('--smooth', action="store_true", dest="smooth", default=False, help="smooth? (default sig = 2)")
parser.add_option('--sigma', action="store", dest="sigma_val", default=2, help="sigma for gaussian smoothing")
parser.add_option('--sum', action="store_true", dest="use_sum", default=False, help="if reduce, sum (otherwise func is np.mean)")
parser.add_option('--global', action="store_true", dest="mean_subtract_global", default=False, help="subtract mean of each frame or subtract global mean")
#parser.add_option('--nomean', action="store_false", dest="mean_subtract", default=True, help="subtract mean at all or no")

parser.add_option('--rolling', action='store_true', default=False, help="Rolling average [window size is 2 cycles] or detrend.")
parser.add_option('--meansub', action='store_true', default=False, help="Remove mean of each frame.")
parser.add_option('--interpolate', action='store_true', default=False, help='Interpolate frames or no.')

(options, args) = parser.parse_args()

imdir = sys.argv[1]

#mean_subtract = options.mean_subtract
mean_subtract_global = options.mean_subtract_global
use_sum = options.use_sum

rolling = options.rolling
interpolate = options.interpolate
meansub = options.meansub

smooth = options.smooth
sigma_val_num = options.sigma_val
sigma_val = (int(sigma_val_num), int(sigma_val_num))

# processed_dir = os.path.join(os.path.split(imdir)[0], 'processed')

# if not os.path.exists(processed_dir):
#     os.makedirs(processed_dir)

detrend_first = options.detrend_first
circle = options.circle
CW = options.CW
get_average_cycle = options.get_average_cycle

im_format = '.' + options.im_format
headless = options.headless
target_freq = float(options.target_freq)
reduce_factor = (int(options.reduce_val), int(options.reduce_val))
if reduce_factor[0] > 0:
    reduceit = 1
else:
    reduceit = 0
#gsigma = int(options.gauss_kernel)

if headless:
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pylab as plt
import matplotlib.cm as cm

sampling_rate = float(options.sampling_rate) # 60.  # np.mean(np.diff(sorted(strt_idxs)))/cycle_dur #60.0
cache_file = True
cycle_dur = 1. / target_freq  # 10.
binspread = 0

append_to_name = str(options.append_name)

basepath = os.path.split(os.path.split(imdir)[0])[0]
session = os.path.split(os.path.split(imdir)[0])[1]
cond = os.path.split(imdir)[1]

if get_average_cycle:
    movie_type = 'avgcycle'
else:
    movie_type = 'all'
if detrend_first is True:
    detrend_flag = '_detrendfirst'
else:
    detrend_flag = ''

if smooth is True:
    smooth_flag = '_smooth%i' % int(sigma_val_num)
else:
    smooth_flag = ''

if mean_subtract_global is True:
    mean_flag = '_global'
else:
    mean_flag = ''

if meansub is True:
    mean_sub_flag = '_meansub'
else:
    mean_sub_flag = ''
processed_dir = os.path.join(os.path.split(imdir)[0], 'processed_%s_reduce%s_%s_%s%s%s%s%s' % (cond, str(reduce_factor[0]), append_to_name, movie_type, detrend_flag, smooth_flag, mean_sub_flag, mean_flag))
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)


files = os.listdir(imdir)
print len(files)
files = sorted([f for f in files if os.path.splitext(f)[1] == str(im_format)])
print len(files)

tiff = TIFF.open(os.path.join(imdir, files[0]), mode='r')
sample = tiff.read_image().astype('float')
print "sample type: %s, range: %s" % (sample.dtype, str([sample.max(), sample.min()]))
print "sample shape: %s" % str(sample.shape)
tiff.close()

# FIND CYCLE STARTS:
if circle:
    positions = [re.findall("\[([^[\]]*)\]", f) for f in files]
    plist = list(itertools.chain.from_iterable(positions))
    pos = []
    for i in plist:
        split_string = i.split(' ')
        split_num = [float(s) for s in split_string if s is not '']
        pos.append([split_num[0], split_num[1]])

    degs = [cart2pol(p[0], p[1], units='deg') for p in pos]

    degrees = [i[0] for i in degs]
    shift_degrees = [i[0] for i in degs]
    for x in range(len(shift_degrees)):
        if shift_degrees[x] < 0:
            shift_degrees[x] += 360.

    if CW:
        find_cycs = list(itertools.chain.from_iterable(np.where(np.diff(shift_degrees) > 0)))
    else:
        find_cycs = list(itertools.chain.from_iterable(np.where(np.diff(shift_degrees) < 0)))
    print "CYC STARTS: ", find_cycs

else:
    # FIND CYCLE STARTS:
    positions = [re.findall("\[([^[\]]*)\]", f) for f in files]
    plist = list(itertools.chain.from_iterable(positions))
    positions = [map(float, i.split(',')) for i in plist]
    if 'H-Up' in cond or 'Bottom' in cond:
        find_cycs = list(itertools.chain.from_iterable(
            np.where(np.diff([p[1] for p in positions]) < 0)))
    if 'H-Down' in cond or 'Top' in cond:
        find_cycs = list(itertools.chain.from_iterable(
            np.where(np.diff([p[1] for p in positions]) > 0)))
    if 'V-Left' in cond or 'Left' in cond:
        find_cycs = list(itertools.chain.from_iterable(
            np.where(np.diff([p[0] for p in positions]) < 0)))
    if 'V-Right' in cond or 'Right' in cond:
        find_cycs = list(itertools.chain.from_iterable(
            np.where(np.diff([p[0] for p in positions]) > 0)))

strt_idxs = [i + 1 for i in find_cycs]
strt_idxs.append(0)
strt_idxs.append(len(positions))
strt_idxs = sorted(strt_idxs)
nframes_per_cycle = [strt_idxs[i] - strt_idxs[i - 1] for i in range(1, len(strt_idxs))]



# strt_idxs = [i+1 for i in find_cycs]
# strt_idxs.append(0)
# strt_idxs = sorted(strt_idxs)
# nframes_per_cycle = [strt_idxs[i] - strt_idxs[i - 1] for i in range(1, len(strt_idxs))]

# Divide into cycles:
# chunks = []
# step = 5
# for i in range(0, len(strt_idxs)-1, step):
#     print i
#     chunks.append(files[strt_idxs[i]:strt_idxs[i+step]])



# INTERPOLATE FRAMES:
ncycles = len(find_cycs) + 1
if interpolate is True:
    print "Interpolating!"
    N = int((ncycles / target_freq) * sampling_rate)
else:
    N = len(files)

FORMAT = '%Y%m%d%H%M%S%f'
datetimes = [f.split('_')[1] for f in files]
tstamps = [float(datetime.datetime.strptime(t, FORMAT).strftime("%H%m%s%f")) for t in datetimes]
actual_tpoints = [(float(i) - float(tstamps[0]))/1E6 for i in tstamps]
tpoints = np.linspace(0, ncycles/target_freq, N)

if interpolate is True:
    moving_win_sz = len(tpoints)/ncycles * 2
    freqs = fft.fftfreq(N, 1 / sampling_rate)
else:
    moving_win_sz = min(nframes_per_cycle)*2
    freqs = fft.fftfreq(len(stack[0, 0, :]), 1 / sampling_rate) # When set fps to 60 vs 120 -- target_bin should be 2x higher for 120, but freq correct (looks for closest matching target_bin )


# READ IN THE FRAMES:
if reduceit:
    sample = block_reduce(sample, reduce_factor, func=np.mean)

tmp_stack = np.empty((sample.shape[0], sample.shape[1], len(files)))
print len(files)

print('copying files')

for i, f in enumerate(files):

    if i % 100 == 0:
        print('%d images processed...' % i)
    tiff = TIFF.open(os.path.join(imdir, f), mode='r')
    im = tiff.read_image().astype('float')
    tiff.close()

    if reduceit:
	if use_sum is True:
	    im_reduced = block_reduce(im, reduce_factor, func=np.sum)
	else:
            im_reduced = block_reduce(im, reduce_factor, func=np.mean)
        # ndimage.gaussian_filter(im_reduced, sigma=gsigma)
        tmp_stack[:, :, i] = im_reduced
    else:
        tmp_stack[:, :, i] = im

    if smooth is True:
        tmp_stack[:,:,i] = ndimage.gaussian_filter(tmp_stack[:,:,i], sigma=sigma_val, order=0)

average_stack = np.mean(tmp_stack, axis=2)


# print "detrending..."

# for x in range(sample.shape[0]):
#     for y in range(sample.shape[1]):

#         # THIS IS BASICALLY MOVING AVG WINDOW...
#         pix = scipy.signal.detrend(stack[x, y, :], type='constant') # HP filter - over time...

#         stack[x, y, :] = pix

if detrend_first is True:
    print "FIRST DETRENDING..."
    for x in range(sample.shape[0]):
        for y in range(sample.shape[1]):
            pix = scipy.signal.detrend(tmp_stack[x, y, :], type='constant')
            tmp_stack[x, y, :] = pix

if meansub is True:
    print "mean subtracting..."
    if mean_subtract_global is True:
        for i in range(stack.shape[2]):
            tmp_stack[:,:,i] -= average_stack
    else:
        for i in range(stack.shape[2]):
            tmp_stack[:,:,i] -= np.mean(tmp_stack[:,:,i].ravel()) 
else:
    print "Not doing a mean subtraction from each frame.  Select option --meansub if this is incorrect."


# if detrend is False:
#     print "detrending..."
stack = np.empty((sample.shape[0], sample.shape[1], N))
for x in range(sample.shape[0]):
    for y in range(sample.shape[1]):

        if interpolate is True:
            pix = np.interp(tpoints, actual_tpoints, tmp_stack[x, y, :])

        # # THIS IS BASICALLY MOVING AVG WINDOW...
        # pix = scipy.signal.detrend(stack[x, y, :], type='constant') # HP filter - over time...

        # THIS IS BASICALLY MOVING AVG WINDOW...
        # curr_pix = scipy.signal.detrend(stack[x, y, :], type='constant') # HP filter - over time...
        if rolling is True:
            pix_padded = [np.ones(moving_win_sz)*pix[0], pix, np.ones(moving_win_sz)*pix[-1]]
            tmp_pix = list(itertools.chain(*pix_padded))
            tmp_pix_rolling = np.convolve(tmp_pix, np.ones(moving_win_sz)/moving_win_sz, 'same')
            remove_pad = (len(tmp_pix_rolling) - len(pix) ) / 2
            rpix = np.array(tmp_pix_rolling[remove_pad:-1*remove_pad])
            pix -= rpix
 
        else:
            if detrend_first is False: # make sure don't do twice:
                print "Now detrending (constant), no rolling avg."
                pix = scipy.signal.detrend(stack[x, y, :], type='constant') # HP filter - over time...

        stack[x, y, :] = pix


print "rescaling to 0-255..."

old_max = stack.max()
old_min = stack.min()
new_max = 255.
new_min = 0.
old_range = float(old_max) - float(old_min)
new_range = 255.
for i in range(stack.shape[2]):

#    old_max = max(stack[:,:,i].ravel())
#    old_min = min(stack[:,:,i].ravel())
#    old_range = (old_max - old_min)  
#    new_min = 0.
#    new_range = 255. #(0 - 255)  
    stack[:,:,i] = (((stack[:,:,i] - old_min) * float(new_range)) / old_range) + new_min
    
#    if smooth is True:
#	stack[:,:,i] = ndimage.gaussian_filter(stack[:,:,i], sigma=sigma_val, order=0)

if smooth is True:
    print "SMOOTHED with val: ", sigma_val	
# for i, f in enumerate(files):


#     # fname = os.path.join(processed_dir, f)
#     old_max = max(stack[:,:,i].ravel())
#     old_min = min(stack[:,:,i].ravel())
#     old_range = (old_max - old_min)  
#     new_min = 0.
#     new_range = 255. #(0 - 255)  
#     stack[:,:,i] = (((stack[:,:,i] - old_min) * float(new_range)) / old_range) + new_min

#     im = Image.fromarray(stack[:,:,i])

#     fname = os.path.join(processed_dir, 'image%05d.png' % i)
#     scipy.misc.imsave(fname, im)



if get_average_cycle:

    print "averaging cycles..."

    idxs = strt_idxs[0:20]
    min_nframes = int(len(tpoints)/ncycles) #min(nframes_per_cycle)
    blocks = []
    for i, s in enumerate(idxs):

        block = stack[:,:,s:s+min_nframes]  #[s:s+min_nframes]
        blocks.append(block)

    average_cycle = sum(blocks) / len(blocks)

    print "saving averaged frames..."

    for i in range(average_cycle.shape[2]):
	print i
        im = Image.fromarray(average_cycle[:,:,i])

        fname = os.path.join(processed_dir, 'image%04d.png' % i)
        scipy.misc.imsave(fname, im)
else:

    print "saving processed frames..."

    for i in range(stack.shape[2]):

        im = Image.fromarray(stack[:,:,i])

        fname = os.path.join(processed_dir, 'image%05d.png' % i)
        scipy.misc.imsave(fname, im)

print "Done."
print "Saved to: ", fname
    # tiff = TIFF.open(fname, mode='w')
    # tiff.write_image(stack[:,:,i])
    # tiff.close()

# Image.fromarray(imarray)
# os.system("ffmpeg -f image2 -r 0.02 -i image%05d.tif -vcodec mpeg4 -y movie.mp4")

# # --CREATE MOVIE FRAMES IF NEED TO--
# if make_frames is True:
#     v_fps = 60 #0.08 # -r :n frames to extract per second
#     # movie = source_dir + 'natural_movie.avi' # -i : read from input files, and write to outputfiles
#     movie = source_file
#     v_fmt = 'image2' # 'image2' # -f : format of input 
#     # v_size = '128x96'
#     # v_opts = "-i %s -r %s -f %s -s %s" % (movie, str(v_fps), v_fmt, v_size)
#     v_opts = "-i %s -r %s -f %s" % (movie, str(v_fps), v_fmt)
#     os.system("ffmpeg "+ v_opts + " " + stimdir + "/%4d.png")
#     # os.system("ffmpeg -i natural_movie.avi -r 0.08 -f image2 -s 128x96 %4d.png")

