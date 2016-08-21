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

#import hickle as hkl

def movingaverage(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'valid')

parser = optparse.OptionParser()
parser.add_option('--headless', action="store_true", dest="headless",
                  default=False, help="run in headless mode, no figs")
parser.add_option('--freq', action="store", dest="target_freq",
                  default="0.05", help="stimulation frequency")
parser.add_option('--reduce', action="store",
                  dest="reduce_val", default="2", help="block_reduce value")
parser.add_option('--sigma', action="store", dest="gauss_kernel",
                  default="0", help="size of Gaussian kernel for smoothing")
parser.add_option('--format', action="store",
                  dest="im_format", default="png", help="saved image format")
parser.add_option('--fps', action="store",
                  dest="sampling_rate", default="60", help="saved image format")
parser.add_option('--append', action="store",
                  dest="append_name", default="", help="append string to saved file name")
parser.add_option('--plot', action="store_true", dest="plot_sample", default=False, help="plots sample at 100,100")



(options, args) = parser.parse_args()

imdir = sys.argv[1]

plot_sample = options.plot_sample

#imdirs = [sys.argv[1], sys.argv[2]]

im_format = '.' + options.im_format
headless = options.headless
target_freq = float(options.target_freq)
reduce_factor = (int(options.reduce_val), int(options.reduce_val))
if reduce_factor[0] > 0:
    reduceit = 1
else:
    reduceit = 0
gsigma = int(options.gauss_kernel)

if headless:
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pylab as plt
import matplotlib.cm as cm

sampling_rate = float(options.sampling_rate) # 60.  # np.mean(np.diff(sorted(strt_idxs)))/cycle_dur #60.0
cache_file = True
cycle_dur = 1. / target_freq  # 10.
binspread = 0

#stacks = dict()
# for imdir in imdirs:
append_to_name = str(options.append_name)

basepath = os.path.split(os.path.split(imdir)[0])[0]
session = os.path.split(os.path.split(imdir)[0])[1]
cond = os.path.split(imdir)[1]

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
positions = [re.findall("\[([^[\]]*)\]", f) for f in files]
plist = list(itertools.chain.from_iterable(positions))
positions = [map(float, i.split(',')) for i in plist]
print "Curr COND: ",  cond
if 'Up' in cond or 'Bottom' in cond:
    print 'UP'
    find_cycs = list(itertools.chain.from_iterable(
        np.where(np.diff([p[1] for p in positions]) < 0)))
if 'Down' in cond or 'Top' in cond:
    find_cycs = list(itertools.chain.from_iterable(
        np.where(np.diff([p[1] for p in positions]) > 0)))
if 'Left' in cond:
    find_cycs = list(itertools.chain.from_iterable(
        np.where(np.diff([p[0] for p in positions]) < 0)))
if 'Right' in cond:
    find_cycs = list(itertools.chain.from_iterable(
        np.where(np.diff([p[0] for p in positions]) > 0)))
print find_cycs
# idxs = [i + 1 for i in find_cycs]
# idxs.append(0)
# idxs.append(len(positions))
# idxs = sorted(idxs)

strt_idxs = [i + 1 for i in find_cycs]
strt_idxs.append(0)
strt_idxs.append(len(positions))
strt_idxs = sorted(strt_idxs)

nframes_per_cycle = [strt_idxs[i] - strt_idxs[i - 1] for i in range(1, len(strt_idxs))]
print "N frames per cyc: ", nframes_per_cycle


if reduceit:
    sample = block_reduce(sample, reduce_factor, func=np.mean)

# READ IN THE FRAMES:
stack = np.empty((sample.shape[0], sample.shape[1], len(files)))
print len(files)

print('copying files')

for i, f in enumerate(files):

    if i % 100 == 0:
        print('%d images processed...' % i)
    tiff = TIFF.open(os.path.join(imdir, f), mode='r')
    im = tiff.read_image().astype('float')
    tiff.close()

    if reduceit:
        im_reduced = block_reduce(im, reduce_factor, func=np.mean)
        # ndimage.gaussian_filter(im_reduced, sigma=gsigma)
        stack[:, :, i] = im_reduced
    else:
        stack[:, :, i] = im

average_stack = np.mean(stack, axis=2)

for i in range(stack.shape[2]):
    stack[:,:,i] -= np.mean(stack[:,:,i].ravel()) # HP filter - This step removes diff value for each frame, and shifts the range of intensities to span around 0.
    # stack[:,:,i] -= np.mean(average_stack.ravel()) # This step subtracts the same value ALL frames, effectively shifting the range down by the same amount.

#stacks[session] = stack

# SET FFT PARAMETERS:
freqs = fft.fftfreq(len(stack[0, 0, :]), 1 / sampling_rate) # When set fps to 60 vs 120 -- target_bin should be 2x higher for 120, but freq correct (looks for closest matching target_bin )
binwidth = freqs[1] - freqs[0]
#target_bin = int(target_freq / binwidth)
target_bin = np.where(
    freqs == min(freqs, key=lambda x: abs(float(x) - target_freq)))[0][0]
print "TARGET: ", target_bin, freqs[target_bin]
# print "FREQS: ", freqs

DC_freq = 0
DC_bin = np.where(
    freqs == min(freqs, key=lambda x: abs(float(x) - DC_freq)))[0][0]
print "DC: ", DC_freq, freqs[DC_bin]


freqs_shift = fft.fftshift(freqs)
target_bin_shift = np.where(freqs_shift == min(
    freqs_shift, key=lambda x: abs(float(x) - target_freq)))[0][0]
print "TARGET-shift: ", target_bin_shift, freqs_shift[target_bin_shift]
print "FREQS-shift: ", freqs_shift


window = sampling_rate * cycle_dur * 2

magnitudes = np.empty((sample.shape[0]*sample.shape[1], len(freqs))) # each row will be 1 pixel's magnitude

i = 0
for x in range(sample.shape[0]):
    for y in range(sample.shape[1]):

        # THIS IS BASICALLY MOVING AVG WINDOW...
        pix = scipy.signal.detrend(stack[x, y, :], type='constant') # HP filter - over time...

        # dynrange[x, y] = np.log2(pix.max() - pix.min())

        curr_ft = fft.fft(pix)

        magnitudes[i,:] = np.abs(curr_ft)

        i += 1

if plot_sample is True:
    x = 100
    y = 100
    plt.plot(freqs, magnitudes[100,:], 'k')
    plt.plot(0, magnitudes[100, target_bin], 'r*')


D = dict()
D['magnitudes'] = magnitudes
D['freqs'] = freqs
D['target_bin'] = target_bin
D['target'] = freqs[target]
D['sampling_rate'] = sampling_rate
D['freqs_shift'] = freqs_shift
D['target_bin_shift'] = target_bin_shift


sessionpath = os.path.split(imdir)[0]
outdir = os.path.join(sessionpath, 'structs')
if not os.path.exists(outdir):
    os.makedirs(outdir)

fext = 'power_%s_%s_%s.pkl' % (cond, str(reduce_factor), append_to_name)
fname = os.path.join(outdir, fext)
with open(fname, 'wb') as f:
    # protocol=pkl.HIGHEST_PROTOCOL)
    pkl.dump(D, f, protocol=pkl.HIGHEST_PROTOCOL)

del D

print "done"

