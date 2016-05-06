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
parser.add_option('--sigma', action="store", dest="gauss_kernel",
                  default="0", help="size of Gaussian kernel for smoothing")
parser.add_option('--format', action="store",
                  dest="im_format", default="png", help="saved image format")
parser.add_option('--fps', action="store",
                  dest="sampling_rate", default="60", help="saved image format")
parser.add_option('--append', action="store",
                  dest="append_name", default="", help="append string to saved file name")


(options, args) = parser.parse_args()

imdir = sys.argv[1]

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
    find_cycs = list(itertools.chain.from_iterable(np.where(np.diff(degrees) < 0)))


strt_idxs = [i+1 for i in find_cycs]
strt_idxs.append(0)
strt_idxs = sorted(strt_idxs)
nframes_per_cycle = [strt_idxs[i] - strt_idxs[i - 1] for i in range(1, len(strt_idxs))]

# Divide into cycles:
# chunks = []
# step = 5
# for i in range(0, len(strt_idxs)-1, step):
#     print i
#     chunks.append(files[strt_idxs[i]:strt_idxs[i+step]])



# READ IN THE FRAMES:
if reduceit:
    sample = block_reduce(sample, reduce_factor, func=np.mean)

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
print "FREQS: ", freqs

# freqs_shift = fft.fftshift(freqs)
# target_bin_shift = np.where(freqs_shift == min(
#     freqs_shift, key=lambda x: abs(float(x) - target_freq)))[0][0]
# print "TARGET-shift: ", target_bin_shift, freqs_shift[target_bin_shift]
# print "FREQS-shift: ", freqs_shift

window = sampling_rate * cycle_dur * 2

# FFT:
mag_map = np.empty(sample.shape)
phase_map = np.empty(sample.shape)

ft = np.empty(sample.shape)
ft = ft + 0j

dynrange = np.empty(sample.shape)

# dlist = []
i = 0
for x in range(sample.shape[0]):
    for y in range(sample.shape[1]):

        # THIS IS BASICALLY MOVING AVG WINDOW...
        pix = scipy.signal.detrend(stack[x, y, :], type='constant') # HP filter - over time...

        dynrange[x, y] = np.log2(pix.max() - pix.min())

        curr_ft = fft.fft(pix)  # fft.fft(pix) / len(pix)])
        #curr_ft_shift = fft.fftshift(curr_ft)

# 		dlist.append((x, y, curr_ft))
# 		i+=1

# DF = pd.DataFrame.from_records(dlist)

        mag = np.abs(curr_ft)
        phase = np.angle(curr_ft)

        ft[x, y] = curr_ft[target_bin]

        mag_map[x, y] = mag[target_bin]
        phase_map[x, y]  = phase[target_bin]
        # dlist.append((x, y, curr_ft))

        i += 1

D = dict()

D['ft'] = ft

D['mag_map'] = mag_map
D['phase_map'] = phase_map
D['mean_intensity'] = np.mean(stack, axis=2)
D['dynrange'] = dynrange
D['target_freq'] = target_freq
D['fps'] = sampling_rate
D['freqs'] = freqs  # fft.fftfreq(len(pix), 1 / sampling_rate)
D['positions'] = positions
D['degrees'] = degrees
D['degrees'] = shift_degrees
D['strt_idxs'] = strt_idxs

D['binsize'] = freqs[1] - freqs[0]
D['target_bin'] = target_bin
D['nframes_per_cycle'] = nframes_per_cycle
D['reduce_factor'] = reduce_factor

# SAVE condition info:
sessionpath = os.path.split(imdir)[0]
outdir = os.path.join(sessionpath, 'structs')
if not os.path.exists(outdir):
    os.makedirs(outdir)

fext = 'Target_fft_%s_%s_%s.pkl' % (cond, str(reduce_factor), append_to_name)
fname = os.path.join(outdir, fext)
with open(fname, 'wb') as f:
    # protocol=pkl.HIGHEST_PROTOCOL)
    pkl.dump(D, f, protocol=pkl.HIGHEST_PROTOCOL)

del D


# SAVE THE FFT, USE .hkl SINCE HUGE...??
# D = dict()
# D['ft'] = DF
# fext = 'Full_fft_%s_%s.pkl' % (cond, str(reduce_factor))
# fname = os.path.join(outdir, fext)
# with open(fname, 'wb') as f:
#     # protocol=pkl.HIGHEST_PROTOCOL)
#     pkl.dump(D, f)





# print target_bin, DC_bin

# if binspread != 0:
# mag_map[x, y] = 20*np.log10(np.mean(mag[target_bin-binspread:target_bin+binspread]))
# mag_map[x, y] = np.mean(mag[target_bin-binspread:target_bin+binspread+1] / mag[0])
# phase_map[x, y] = np.mean(phase[target_bin-binspread:target_bin+binspread])
# else:
# mag_map[x, y] = 20*np.log10(mag[target_bin])
# mag_map[x,y] = mag[target_bin] # / mag[DC_bin]
# mag_map[x,y] = mag[target_bin]
# phase_map[x, y] = phase[target_bin]


# if x % int(sample.shape[0] / 4) == 0 or y % int(sample.shape[1] / 4) == 0:
# plt.subplot(2,1,1)
# plt.plot(freqs, mag, '*')
# plt.subplot(2,1,2)
# plt.plot(freqs, phase, '*')
# plt.show()


# if binspread != 0:
# mag_map[x, y] = 20*np.log10(np.mean(mag[target_bin-binspread:target_bin+binspread]))
# phase_map[x, y] = np.mean(phase[target_bin-binspread:target_bin+binspread])
# else:
# mag_map[x, y] = 20*np.log10(mag[target_bin])
# mag_map[x,y] = mag[target_bin]/mag[0.]
# mag_map[x,y] = mag[target_bin]
# phase_map[x, y] = phase[target_bin]

# PLOT IT:

# basepath = os.path.split(os.path.split(imdir)[0])[0]
# session = os.path.split(os.path.split(imdir)[0])[1]
# cond = os.path.split(imdir)[1]

# plt.subplot(2,2,1) # GREEN LED image
# outdir = os.path.join(basepath, 'output')
# if os.path.exists(outdir):
# 	flist = os.listdir(outdir)
# GET BLOOD VESSEL IMAGE:
# 	ims = [f for f in flist if os.path.splitext(f)[1] == '.png']
# 	if ims:
# 		impath = os.path.join(outdir, ims[0])
# 		image = Image.open(impath).convert('L')
# 		imarray = np.asarray(image)

# 		plt.imshow(imarray,cmap=cm.Greys_r)
# 	else:
# 		print "*** Missing green-LED photo of cortex surface. ***"
# else:
# 	spnum = 2

# plt.subplot(2, 2, 2)
# fig =  plt.imshow(dynrange)
# plt.title('Dynamic range (bits)')
# plt.colorbar()


# plt.subplot(2, 2, 3)
# mag_map = mag_map*1E4
# fig =  plt.imshow(np.clip(mag_map, 0, mag_map.max()), cmap=cm.hot)
# fig = plt.imshow(np.clip(mag_map, 0, mag_map.max()), cmap = plt.get_cmap('gray'), vmin = 0, vmax = 1.0)
# fig = plt.imshow(mag_map, cmap = plt.get_cmap('gray'))#mag_map.max())
# plt.title('Magnitude @ %0.3f' % (freqs[round(target_bin)]))
# fig.set_cmap("hot")
# plt.colorbar()


# plt.subplot(2, 2, 4)
# fig = plt.imshow(phase_map)
# plt.title('Phase (rad) @ %0.3f' % freqs[round(target_bin)])
# fig.set_cmap("spectral")
# plt.colorbar()

# plt.suptitle(session + ': ' + cond)

# plt.show()

# SAVE FIG
# figdir = os.path.join(basepath, 'figures', session, 'fieldmap')
# print figdir
# if not os.path.exists(figdir):
# 	os.makedirs(figdir)
# imname = session + '_' + cond + '_fieldmap' + str(reduce_factor) + '.png'
# plt.savefig(figdir + '/' + imname)

# plt.show()


# SAVE MAPS:
# outdir = os.path.join(basepath, 'output', session)
# if not os.path.exists(outdir):
# 	os.makedirs(outdir)

# fext = 'magnitude_%s_%s_%i.pkl' % (cond, str(reduce_factor), gsigma)
# fname = os.path.join(outdir, fext)
# with open(fname, 'wb') as f:
# pkl.dump(mag_map, f, protocol=pkl.HIGHEST_PROTOCOL)
# protocol=pkl.HIGHEST_PROTOCOL)

# fext = 'phase_%s_%s_%i.pkl' % (cond, str(reduce_factor), gsigma)
# fname = os.path.join(outdir, fext)
# with open(fname, 'wb') as f:
# pkl.dump(phase_map, f, protocol=pkl.HIGHEST_PROTOCOL)
# protocol=pkl.HIGHEST_PROTOCOL)


# fext = 'dynrange_%s_%s_%i.pkl' % (cond, str(reduce_factor), gsigma)
# fname = os.path.join(outdir, fext)
# with open(fname, 'wb') as f:
# pkl.dump(dynrange, f, protocol=pkl.HIGHEST_PROTOCOL)
# protocol=pkl.HIGHEST_PROTOCOL)
