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
parser.add_option('--circle', action="store_true", dest="circle",
                  default=False, help="is the stimulus a circle or bar?")
parser.add_option('--CW', action="store_true", dest="CW",
                  default=False, help="circle stim ONLY: CW or not?")
parser.add_option('--average', action="store_true", dest="get_average_cycle",
                  default=False, help="average cycles or no?")

(options, args) = parser.parse_args()

imdir = sys.argv[1]

# processed_dir = os.path.join(os.path.split(imdir)[0], 'processed')

# if not os.path.exists(processed_dir):
#     os.makedirs(processed_dir)


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

if get_average_cycle:
    movie_type = 'avgcycle'
else:
    movie_type = 'all'
processed_dir = os.path.join(os.path.split(imdir)[0], 'processed_%s_%s' % (cond, movie_type))
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

else:
    # FIND CYCLE STARTS:
    positions = [re.findall("\[([^[\]]*)\]", f) for f in files]
    plist = list(itertools.chain.from_iterable(positions))
    positions = [map(float, i.split(',')) for i in plist]
    if 'H-Up' in cond:
        find_cycs = list(itertools.chain.from_iterable(
            np.where(np.diff([p[1] for p in positions]) < 0)))
    if 'H-Down' in cond:
        find_cycs = list(itertools.chain.from_iterable(
            np.where(np.diff([p[1] for p in positions]) > 0)))
    if 'V-Left' in cond:
        find_cycs = list(itertools.chain.from_iterable(
            np.where(np.diff([p[0] for p in positions]) < 0)))
    if 'V-Right' in cond:
        find_cycs = list(itertools.chain.from_iterable(
            np.where(np.diff([p[0] for p in positions]) > 0)))

strt_idxs = [i + 1 for i in find_cycs]
strt_idxs.append(0)
strt_idxs.append(len(positions))
strt_idxs = sorted(strt_idxs)
nframes_per_cycle = [strt_idxs[i] - strt_idxs[i - 1] for i in range(1, len(strt_idxs))]


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


print "mean subtracting..."
for i in range(stack.shape[2]):
    stack[:,:,i] -= np.mean(stack[:,:,i].ravel()) 


print "detrending..."

for x in range(sample.shape[0]):
    for y in range(sample.shape[1]):

        # THIS IS BASICALLY MOVING AVG WINDOW...
        pix = scipy.signal.detrend(stack[x, y, :], type='constant') # HP filter - over time...

        stack[x, y, :] = pix


# save stack:
D=dict()
D['stack'] = stack
print D.keys()

fext = '%s_%s_stack.pkl' % (cond, str(reduce_factor))
fname = os.path.join(os.path.split(imdir)[0], fext)
print fname

with open(fname, 'wb') as f:
    # protocol=pkl.HIGHEST_PROTOCOL)
    pkl.dump(D, f)


# print "rescaling to 0-255..."

# for i in range(stack.shape[2]):

#     old_max = max(stack[:,:,i].ravel())
#     old_min = min(stack[:,:,i].ravel())
#     old_range = (old_max - old_min)  
#     new_min = 0.
#     new_range = 255. #(0 - 255)  
#     stack[:,:,i] = (((stack[:,:,i] - old_min) * float(new_range)) / old_range) + new_min

import numpy.fft as fft

fig = plt.figure()

x=80
y=120

pix = stack[x,y,:]

# intensity over time
t = range(len(pix))
plt.plot(t, pix)
for i in strt_idxs:
	plt.axvline(i,color='r')
plt.xlabel('frames')
plt.ylabel('intensity')
plt.title('sample pixel (%i, %i)' % (x, y))

plt.show()


# # sample params:
# N = len(pix)
# target_freq = 0.13
# sampling_rate = 60.
# dt = 1 / sampling_rate

# time = 1 / sampling_rate * np.arange(N)
# freqs = fft.fftfreq(len(pix), 1 / sampling_rate)


# Ny = len(pix)/2+1

# plt.plot(np.abs(ft[0:N/2]))
# # time axes:
# # time = 1 / sampling_rate * np.arange(N)
# # freqs = fft.fftfreq(len(pix), 1 / sampling_rate)

# # fft for target frequency:
# target_bin = np.where(freqs == min(freqs, key=lambda x: abs(float(x) - target_freq)))[0][0]
# ft = fft.fft(pix)
# # ft_scaled = ft / dt



# # PLOT:  amplitude spectrum (or power)
# plt.plot(freqs, np.abs(ft))
# # plt.plot(freqs, np.abs(ft)/N) # normalize?
# plt.xlabel('frequencies (Hz)')
# plt.ylabel('amplitude ')


# # PLOT:  only show 1/2, limit to Nyquist:
# plt.plot(np.abs(ft[0:Ny]))
# plt.plot(freqs[0:Ny], 2*np.abs(ft[0:Ny])/Ny)
# plt.plot(freqs[target_bin], 2*np.abs(ft[target_bin])/Ny, 'r*')
# # plt.xlim([0,Ny])


# # Hanning window?  wtf units...
# hann = np.hanning(len(pix))
# plt.plot(time, hann*pix)
# ft_hann = fft.fft(hann*pix)
# plt.plot(freqs[0:Ny], 2*np.abs(Yhann[0:Ny])/Ny)

# # http://www.cbcity.de/die-fft-mit-python-einfach-erklaert




# # PLOT:  POWER?
# plt.figure(); plt.plot(freqs, np.abs(ft)**2)

# # normalization = 2 / N
# # plt.plot(freqs[:N // 2], normalization * np.abs(ft[:N // 2]))# magnitude maps...



# # divide by DC component at each pixel:

# DC = np.empty(stack[:,:,1].shape)
# # DC = DC + 0j
# for x in range(stack.shape[0]):
# 	for y in range(stack.shape[1]):
# 		ft = fft.fft(stack[x,y,:])
# 		DC[x,y] = np.abs(ft[0])



# outdir = sys.argv[1]
# files = os.listdir(outdir)


# files = sorted([f for f in files if '_fft' in f]) # get giant FFT file for all runs 
# fname = os.path.join(outdir, files[0]) # choose particular run condition
# with open(fname, 'rb') as f:
# 	F = pkl.load(f)

# curr_file = os.path.split(fname)[1] # get name of .pkl file for particular condition
# curr_cond = str.split(curr_file, '_')[2] # get specific run condition
# curr_run = str.split(curr_file, '_')[3] # get run number

# parenth = re.compile("\((.+)\)") # find pattern of "( xxxx )"
# m = parenth.search(curr_file).group(1)
# reduce_value = [int(i) for i in m[0]][0] # just use the first num 
# n_x = len(set(F['ft'][0])) # number of pixels HEIGHT (164, if bin 3 and reduce=1)
# n_y = len(set(F['ft'][1])) # number of pixels WIDTH (218, if bin 3 and reduce=1)
# n_pixels = n_x * n_y

# files = os.listdir(outdir)
# files = sorted([f for f in files if '_target' in f and curr_cond in f]) # get matching run info
# fname = os.path.join(outdir, files[0]) # look at first matching run condition
# with open(fname, 'rb') as f:
# 	D = pkl.load(f)


# magnitudes = [np.abs(F['ft'][2][p]) for p in range(n_pixels)]
# freqs = D['freqs']


# fig = plt.figure()

# # for x in range(n_x):
# # 	for y in range(n_y):

# # GOOD: 35722 -- F['ft'][0][35722] = 163 ("x") | F['ft'][1][35722] = 188 ("y")

# pidx = 0
# for i in xrange(35650, 35700, 1):#range(n_pixels):

# 	fig.add_subplot(5, 10, pidx) 
# 	plt.plot(freqs[0:int(len(freqs)*.25)], magnitudes[i][0:int(len(freqs)*.25)])
# 	plt.plot(freqs[D['target_bin']], magnitudes[i][D['target_bin']], 'r*')
# 	pidx += 1