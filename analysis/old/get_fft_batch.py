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

import time
import datetime

from multiprocessing import Process

#import hickle as hkl

def movingaverage(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'valid')


def get_fft(run_path):
    
    files = os.listdir(run_path)
    print len(files)
    files = sorted([f for f in files if os.path.splitext(f)[1] == str(im_format)])
    print len(files)
    cond = os.path.split(run_path)[1]

    tiff = TIFF.open(os.path.join(run_path, files[0]), mode='r')
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
    if 'Left' in cond or 'Blank' in cond:
        find_cycs = list(itertools.chain.from_iterable(
            np.where(np.diff([p[0] for p in positions]) < 0)))
    if 'Right' in cond:
        find_cycs = list(itertools.chain.from_iterable(
            np.where(np.diff([p[0] for p in positions]) > 0)))
    # print find_cycs
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


    # INTERPOLATE FRAMES:
    ncycles = len(find_cycs) + 1
    N = int(round((ncycles / target_freq) * sampling_rate))

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


    # SET FFT PARAMETERS:
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

    window = sampling_rate * cycle_dur * 2


    # READ IN THE FRAMES:times
    if motion_corrected is True:
        motion_dir = os.path.join(run_path, 'mCorrected', 'Motion', 'Registration')
        mfiles = os.listdir(motion_dir)
        mfiles = [m for m in mfiles if '_correctedFrames.npz' in m and run in m]
        data = np.load(os.path.join(motion_dir, mfiles[0]))
        stack = data['correctedFrameArray']
        sample = stack[:,:,0]
        print "Stack is: ", stack.shape
    else:
        #    
        stack = np.empty((sample.shape[0], sample.shape[1], len(files)))
        print len(files)

        print('copying files')

        for i, f in enumerate(files):

            if i % 100 == 0:
                print('%d images processed...' % i)
            tiff = TIFF.open(os.path.join(run_path, f), mode='r')
            im = tiff.read_image().astype('float')
            tiff.close()

            if reduceit:
                im_reduced = block_reduce(im, reduce_factor, func=np.mean)
                # ndimage.gaussian_filter(im_reduced, sigma=gsigma)
                stack[:, :, i] = im_reduced
            else:
                stack[:, :, i] = im

    average_stack = np.mean(stack, axis=2)


    # FFT:
    mag_map = np.empty(sample.shape)
    phase_map = np.empty(sample.shape)
    sum_all_mags = np.empty(sample.shape)
    mag_other_freqs = np.empty(sample.shape)
    ratio_map = np.empty(sample.shape)

    # ft_real = np.empty(sample.shape)
    # ft_imag = np.empty(sample.shape)

    ft = np.empty(sample.shape)
    ft = ft + 0j

    DC_mag = np.empty(sample.shape)
    DC_phase = np.empty(sample.shape)

    DC = np.empty(sample.shape)
    DC = DC + 0j

    dynrange = np.empty(sample.shape)

    i = 0
    for x in range(sample.shape[0]):
        for y in range(sample.shape[1]):

            if interpolate is True:
                pix = np.interp(tpoints, actual_tpoints, stack[x, y, :])

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
               pix = scipy.signal.detrend(stack[x, y, :], type='constant') # HP filter - over time...

            dynrange[x, y] = np.log2(pix.max() - pix.min())

            curr_ft = fft.fft(pix) #*(1./60.)  # fft.fft(pix) / len(pix)])

            mag = np.abs(curr_ft)
            phase = np.angle(curr_ft)
            ft[x, y] = curr_ft[target_bin]

            mag_map[x, y] = mag[target_bin] + mag[int(N) - target_bin]
            phase_map[x, y]  = phase[target_bin]

            sum_all_mags[x, y] = sum(mag) 
            mag_other_freqs[x, y] = sum(mag) - mag[DC_bin]
            # ratio_map[x, y] = (mag[target_bin]+mag[int(N)-target_bin]) / mag_other_freqs[x, y]
               
            ratio_map[x, y] = (mag[target_bin]*2.) / mag_other_freqs[x, y]
                
            DC[x, y] = curr_ft[DC_bin]
            DC_mag[x, y] = mag[DC_bin]
            DC_phase[x, y]  = phase[DC_bin]

            i += 1

    D = dict()
    D['ft'] = ft
    D['mag_map'] = mag_map
    D['phase_map'] = phase_map

    D['sum_all_mags'] = sum_all_mags
    D['mag_other_freqs'] = mag_other_freqs
    D['ratio_map'] = ratio_map

    D['mean_intensity'] = np.mean(stack, axis=2)
    D['dynrange'] = dynrange
    D['target_freq'] = target_freq
    D['fps'] = sampling_rate
    D['freqs'] = freqs  # fft.fftfreq(len(pix), 1 / sampling_rate)

    D['binsize'] = freqs[1] - freqs[0]
    D['nframes'] = nframes_per_cycle
    D['reduce_factor'] = reduce_factor

    D['DC_bin'] = DC_bin
    D['DC_freq'] = DC_freq
    D['DC'] = DC
    D['DC_mag'] = DC_mag
    D['DC_phase'] = DC_phase

    D['meansub'] = meansub
    D['interpolated'] = interpolate
    D['rolling'] = rolling

    # SAVE condition info:
    sessionpath = os.path.split(run_path)[0]
    outdir = os.path.join(sessionpath, 'structs')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fext = 'Target_fft_%s_%s_%s.pkl' % (cond, str(reduce_factor), append_to_name)
    fname = os.path.join(outdir, fext)
    with open(fname, 'wb') as f:
        # protocol=pkl.HIGHEST_PROTOCOL)
        pkl.dump(D, f, protocol=pkl.HIGHEST_PROTOCOL)

    del D


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

parser.add_option('--rolling', action='store_true', default=False, help="Rolling average [window size is 2 cycles] or detrend.")
parser.add_option('--meansub', action='store_true', default=False, help="Remove mean of each frame.")
parser.add_option('--interpolate', action='store_true', default=False, help='Interpolate frames or no.')

parser.add_option('--path', action="store",
                  dest="session_path", default="", help="input dir")

parser.add_option('--ncycles', action="store",
                  dest="ncycles", default=20, help="ncycles (default 20)")
parser.add_option('--motion', action='store_true', dest="motion_corrected", default=False, help="Motion corrected with WiPy or no?")


(options, args) = parser.parse_args()
motion_corrected = options.motion_corrected

#imdir = sys.argv[1]
session_path = options.session_path
#imdirs = [sys.argv[1], sys.argv[2]]
interpolate = options.interpolate
rolling = options.rolling
meansub = options.meansub

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
imformat = str(im_format)
session = os.path.split(session_path)[0]
all_runs = os.listdir(session_path)
run_list = [r for r in all_runs if '_run' in r and 'processed' not in r]

print "Found %i runs to process:" % len(run_list)
for run in run_list:
    print run
for run in run_list:
    Process(target=get_fft, args=(os.path.join(session_path, run), )).start()
    print "Started process for run -- %s" % run


