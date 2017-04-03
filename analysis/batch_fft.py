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


def average_runs(session_path, condition):
    runs = os.listdir(session_path)
    runs = [run for run in runs if os.path.isdir(os.path.join(session_path,run)) and condition in run]
    print "Found %i runs to average:" % len(runs)
    
    fnames = dict()
    tpoints = dict()
    for ridx,run in enumerate(runs):
        tpoints[run] = dict()
        print ridx, run
        tmp_files = os.listdir(os.path.join(session_path, run))
        tmp_files = [f for f in tmp_files if f.endswith(imformat)]
        fnames[run] = tmp_files

        tmp_N, tmp_ncycles, tmp_nframes_per_cycle, tmp_strt_idxs, tmp_ts, tmp_actual_tpoints = get_tpoints(os.path.join(session_path, run))
        tpoints[run]['ts'] = tmp_ts
        tpoints[run]['actual_ts'] = tmp_actual_tpoints
        tpoints[run]['N'] = tmp_N
        tpoints[run]['ncycles'] = tmp_ncycles
        tpoints[run]['nframes_per_cycle'] = tmp_nframes_per_cycle
        tpoints[run]['start_idxs'] = tmp_strt_idxs
        print len(tpoints[run]['actual_ts'])
    
    tiff = TIFF.open(os.path.join(session_path, run, tmp_files[0]), mode='r')
    sample = tiff.read_image().astype('float')
    tiff.close()
    expected_nframes = sum(np.array([min([tpoints[run]['nframes_per_cycle'][n] for run in runs]) for n in range(tmp_ncycles)]))        
    print "Expected frame count for averaged run: %i" % expected_nframes
    stack = np.zeros((sample.shape[0], sample.shape[1], expected_nframes))
    curr_sidx = 0

    AVG = dict()
    AVG['nframes_per_cycle'] = []
    AVG['ts'] = []
    AVG['actual_ts'] = []
    for n in range(tmp_ncycles):
        curr_cycle_nframes = [tpoints[run]['nframes_per_cycle'][n] for run in runs]
        curr_cycle_minframes = min(curr_cycle_nframes)
        ref_run = [run for run in runs if tpoints[run]['nframes_per_cycle'][n]==curr_cycle_minframes][0]
        
        curr_cycle_stack = np.zeros((sample.shape[0], sample.shape[1], curr_cycle_minframes))
        curr_ts = []
        curr_actual_ts = []
        for fidx,frame in enumerate(np.arange(curr_sidx, curr_sidx+curr_cycle_minframes)):

            curr_frame_avg = np.zeros((curr_cycle_stack.shape[0], curr_cycle_stack.shape[1], len(runs)))
            tmp_curr_ts = []
            tmp_curr_actual_ts = []
            for ridx,run in enumerate(runs):
                curr_cyc_idx = tpoints[run]['start_idxs'][n]
                print curr_cyc_idx
                curr_frame_avg[:,:,ridx] = imread(os.path.join(session_path, run, fnames[run][curr_cyc_idx+fidx]))
                tmp_curr_actual_ts.append(tpoints[run]['actual_ts'][curr_cyc_idx+fidx])
            
            curr_cycle_stack[:,:,fidx] = np.mean(curr_frame_avg,2)
            #curr_ts.append(np.mean(np.array(tmp_curr_ts)))
            curr_actual_ts.append(np.mean(np.array(tmp_curr_actual_ts)))

        stack[:,:,curr_sidx:curr_sidx+curr_cycle_minframes] = curr_cycle_stack  
        curr_sidx += curr_cycle_minframes 
        AVG['nframes_per_cycle'].append(curr_cycle_minframes)
        #AVG['ts'].append(tpoints[ref_run]['ts'][curr_sidx:curr_sidx+curr_cycle_minframes])
        #AVG['actual_ts'].append(tpoints[ref_run]['actual_ts'][curr_sidx:curr_sidx+curr_cycle_minframes])
        #AVG['ts'].append(curr_ts)
        AVG['actual_ts'].append(curr_actual_ts)
    print AVG['nframes_per_cycle']

    avg_nframes_per_cycle = AVG['nframes_per_cycle']
    #avg_tstamps = list(itertools.chain.from_iterable(AVG['ts']))
    avg_actual_ts = list(itertools.chain.from_iterable(AVG['actual_ts']))    

    return stack, tpoints[ref_run]['N'], tpoints[ref_run]['ncycles'], avg_nframes_per_cycle, avg_actual_ts   


def process_averaged_run(session_path, condition, sample_rate, target_freq, append_to_name):


    stack, N, ncycles, nframes_per_cycle, actual_tpoints = average_runs(session_path, condition)
    curr_cond_name = condition+'_avg'
  
    D = get_fft(stack, sample_rate, target_freq, N, ncycles, nframes_per_cycle, actual_tpoints)

    outdir = os.path.join(session_path, 'structs')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fext = 'Target_fft_%s_%s_%s.pkl' % (curr_cond_name, str(reduce_factor), append_to_name)
    fname = os.path.join(outdir, fext)
    with open(fname, 'wb') as f:
        # protocol=pkl.HIGHEST_PROTOCOL)
        pkl.dump(D, f, protocol=pkl.HIGHEST_PROTOCOL)



def get_fft(stack, sample_rate, target_freq, N, ncycles, nframes_per_cycle, actual_tpoints):
    
    tpoints = np.linspace(0, (ncycles/target_freq), N)
    if interpolate is True:
        moving_win_sz = len(tpoints)/ncycles * 2
        freqs = fft.fftfreq(N, 1 / sample_rate)
    else:
        moving_win_sz = min(nframes_per_cycle)*2
        freqs = fft.fftfreq(len(stack[0, 0, :]), 1 / sample_rate) # When set fps to 60 vs 120 -- target_bin should be 2x higher for 120, but freq correct (looks for closest matching target_bin )
    
    target_bin = np.where(
        freqs == min(freqs, key=lambda x: abs(float(x) - target_freq)))[0][0]
    print "TARGET: ", target_bin, freqs[target_bin]

    # print "FREQS: ", freqs

    DC_freq = 0
    DC_bin = np.where(
        freqs == min(freqs, key=lambda x: abs(float(x) - DC_freq)))[0][0]
    print "DC: ", DC_freq, freqs[DC_bin]

    #tpoints = np.linspace(0, ncycles/target_freq, N)

    D = dict()

    # FFT:
    mag_map = np.empty((stack.shape[0], stack.shape[1]))
    phase_map = np.empty((stack.shape[0], stack.shape[1]))
    sum_all_mags = np.empty((stack.shape[0], stack.shape[1]))
    mag_other_freqs = np.empty((stack.shape[0], stack.shape[1]))
    ratio_map = np.empty((stack.shape[0], stack.shape[1]))

    ft = np.empty(mag_map.shape)
    ft = ft + 0j

    DC_mag = np.empty(mag_map.shape)
    DC_phase = np.empty(mag_map.shape)

    DC = np.empty(mag_map.shape)
    DC = DC + 0j

    dynrange = np.empty(mag_map.shape)

    i = 0
    for x in range(stack.shape[0]):
        for y in range(stack.shape[1]):

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

            mag_map[x, y] = mag[target_bin]*2. #+ mag[int(N) - target_bin]
            phase_map[x, y]  = phase[target_bin]

            sum_all_mags[x, y] = sum(mag) 
            mag_other_freqs[x, y] = sum(mag) - mag[target_bin]*2.
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
    D['tpoints'] = tpoints
    D['N'] = N
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

    return D


#def get_fft_by_run(run_path):
def get_tpoints(run_path):    
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

#
#    if reduceit:
#        sample = block_reduce(sample, reduce_factor, func=np.mean)
#
#
    # INTERPOLATE FRAMES:
    ncycles = len(find_cycs) + 1
    N = int(round((ncycles / target_freq) * sampling_rate))
    print "N samples should be: ", N

    FORMAT = '%Y%m%d%H%M%S%f'
    datetimes = [f.split('_')[1] for f in files]
    tstamps = [float(datetime.datetime.strptime(t, FORMAT).strftime("%H%m%s%f")) for t in datetimes]
    actual_tpoints = [(float(i) - float(tstamps[0]))/1E6 for i in tstamps]
    tpoints = np.linspace(0, (ncycles/target_freq), N)
    
    return N, ncycles, nframes_per_cycle, strt_idxs, tpoints, actual_tpoints


def get_fft_by_run(run_path):
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
    tpoints = np.linspace(0, (ncycles/target_freq)*sampling_rate, N)
 

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
        motion_dir = os.path.join(os.path.split(run_path)[0], 'mCorrected', 'Motion', 'Registration')
        mfiles = os.listdir(motion_dir)
        mfiles = [m for m in mfiles if m.endswith('_correctedFrames.npz') and run in m]
        print mfiles
        data = np.load(os.path.join(motion_dir, mfiles[0]))
        print "Loaded motion-corrected frames for RUN: %s" % run
        tmp_stack = data['correctedFrameArray']
        #sample = tmp_stack[:,:,0]
        print "Stack is: ", tmp_stack.shape
        if reduceit:
            stack = np.empty((sample.shape[0], sample.shape[1], len(files)))
            for i in range(tmp_stack.shape[2]):
                im = tmp_stack[:,:,i]
                im_reduced = block_reduce(im, reduce_factor, func=np.mean)
                stack[:,:,i] = im_reduced
        else:
            stack = tmp_stack
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

            mag_map[x, y] = mag[target_bin]*2.  #+ mag[int(N) - target_bin]
            phase_map[x, y]  = phase[target_bin]

            sum_all_mags[x, y] = sum(mag) 
            mag_other_freqs[x, y] = sum(mag) - mag[target_bin]*2.
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
parser.add_option('--average', action='store_true', dest='average', default=False, help='Average runs of the same condition?')


(options, args) = parser.parse_args()
average = options.average
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

if average is True:
    condition_names = ['Left', 'Right', 'Top', 'Bottom']
    for condition_name in condition_names:
        Process(target=process_averaged_run, args=(session_path, condition_name, sampling_rate, target_freq,  append_to_name,)).start()
    print "Started process for cond %s runs." % condition_name
else:
    print "Found %i runs to process:" % len(run_list)
    for run in run_list:
        print run
    for run in run_list:
        Process(target=get_fft_by_run, args=(os.path.join(session_path, run), )).start() 
        print "Started process for run -- %s" % run


