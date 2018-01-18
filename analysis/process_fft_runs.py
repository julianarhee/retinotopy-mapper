import os
import matplotlib as mpl
mpl.use('Agg')
import optparse
import re
import itertools
import datetime
import time
import traceback
import h5py

import pylab as pl
import matplotlib.cm as cm
import numpy as np
import cPickle as pkl
import scipy.signal
import numpy.fft as fft
import pandas as pd
import multiprocessing as mp

from skimage.measure import block_reduce
from scipy.misc import imread
from libtiff import TIFF
from PIL import Image
from scipy import ndimage



#import hickle as hkl
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def movingaverage(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'valid')



def extract_frame_info_for_trial(trial_dir):
    
    frame_log_fpath = os.path.join(trial_dir, 'frame_info.txt')
    
    framedata = pd.read_csv(frame_log_fpath, sep='\t')
    print framedata.columns

    ### Extract events from serialdata:
    frame_tstamps = framedata[' experimentTime']
    stim_positions = framedata[' stimPosition']
    
    return frame_tstamps, stim_positions


def load_movie_frames(source_path, img_fmt='tif', reduce_factor=(1, 1)):
    
    tiffs = sorted([f for f in os.listdir(source_path) if f.endswith(img_fmt)], key=natural_keys)

    tf = TIFF.open(os.path.join(source_path, tiffs[0]), mode='r')
    sample = tf.read_image().astype('float')
    tf.close()
    if reduce_factor[0] > 1:
        sample = block_reduce(sample, reduce_factor, func=np.mean)
    print "sample type: %s, range: %s" % (sample.dtype, str([sample.max(), sample.min()]))
    print "sample shape: %s" % str(sample.shape)        

    stack = np.empty((sample.shape[0], sample.shape[1], len(tiffs)))
    for i, f in enumerate(tiffs):

        if i % 1000 == 0:
            print('%d images processed...' % i)
        tf = TIFF.open(os.path.join(source_path, f), mode='r')
        im = tf.read_image().astype('float')
        tf.close()

        if reduce_factor[0] > 1:
            im_reduced = block_reduce(im, reduce_factor, func=np.mean) # ndimage.gaussian_filter(im_reduced, sigma=gsigma)
            stack[:, :, i] = im_reduced
        else:
            stack[:, :, i] = im
    
    return stack

            
def get_fft(source_path,
            img_fmt='tif',
            target_freq=0.13,
            ncycles=20,
            sampling_rate=30.,
            reduce_factor=(1,1),
            interpolate=True,
            high_pass=True):
    
    proc_id = os.getpid()
    trialname = os.path.split(source_path)[1]
    runname = os.path.split(os.path.split(source_path)[0])[1]
    print "Starting fft for PID: {0} (run: {1} --- trial: {2})...".format(
        proc_id, runname, trialname)
    
    trial_dir = os.path.split(source_path)[0]
    
    tiffs = sorted([f for f in os.listdir(source_path) if f.endswith(img_fmt)], key=natural_keys)
    print "Found %i tiffs to process (src: %s)" % (len(tiffs), source_path)

    # Load data:
    # -------------------------------------------------------------------------
    frame_tstamps, stim_positions = extract_frame_info_for_trial(trial_dir)
    stack = load_movie_frames(source_path, img_fmt=img_fmt, reduce_factor=reduce_factor)
    sample = stack[:,:,0]
    print sample.shape
    
    # Set FFT params:
    # -------------------------------------------------------------------------
    N = int(round((ncycles / target_freq) * sampling_rate))
    N_samples = len(frame_tstamps)
    nframes_per_cycle = N_samples/ncycles
    tpoints = np.linspace(0, (ncycles/target_freq), N)
    if interpolate is True:
        moving_win_sz = len(tpoints)/ncycles * 2
        freqs = fft.fftfreq(N, 1 / sampling_rate)
    else:
        moving_win_sz = min(nframes_per_cycle) * 2
        freqs = fft.fftfreq(N_samples, 1 / sampling_rate) 
        
    # Get frequency bins for frequencies of interest:
    # -------------------------------------------------------------------------
    # When set fps to 60 vs 120 -- target_bin should be 2x higher for 120 (looks for closest matching target_bin )
    #binwidth = freqs[1] - freqs[0]
    target_bin = np.where(
        freqs == min(freqs, key=lambda x: abs(float(x) - target_freq)))[0][0]
    print "TARGET: ", target_bin, freqs[target_bin]
    DC_freq = 0
    DC_bin = np.where(
        freqs == min(freqs, key=lambda x: abs(float(x) - DC_freq)))[0][0]
    print "DC: ", DC_freq, freqs[DC_bin]    
    
    # Run FFT:
    # -------------------------------------------------------------------------
    fft_filepath = os.path.join(trial_dir, 'fft_%s.hdf5' % datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S"))
    fftfile = h5py.File(fft_filepath, 'w')
    maps = dict()
    maps['dynamic_range'] = np.empty(sample.shape)
    maps['magnitude_target'] = np.empty(sample.shape)
    maps['phase_target'] = np.empty(sample.shape)
    maps['magnitude_sum_all'] = np.empty(sample.shape)
    maps['magnitude_nontarget'] = np.empty(sample.shape)
    maps['magnitude_ratios'] = np.empty(sample.shape)
    maps['magnitude_DC'] = np.empty(sample.shape)
    maps['phase_DC'] = np.empty(sample.shape)

    try:
        for x in range(sample.shape[0]):
            for y in range(sample.shape[1]):
    
                pix = process_timecourse(stack[x, y, :], tpoints, frame_tstamps, moving_win_sz=moving_win_sz, interpolate=interpolate, high_pass=high_pass)
    
                curr_fft = fft.fft(pix)
                mag = np.abs(curr_fft)
                phase = np.angle(curr_fft)
                
                maps['dynamic_range'][x, y] = np.log2(pix.max() - pix.min())
                maps['magnitude_target'][x, y] = mag[target_bin]*2.  #+ mag[int(N) - target_bin]
                maps['phase_target'][x, y] = phase[target_bin]
                maps['magnitude_sum_all'][x, y] = sum(mag)
                maps['magnitude_nontarget'][x, y] = sum(mag) - mag[target_bin]*2.
                maps['magnitude_ratios'][x, y]  = (mag[target_bin]*2.) / maps['magnitude_nontarget'][x, y] 
                maps['magnitude_DC'][x, y]  = mag[DC_bin]
                maps['phase_DC'][x, y] = phase[DC_bin]    
                
                ft = fftfile.create_dataset('/'.join(['fft', str(x), str(y)]), curr_fft.shape, curr_fft.dtype)
                ft[...] = curr_fft
                ft.attrs['target_freq'] = target_freq
                ft.attrs['target_freq_bin'] = target_bin
                ft.attrs['DC_freq_bin'] = DC_bin
                ft.attrs['interpolate'] = interpolate
                ft.attrs['high_pass'] = high_pass
                ft.attrs['dim_x'] = sample.shape[0]
                ft.attrs['dim_y'] = sample.shape[1]
                ft.attrs['N_frames'] = N_samples
                ft.attrs['ncycles'] = ncycles
                ft.attrs['sampling_rate'] = sampling_rate
        
        if 'maps' not in fftfile.keys():
            mapgrp = fftfile.create_group('maps')
        else:
            mapgrp = fftfile['maps']
            
        for maptype in maps.keys():
            mp = mapgrp.create_dataset(maptype, maps[maptype].shape, maps[maptype].dtype)
            mp[...] = maps[maptype]    
    except Exception as e:
        print "--- ERROR processing FFT from source: -------------------------"
        print source_path
        traceback.print_exc()
        print "---------------------------------------------------------------"
    finally:
        fftfile.close
        
    return fft_filepath


def process_timecourse(tseries, desired_tpoints, actual_tpoints, moving_win_sz=0, interpolate=True, high_pass=True):
    
    if interpolate is True:
        pix = np.interp(desired_tpoints, actual_tpoints, tseries)
    else:
        pix = tseries.copy()
        
    # curr_pix = scipy.signal.detrend(stack[x, y, :], type='constant') # HP filter - over time...
    if high_pass is True:
        pix_padded = [np.ones(moving_win_sz)*pix[0], pix, np.ones(moving_win_sz)*pix[-1]]
        tmp_pix = list(itertools.chain(*pix_padded))
        tmp_pix_rolling = np.convolve(tmp_pix, np.ones(moving_win_sz)/moving_win_sz, 'same')
        remove_pad = (len(tmp_pix_rolling) - len(pix) ) / 2
        rpix = np.array(tmp_pix_rolling[remove_pad:-1*remove_pad])
        pix -= rpix
    
    return pix


def get_fft_by_run(acquisition_dir, run, img_fmt='tif', ncycles=20, target_freq=0.13,
                   sampling_rate=60., reduce_factor=(1, 1), interpolate=True, high_pass=True):
    
    
    run_dir = os.path.join(acquisition_dir, run)
    trial_source_paths = [(t, os.path.join(run_dir, t, 'frames')) for t in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, t))]
    
    print "RUN %s: Found %i trials to process." % (run, len(trial_source_paths))
    
#    for trial in trials:
#        source_path = os.path.join(run_dir, trial, 'frames')

    fft_kwargs = {'img_fmt': img_fmt, 
                  'target_freq': target_freq,
                  'ncycles': ncycles,
                  'sampling_rate': sampling_rate,
                  'reduce_factor': reduce_factor,
                  'interpolate': interpolate,
                  'high_pass': high_pass}

    proc_name = mp.current_process().name
    proc_id = os.getpid()
    print "Starting pool for PID: {0} (run: {1})...".format(
        proc_id, proc_name)
    
    ntrials = len(trial_source_paths)
    pool = mp.Pool(processes=ntrials)
    results = [(t, pool.apply_async(get_fft, (tpath,), fft_kwargs)) for t, tpath in trial_source_paths]
    t_start = time.time()
    for trial, result in results:
        print "Trial %s -- result: %s (%.2f secs)" % (trial, result.get(), time.time() - t_start)
        
    
    return results


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-R', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
    
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
    #parser.add_option('--append', action="store",
    #                  dest="append_name", default="", help="append string to saved file name")
    
    parser.add_option('--rolling', action='store_true', default=False, help="Rolling average [window size is 2 cycles] or detrend.")
    parser.add_option('--meansub', action='store_true', default=False, help="Remove mean of each frame.")
    parser.add_option('--interpolate', action='store_true', default=False, help='Interpolate frames or no.')
    
    #parser.add_option('--path', action="store",
    #                  dest="session_path", default="", help="input dir")
    
    parser.add_option('--ncycles', action="store",
                      dest="ncycles", default=20, help="ncycles (default 20)")
    parser.add_option('--motion', action='store_true', dest="motion_corrected", default=False, help="Motion corrected with WiPy or no?")
    parser.add_option('--average', action='store_true', dest='average', default=False, help='Average runs of the same condition?')
    
    
    (options, args) = parser.parse_args()
    average = options.average
    motion_corrected = options.motion_corrected
    
    #imdir = sys.argv[1]
    rootdir = options.rootdir
    animalid = options.animalid
    session = options.session
    acquisition = options.acquisition

    interpolate = options.interpolate
    high_pass = options.rolling
    meansub = options.meansub
    img_format = options.im_format
    reduce_factor = (int(options.reduce_val), int(options.reduce_val))
    gsigma = int(options.gauss_kernel)
    
    target_freq = float(options.target_freq)
    sampling_rate = float(options.sampling_rate) # 60.  # np.mean(np.diff(sorted(strt_idxs)))/cycle_dur #60.0
    ncycles = int(options.ncycles)
    cache_file = True
    cycle_dur = 1. / target_freq  # 10.
    binspread = 0
    
    acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)
    runs = [r for r in os.listdir(acquisition_dir) if os.path.isdir(os.path.join(acquisition_dir, r)) and 'surface' not in r]
    trial_list = []
    trials = dict()
    for r in runs:
        curr_trials = [t for t in os.listdir(os.path.join(acquisition_dir, r)) if os.path.isdir(os.path.join(acquisition_dir, r, t))]
        for t in curr_trials:
            trial_list.append((r, t))
            #trial_list.append((r, os.path.join(acquisition_dir, r, t, 'frames')))
            
        trials[r] = [t for t in os.listdir(os.path.join(acquisition_dir, r)) if os.path.isdir(os.path.join(acquisition_dir, r, t))]
        print "RUN %s: Found %i trials to process." % (r, len(trials[r]))
        #print "RUN %s: Found %i trials to process." % (r, len(curr_trials))
    
    t_start = time.time()
    processes = []
    for run, trial in trial_list:
        keyargs = {'img_fmt': img_format,
                   'target_freq': target_freq, 
                   'sampling_rate': sampling_rate,
                   'reduce_factor': reduce_factor,
                   'ncycles': ncycles,
                   'interpolate': interpolate,
                   'high_pass': high_pass
                   }
        
        trial_source_path = os.path.join(acquisition_dir, run, trial, 'frames')
        trial_name = '%s_%s' % (run, trial)
        proc = mp.Process(name=trial_name, target=get_fft, args=(trial_source_path,), kwargs=keyargs)
        
        #proc = mp.Process(name=run, target=get_fft_by_run, args=(acquisition_dir, run,), kwargs=keyargs)
        proc.start()
        processes.append(proc) #start()
        print "Started process for run -- %s" % proc.name

    for proc in processes:
        proc.join()
        
#    keyargs = {'img_fmt': img_format,
#               'target_freq': target_freq, 
#               'sampling_rate': sampling_rate,
#               'reduce_factor': reduce_factor,
#               'ncycles': ncycles,
#               'interpolate': interpolate,
#               'high_pass': high_pass
#               }
#    ntotal_trials = len(trial_list)
#    pool = mp.Pool(processes=ntotal_trials)
#    results = [(t, pool.apply_async(get_fft, (tpath,), keyargs)) for t, tpath in trial_list]
#    t_start = time.time()
#    for trial, result in results:
#        print "Trial %s -- result: %s (%.2f secs)" % (trial, result.get(), time.time() - t_start)
#        
#    
    print "DONE! TOTAL TIME:", time.time() - t_start
        