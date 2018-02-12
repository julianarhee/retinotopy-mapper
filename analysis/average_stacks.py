#!/user/bin/env python2

import optparse
import os
import re
import pprint
pp = pprint.PrettyPrinter(indent=4)
import pandas as pd
from libtiff import TIFF
import numpy as np

from process_fft_runs import load_movie_frames, get_trial_list, extract_frame_info_for_trial

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def get_trials(acquisition_dir, runs, excluded=None):

    if excluded is None:
        excluded = dict((run, []) for run in runs)

    trials = dict()
    for run in runs:
        if 'average' in run:
            continue
	trial_list = [t for t in os.listdir(os.path.join(acquisition_dir, run, 'raw')) if 'trial' in t]
	tinfo = []
	for trial in trial_list:
	    if trial in excluded[run]:
		continue
	    # laod frame info:
	    finfo_path = os.path.join(acquisition_dir, run, 'raw', trial, 'frame_info.txt')
	    fdata = pd.read_csv(finfo_path, sep='\t')
            if 'cond_idx' in fdata.keys():
	        cond = fdata['cond_idx'][0] #fdata['currcond'][0]
            else:
                cond = fdata['currcond'][0]
	    if cond == 2:
		condname = 'right'
	    elif cond == 4:
		condname = 'bottom'
	    tinfo.append((trial, condname))
	trials[run] = tinfo

    return trials

def check_excluded_trials(runs):

    excluded = dict((run, []) for run in runs)
    for run in excluded.keys():
	excluded = raw_input('Enter comma-sep list of trials to exclude: ')
	excluded_trials = []
	if len(excluded) > 0:
	    excluded_trials = [int(t) for t in excluded.split(',')]
	excluded[run] = excluded_trials

    return excluded

def get_trials_to_average(trialdict):
    runs = sorted(trialdict.keys(), key=natural_keys)

    trials_to_avg = dict((run, []) for run in runs)
    for run in trials_to_avg.keys():
	trial_str = raw_input('Enter comma-sep list of trials to average. Press <ENTER> for all trials in run: ')
	if len(trial_str) > 0:
	    curr_trial_list = ['trial%03d' % int(t) for t in trial_str.split(',')]
            print "Selected %i trials to average:" % len(curr_trial_list)
        else:
	    curr_trial_list = [t[0] for t in trialdict[run]]
            print "Averaging ALL trials in run:"
        pp.pprint(curr_trial_list)
	trials_to_avg[run] = curr_trial_list 

    return trials_to_avg

def get_averaged_stack(raw_dir, trial_list):
    ntrials = len(trial_list)
    for tidx,curr_trial in enumerate(sorted(trial_list, key=natural_keys)):
        # Load raw frames:
        trial_dir = os.path.join(raw_dir, curr_trial)
        img_path = os.path.join(trial_dir, 'frames')
        print "Loading trial %i of %i." % (tidx, ntrials)
        stack = load_movie_frames(img_path)
        curr_tstamps, curr_pos = extract_frame_info_for_trial(trial_dir, averaged=False)
        if tidx == 0:
            avg = stack.copy()
            tstamps = np.array(curr_tstamps)
            xpos = np.array([c[0] for c in curr_pos])
            ypos = np.array([c[1] for c in curr_pos])
        else:
            # Adjust n elements, if needed:
            if stack.shape[-1] > avg.shape[-1]:
                stack = stack[:,:,0:avg.shape[-1]]
                curr_tstamps = curr_stamps[0:avg.shape[-1]]
                curr_pos = curr_pos[0:avg.shape[-1]]
            elif stack.shape[-1] < avg.shape[-1]:
                avg = avg[:,:,0:stack.shape[-1]]
                curr_tstamps = curr_tstamps[0:stack.shape[-1]]
                curr_pos = curr_pos[0:stack.shape[-1]]
    
            avg += stack
            tstamps += np.array(curr_tstamps)
            xpos += np.array([c[0] for c in curr_pos]) 
            ypos += np.array([c[1] for c in curr_pos])

    # Get avearge:
    avg /= ntrials
    tstamps /= ntrials
    xpos /= ntrials
    ypos /= ntrials

    # Adjust "frame info": 
    nframes = avg.shape[-1]           
    sample_frame_log_path = os.path.join(trial_dir, 'frame_info.txt')
    fdata = pd.read_csv(sample_frame_log_path, sep='\t')
    fdata = fdata.iloc[0:nframes]
    fdata['tstamp'] = tstamps
    fdata['xpos'] = xpos
    fdata['ypos'] = ypos

    return avg, fdata

if __name__ == '__main__':

    parser = optparse.OptionParser()

    parser.add_option('-R', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
    parser.add_option('-x', '--ext', action='store', dest='ext', default='tif', help="image extension [default: tif]")

    (options, args) = parser.parse_args()

    rootdir = options.rootdir
    animalid = options.animalid
    session = options.session
    acquisition = options.acquisition
    img_ext = options.ext
    uint16 = True

    acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)
    runs = sorted([run for run in os.listdir(acquisition_dir) if os.path.isdir(os.path.join(acquisition_dir, run)) and not re.search('surface', run, re.IGNORECASE)], key=natural_keys)

    # Get dict of trial-condition for each run:
    trials = get_trials(acquisition_dir, runs) 
    print "Trials by run:"
    pp.pprint(trials)
 
    # For each run, get user-input on which trials to average:
    trialdict = get_trials_to_average(trials)
     
    # Write each frame to file:
    for run in sorted(trialdict.keys(), key=natural_keys):
        curr_raw_dir = os.path.join(acquisition_dir, run, 'raw') 
        trial_list = trialdict[run]
        which_trials = '_'.join(trial_list)
        avg_trial_dir = os.path.join(acquisition_dir, run, 'averaged_trials', which_trials)
        out_frame_dir = os.path.join(acquisition_dir, run, 'averaged_trials', which_trials, 'frames')
        if not os.path.exists(out_frame_dir):
            os.makedirs(out_frame_dir)
        print "Writing averaged frames to: %s" % out_frame_dir
 
        avg, fdata = get_averaged_stack(curr_raw_dir, trial_list)
        fdata.to_csv(os.path.join(avg_trial_dir, 'frame_info.txt'), sep='\t')
 
        if uint16 is True:
            avg = avg.astype('uint16')
        for f in range(avg.shape[-1]):
            if f % 100 == 0:
                print "Writing frames... %i of %i" % (f, avg.shape[-1])
            tf = TIFF.open(os.path.join(out_frame_dir, "%i.%s" % (f, img_ext)), 'w')
            tf.write_image(avg[:,:,f])
        print "Finished averaging RUN: %s" % run

    print "DONE!"

 
