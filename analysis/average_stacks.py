#!/user/bin/env python2

import optparse
import os

from process_fft_runs import load_movie_frames, get_trial_list


def get_trials(acquisition_dir, runs, excluded=None):

    if excluded is None:
        excluded = dict((run, []) for run in runs)

    trials = dict()
    for run in runs:
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
	if len(excluded) > 0:
	    curr_trial_list = [int(t) for t in trial_str.split(',')]
            print "Selected %i trials to average:" % len(curr_trial_list)
        else:
	    curr_trial_list = [t for t in trialdict[run]]
            print "Averaging ALL trials in run:"
        pp.pprint(curr_trial_list)
	trials_to_avg[run] = curr_trial_list 

    return trials_to_avg

def get_averaged_stack(raw_dir, trial_list):
    ntrials = len(trial_list)
    for tidx,curr_trial in enumerate(sorted(trial_list, key=natural_keys)):
        img_path = os.path.join(raw_dir, curr_trial, 'frames')
        print "Loading trial %i of %." % (tidx, ntrials)
        stack = load_movie_frames(img_path)
        if tidx == 0:
            avg = stack.copy()
        else:
            avg += stack
    
    avg /= ntrials
    return avg
 
if __name__ == '__main__':

    parser = optparse.OptionParser()

    parser.add_option('-R', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")

    (options, args) = parser.parse_args()

    rootdir = options.rootdir
    animalid = options.animalid
    session = options.session
    acquisition = options.acquisition

    acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)
    runs = sorted([run for run in os.listdir(acquisition_dir) if os.path.isdir(os.path.join(acquisition_dir, run)) and 'surface' not in run and 'Surface' not in run], key=natural_keys)

    # Get dict of trial-condition for each run:
    trials = get_trials(acquisition_dir, runs, excluded=excluded) 
    print "Trials by run:"
    pp.pprint(trials)
 
    # For each run, get user-input on which trials to average:
    trialdict = get_trials_to_average(trials)
     
    # Write each frame to file:
    for run in sorted(trialdict.keys(), key=natural_keys):
        curr_raw_dir = os.path.join(acquisition_dir, run, 'raw') 
        trial_list = trialdict[run]
        which_trials = trial_list.join('_')
        write_dir = os.path.join(acquisition_dir, 'averaged_trials', which_trials)
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)
        print "Writing averaged frames to: %s" % write_dir
 
        avg = get_averaged_stack(curr_raw_dir, trial_list)
        for f in avg.shape[-1]:
            if f % 100 == 0:
                print "Writing frames... %i of %i" % (f, avg.shape[-1])
            tf = TIFF.open(os.path.join(write_dir, "%i.%s" % (f, img_ext)), 'w')
            tf.write_image(avg[:,:,f])
        print "Finished averaging RUN: %s" % run

    print "DONE!"

 
