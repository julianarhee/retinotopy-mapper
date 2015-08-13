#!/usr/bin/env python2
#rotate flashing wedge
from psychopy import visual, event, core, monitors, logging, tools
from pvapi import PvAPI, Camera
import time
from scipy.misc import imsave
import numpy as np
import multiprocessing as mp
import threading
from Queue import Queue
import sys
import errno
import os
import optparse

import pylab, math, serial, numpy

import random
import itertools
import cPickle as pkl

from datetime import datetime
import re
import StringIO
import scipy.misc

from libtiff import TIFF
import random

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def valid_duplicate_spacing(x, nconds):
    for i, elem in enumerate(x):
        if elem in x[i+1:i+nconds-1]:
            return False
    return True

def sample_permutations_with_duplicate_spacing(seq, nconds, nreps):
    sample_seq = []
    sample_seq = [sample_seq + seq for i in range(nreps)] 
    sample_seq = list(itertools.chain.from_iterable(sample_seq))    
    print sample_seq
    # sample_seq = seq + seq        
    random.shuffle(sample_seq)    
    while not valid_duplicate_spacing(sample_seq, nconds):
        random.shuffle(sample_seq)
    return sample_seq



monitor_list = monitors.getAllMonitors()

parser = optparse.OptionParser()
parser.add_option('--no-camera', action="store_false", dest="acquire_images", default=True, help="just run PsychoPy protocol")
parser.add_option('--save-images', action="store_true", dest="save_images", default=False, help="save camera frames to disk")
parser.add_option('--output-path', action="store", dest="output_path", default="/tmp/frames", help="out path directory [default: /tmp/frames]")
parser.add_option('--output-format', action="store", dest="output_format", type="choice", choices=['tif', 'png', 'npz', 'pkl'], default='tif', help="out file format, tif | png | npz | pkl [default: png]")
parser.add_option('--use-pvapi', action="store_true", dest="use_pvapi", default=True, help="use the pvapi")
parser.add_option('--use-opencv', action="store_false", dest="use_pvapi", help="use some other camera")
parser.add_option('--fullscreen', action="store_true", dest="fullscreen", default=True, help="display full screen [defaut: True]")
parser.add_option('--debug-window', action="store_false", dest="fullscreen", help="don't display full screen, debug mode")
parser.add_option('--write-process', action="store_true", dest="save_in_separate_process", default=True, help="spawn process for disk-writer [default: True]")
parser.add_option('--write-thread', action="store_false", dest="save_in_separate_process", help="spawn threads for disk-writer")
parser.add_option('--monitor', action="store", dest="whichMonitor", default="testMonitor", help=str(monitor_list))
parser.add_option('--type', action="store", dest="imtype", default="auto", help="auto | gcamp")

(options, args) = parser.parse_args()

acquire_images = options.acquire_images
save_images = options.save_images
output_path = options.output_path
output_format = options.output_format
save_in_separate_process = options.save_in_separate_process
fullscreen = options.fullscreen
whichMonitor = options.whichMonitor
if not fullscreen:
    winsize = [800, 600]
else:
    winsize = monitors.Monitor(whichMonitor).getSizePix()
use_pvapi = options.use_pvapi

print winsize
print output_format

imtype = options.imtype
if imtype == 'gcamp':
    print "shorter trials for GCAMP selected"

if not acquire_images:
    save_images = False

save_as_png = False
save_as_npz = False
save_as_dict = False
save_as_tif = False
if output_format == 'png':
    save_as_png = True
elif output_format == 'tif':
    save_as_tif = True
elif output_format == 'npz':
    save_as_npz = True
else:
    save_as_dict = True

print save_as_dict

# Make the output path if it doesn't already exist
try:
    os.mkdir(output_path)
except OSError, e:
    if e.errno != errno.EEXIST:
        raise e
    pass


# -------------------------------------------------------------
# Camera Setup
# -------------------------------------------------------------

camera = None

if acquire_images:

    print('Searching for camera...')


    # try PvAPI
    if use_pvapi:

        pvapi_retries = 50

        try:
            camera_driver = PvAPI(libpath='./')
            cameras = camera_driver.camera_list()
            cam_info = cameras[0]

            # Let it have a few tries in case the camera is waking up
            n = 0
            while cam_info.UniqueId == 0L and n < pvapi_retries:
                cameras = camera_driver.camera_list()
                cam_info = cameras[0]
                n += 1
                time.sleep(0.1)

            if cameras[0].UniqueId == 0L:
                raise Exception('No cameras found')
            camera = Camera(camera_driver, cameras[0])

            print("Bound to PvAPI camera (name: %s, uid: %s)" % (camera.name, camera.uid))

        except Exception as e:

            print("Unable to find PvAPI camera: %s" % e)


    if camera is None:
        try:
            import opencv_fallback

            camera = opencv_fallback.Camera(0)

            print("Bound to OpenCV fallback camera.")
        except Exception as e2:
            print("Could not load OpenCV fallback camera")
            print e2
            exit()


# -------------------------------------------------------------
# Set up a thread to write stuff to disk
# -------------------------------------------------------------

if save_in_separate_process:
    im_queue = mp.Queue()
else:
    im_queue = Queue()

disk_writer_alive = True


def save_images_to_disk():
    print('Disk-saving thread active...')
    n = 0

    currdict = im_queue.get()

    # currpath = '%s/%s' % (output_path, currdict['condName'])
    # if not os.path.exists(currpath):
    #     os.mkdir(currpath)

    while currdict is not None:
        # Make the output path if it doesn't already exist
        currpath = '%s/%s' % (output_path, currdict['condName'])
        if not os.path.exists(currpath):
            os.mkdir(currpath)

        subpath = '%s/trial%i' % (currpath, int(currdict['cyc']))
        if not os.path.exists(subpath):
            os.mkdir(subpath)

        if save_as_png:
            fname = '%s/%i_%i_%i_SZ%s_SF%s_TF%s_%s_stim%s.png' % (subpath, int(currdict['time']), int(currdict['frame']), int(n), str(currdict['size']), str(currdict['sf']), str(currdict['tf']), str(currdict['pos']), str(currdict['stim']))
            tiff = TIFF.open(fname, mode='w')
            tiff.write_image(currdict['im'])
            tiff.close()

        elif save_as_tif:
            fname = '%s/%i_%i_%i_SZ%s_SF%s_TF%s_%s_stim%s.tif' % (subpath, int(currdict['time']), int(currdict['frame']), int(n), str(currdict['size']), str(currdict['sf']), str(currdict['tf']), str(currdict['pos']), str(currdict['stim']))
            tiff = TIFF.open(fname, mode='w')
            tiff.write_image(currdict['im'])
            tiff.close()

        elif save_as_npz:
            np.savez_compressed('%s/test%d.npz' % (output_path, n), currdict['im'])
        
        else:

            fname = '%s/%s/00%i_%i_%i_%i.pkl' % (output_path, currdict['condName'], int(currdict['condNum']), int(currdict['time']), int(currdict['frame']), int(n))
            with open(fname, 'wb') as f:
                pkl.dump(currdict, f, protocol=pkl.HIGHEST_PROTOCOL) #protocol=pkl.HIGHEST_PROTOCOL)
        #if n % 100 == 0:
        #print 'DONE SAVING FRAME: ', currdict['frame'], n #fdict
        n += 1
        currdict = im_queue.get()

    disk_writer_alive = False
    print('Disk-saving thread inactive...')


if save_in_separate_process:
    disk_writer = mp.Process(target=save_images_to_disk)
else:
    disk_writer = threading.Thread(target=save_images_to_disk)

# disk_writer.daemon = True

if save_images:
    disk_writer.daemon = True
    disk_writer.start()

FORMAT = '%Y%m%d%H%M%S%f'

# -------------------------------------------------------------
# Psychopy stuff here (just lifted from a demo)
# -------------------------------------------------------------
expdict = dict()

strt_timestamp = datetime.now().strftime(FORMAT)

globalClock = core.Clock()

#make a window
win = visual.Window(fullscr=fullscreen, size=winsize, units='deg', monitor=whichMonitor)

# SET CONDITIONS:
num_cond_reps = 5 #20 # 8 how many times to run each condition
#num_seq_reps = 20 # how many times to do the cycle of 1 condition
# conditionTypes = ['1', '2', '3', '4']
conditionTypes = ['1', '2']
condLabel = ['blank', 'gab-left', 'gab-right'] #['V-Left','V-Right','H-Down','H-Up']
conditionMatrix = sample_permutations_with_duplicate_spacing(conditionTypes, len(conditionTypes), num_cond_reps) # constrain so that at least 2 diff conditions separate repeats
#conditionMatrix = ['0',' 1', '2']

#conditionMatrix = []
# for i in conditionTypes:
#     conditionMatrix.append([np.tile(i, num_cond_reps)])
# conditionMatrix = list(itertools.chain(*conditionMatrix))
# conditionMatrix = sorted(list(itertools.chain(*conditionMatrix)), key=natural_keys)
#conditionMatrix = random.shuffle(conditionMatrix)

#conditionMatrix = [int(i) for i in conditionMatrix]

#blanks = np.zeros((1,len(conditionMatrix)))[0]
#blanks = [str(int(i)) for i in blanks]
#fullmat = [iter(blanks), iter(conditionMatrix)]
#conditionMatrix = list(it.next() for it in itertools.cycle(fullmat))
#conditionMatrix.append('0')
conditionMatrix.insert(0, '0')
conditionMatrix.append('0')
print "COND:", conditionMatrix


#input parameters
screen_width_cm = monitors.Monitor(whichMonitor).getWidth()
screen_height_cm = (float(screen_width_cm)/monitors.Monitor(whichMonitor).getSizePix()[0])*monitors.Monitor(whichMonitor).getSizePix()[1]
total_length = max([screen_width_cm, screen_height_cm])
screen_width_deg = tools.monitorunittools.cm2deg(screen_width_cm, monitors.Monitor(whichMonitor))
screen_height_deg = tools.monitorunittools.cm2deg(screen_height_cm, monitors.Monitor(whichMonitor))
print "width", screen_width_cm, screen_width_deg
print "height", screen_height_cm, screen_height_deg
print total_length

#time parameters
fps = 60.
#total_time = 120.0 #total_length/(total_length*cyc_per_sec) #how long it takes for a bar to move from startPoint to endPoint
if imtype=='auto':
    dur_stimulus = 5.0
    dur_blank = 15.0
else:
    dur_stimulus = 5.0
    dur_blank = 5.0

total_time = dur_stimulus + dur_blank # length of each trial

frames_per_cycle = fps*total_time #fps/cyc_per_sec
distance = monitors.Monitor(whichMonitor).getDistance()

duration = total_time #total_time*num_seq_reps; #how long to run the same condition for (seconds)

# SPECIFY STIM PARAMETERS
patch = visual.GratingStim(win=win, tex='sin', mask='gauss', units='deg') #gives a 'Gabor'
patch.sf = 0.08
patch.ori = 90 # horizontal is 90, vertical is 0
patch.size = (30, 30)
patch.setAutoDraw(False)
driftFrequency = 4.0 # drifting frequency in Hz

t=0
nframes = 0.
frame_accumulator = 0
flash_count = 0
last_t = None

report_period = 60 # frames

if acquire_images:
    # Start acquiring
    win.flip()
    time.sleep(0.002)
    camera.capture_start()
    camera.queue_frame()
        
# RUN:
getout = 0
cyc = 1
for condType in conditionMatrix:
    print cyc
    print condLabel[int(condType)]

    # SPECIFICY CONDITION TYPES:
    if condLabel[int(condType)] == 'gab-left':
        patch.pos = (0 - screen_width_deg*0.5, 0 + screen_height_deg*0.15)
        stim = 1
    elif condLabel[int(condType)] == 'gab-right':
        patch.pos = (0 + screen_width_deg*0, 0 + screen_height_deg*0.15)
        stim = 1
    elif condType == 'blank': # BLANK
        #patch.pos = (0, 0)
        patch.tex = np.zeros(patch.size)
        patch.mask = None
        stim = -1

    # DISPLAY LOOP:
    win.flip() # first clear everything
    #time.sleep(duration) # wait a sec

    # clock = core.Clock()
    frame_counter = 0

    print "DUR:", duration
    clock = core.Clock()
    while clock.getTime()<=duration: #frame_counter < frames_per_cycle*num_seq_reps: #endPoint - posLinear <= dist: #frame_counter <= frames_per_cycle*num_seq_reps: 
        t = globalClock.getTime()

        patch.phase = 1 - clock.getTime() * driftFrequency
        # if int(condType) > 0:

        if clock.getTime() > dur_stimulus:
            patch.tex = np.zeros((256, 256))
            #patch.mask = None
            stim = 0
        else:
            patch.tex = 'sin'
            #patch.mask = 'gauss'
            stim = 1
            #t = globalClock.getTime()
            #patch.phase = 1 - clock.getTime() * driftFrequency

        patch.draw()
        win.flip()

        if acquire_images:
            im_array = camera.capture_wait()
            camera.queue_frame()

            if save_images:
                fdict = dict()
                fdict['im'] = im_array
                fdict['size'] = patch.size[0]
                fdict['tf'] = driftFrequency
                fdict['sf'] = patch.sf[0]
                fdict['ori'] = patch.ori
                fdict['condName'] = condLabel[int(condType)]#condLabel[int(condType)-1]
                fdict['frame'] = frame_counter
                fdict['time'] = datetime.now().strftime(FORMAT)
                fdict['pos'] = patch.pos
                fdict['stim'] = stim
                fdict['cyc'] = cyc
                im_queue.put(fdict)

        if nframes % report_period == 0:
            if last_t is not None:
                print('avg frame rate: %f' % (report_period / (t - last_t)))
            last_t = t

        nframes += 1
        frame_counter += 1
        flash_count += 1

        # Break out of the while loop if these keys are registered
        if event.getKeys(keyList=['escape', 'q']):
            getout = 1
            break  

    cyc += 1

    # Break out of the FOR loop if these keys are registered        
    if getout==1:
        break
    else:
        continue

win.close() 

if acquire_images:
    camera.capture_end()
    camera.close()

# fdict['im'] = None
# fdict = None
print "GOT HERE"
im_queue.put(None)


if save_images:
    hang_time = time.time()
    # nag_time = 0.05
    nag_time = 2.0

    sys.stdout.write('Waiting for disk writer to catch up (this may take a while)...')
    sys.stdout.flush()

    while disk_writer.is_alive():
        sys.stdout.write('.')
        sys.stdout.flush()
        # disk_writer.pid(), disk_writer.exitcode()
        time.sleep(nag_time)

    print("\n")

    print 'disk_writer.isAlive()', disk_writer.is_alive()
    if not im_queue.empty():
        print "NOT EMPTY"
        print im_queue.get()
        print "disk_writer_alive", disk_writer_alive
        print("WARNING: not all images have been saved to disk!")
    else:
        print "EMPTY QUEUE"

    # disk_writer_alive = False

    # if save_in_separate_process and disk_writer is not None:
    #     print("Terminating disk writer...")
    #     disk_writer.join()
    #     disk_writer.terminate()

    # disk_writer.terminate()
    disk_writer.join()
    print('Disk writer terminated')

expdict['condMat'] = conditionMatrix
expdict['start_time'] = strt_timestamp
expdict['end_time'] = datetime.now().strftime(FORMAT)

metaname = os.path.join(os.path.split(output_path)[0], 'session_'+os.path.split(output_path)[1]+'.pkl')
with open(metaname, 'wb') as f:
    pkl.dump(expdict, f, protocol=pkl.HIGHEST_PROTOCOL)

    

