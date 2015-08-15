#!/usr/bin/env python2
#rotate flashing wedge
from psychopy import visual, event, core, monitors, logging, tools, filters
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
import string

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


def flatten(l, limit=1000, counter=0):
  for i in xrange(len(l)):
    if (isinstance(l[i], (list, tuple)) and
        counter < limit):
      for a in l.pop(i):
        l.insert(i, a)
        i += 1
      counter += 1
      return flatten(l, limit, counter)
  return l


from psychopy.tools.unittools import radians


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


monitor_list = monitors.getAllMonitors()

parser = optparse.OptionParser()
parser.add_option('--no-camera', action="store_false", dest="acquire_images", default=True, help="just run PsychoPy protocol")
parser.add_option('--save-images', action="store_true", dest="save_images", default=False, help="save camera frames to disk")
parser.add_option('--output-path', action="store", dest="output_path", default="/tmp/frames", help="out path directory [default: /tmp/frames]")
parser.add_option('--output-format', action="store", dest="output_format", type="choice", choices=['tif', 'png', 'npz', 'pkl'], default='tif', help="out file format, tif | png | npz | pkl [default: tif]")
parser.add_option('--use-pvapi', action="store_true", dest="use_pvapi", default=True, help="use the pvapi")
parser.add_option('--use-opencv', action="store_false", dest="use_pvapi", help="use some other camera")
parser.add_option('--fullscreen', action="store_true", dest="fullscreen", default=True, help="display full screen [defaut: True]")
parser.add_option('--debug-window', action="store_false", dest="fullscreen", help="don't display full screen, debug mode")
parser.add_option('--write-process', action="store_true", dest="save_in_separate_process", default=True, help="spawn process for disk-writer [default: True]")
parser.add_option('--write-thread', action="store_false", dest="save_in_separate_process", help="spawn threads for disk-writer")
parser.add_option('--monitor', action="store", dest="whichMonitor", default="testMonitor", help=str(monitor_list))
parser.add_option('--use-images', action="store_true", dest="use_images", default=False, help="show images, otherwise Gabors, thru aperture")
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

use_images = options.use_images
print use_images
print winsize
print output_format

if not acquire_images:
    save_images = False

save_as_tif = False
save_as_png = False
save_as_npz = False
save_as_dict = False
if output_format == 'png':
    save_as_png = True
elif output_format == 'tif':
    save_as_tif = True
elif output_format == 'npz':
    save_as_npz = True
else:
    save_as_dict = True


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

    while currdict is not None:
        # Make the output path if it doesn't already exist
        currpath = '%s/%s/' % (output_path, currdict['condName'])
        if not os.path.exists(currpath):
            os.mkdir(currpath)

        if save_as_png:
            fname = '%s/%s/%i_%i_%i_SZ%s_SF%s_TF%s_pos%s_cyc%s_stim%s.png' % (output_path, currdict['condName'], int(currdict['time']), int(currdict['frame']), int(n), str(currdict['size']), str(currdict['sf']), str(currdict['tf']), str(currdict['pos']), str(currdict['cycleidx']), str(currdict['stim']))
            tiff = TIFF.open(fname, mode='w')
            tiff.write_image(currdict['im'])
            tiff.close()

        elif save_as_tif:
            fname = '%s/%s/%i_%i_%i_SZ%s_SF%s_TF%s_pos%s_cyc%s_stim%s.tif' % (output_path, currdict['condName'], int(currdict['time']), int(currdict['frame']), int(n), str(currdict['size']), str(currdict['sf']), str(currdict['tf']), str(currdict['pos']), str(currdict['cycleidx']), str(currdict['stim']))
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

# Formatting stuff for saving:
FORMAT = '%Y%m%d%H%M%S%f'
allow = string.letters + string.digits + '-'

# -------------------------------------------------------------
# Psychopy stuff here (just lifted from a demo)
# -------------------------------------------------------------

globalClock = core.Clock()

# make a window
win = visual.Window(fullscr=fullscreen, size=winsize, units='deg', monitor=whichMonitor)

# SET CONDITIONS:
num_cond_reps = 30 #20 # 8 how many times to run each condition
condTypes = flatten(['0', list(np.tile('1', num_cond_reps))])
condMatrix = ['0', '1'] #flatten(condTypes)
print condMatrix
labels = ['blank', 'stimulus']
condLabels = [labels[int(s)] for s in condTypes]
num_cycles = {'0': 1, '1': num_cond_reps}

# SCREEN PARAMETERS:
screen_width_cm = monitors.Monitor(whichMonitor).getWidth()
screen_height_cm = (float(screen_width_cm)/monitors.Monitor(whichMonitor).getSizePix()[0])*monitors.Monitor(whichMonitor).getSizePix()[1]
total_length = max([screen_width_cm, screen_height_cm])
screen_width_deg = tools.monitorunittools.cm2deg(screen_width_cm, monitors.Monitor(whichMonitor))
screen_height_deg = tools.monitorunittools.cm2deg(screen_height_cm, monitors.Monitor(whichMonitor))
screen_size = (screen_width_deg,screen_height_deg)
print "width", screen_width_cm, screen_width_deg
print "height", screen_height_cm, screen_height_deg

# TIMING PARAMETERS:
fps = 60.
cyc_per_sec = 0.05 # cycle freq in Hz
total_time = 1./cyc_per_sec #total_length/(total_length*cyc_per_sec) #how long it takes for a bar to move from startPoint to endPoint
frames_per_cycle = fps*total_time #fps/cyc_per_sec
distance = monitors.Monitor(whichMonitor).getDistance()
duration = total_time #total_time*num_seq_reps; #how long to run the same condition for (seconds)

# SET UP ALL THE STIMULI:
if use_images:
    stimdir = '../stimuli'
    stimset = os.listdir(stimdir)[0]
    print stimset
    stims = os.listdir(os.path.join(stimdir, stimset))
    stims = [s for s in stims if os.path.splitext(s)[1] == '.tif']
    stims = [os.path.join(stimdir, stimset, s) for s in stims]
else:
    oris = [s for s in range(0, 360, 30)]
    stims = oris

tmp_stimIdxs = [i for i in range(len(stims))]
random.shuffle(tmp_stimIdxs)
stimIdxs = flatten(np.tile(tmp_stimIdxs, num_cond_reps))

# SPECIFY TRAVEL PARAMETERS:
path_diam = 0.3*min([screen_width_deg, screen_height_deg]) # limiting dimension of screen for circle
deg_per_frame = 360 * cyc_per_sec / fps # number of degrees to move per frame
path_pos = np.arange(0, 360, deg_per_frame)
driftFrequency = 4.0 # drifting frequency in Hz
patch_size = (45, 45)
dwell_time = duration * cyc_per_sec

if use_images:
    print "Creating textures..."
    textures = []
    for p in stims:
        texture = visual.PatchStim(win=win, tex=p, mask='raisedCos', size=patch_size, units='deg')
        texture.sf = None
        texture.setAutoDraw(False)
        textures.append(texture)
    driftFrequency = 0.0 

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
#clock = core.Clock()
for curr_cond in condMatrix:
    print condLabels[int(curr_cond)]

    # SPECIFICY CONDITION TYPES:
    if curr_cond == '0': # BLANK
        blankscreen = numpy.zeros([256,256,3]);
        #blankscreen[:,:,0] = 0.
        patch = visual.PatchStim(win=win,tex=blankscreen,mask='none',units='deg',size=screen_size, ori=0.)
        patch.sf = None
        patch.pos = (0, 0)
        
    elif curr_cond == '1': # STIMULUS
        if use_images: # USE SCENE STIMULI
            patch = textures[0]
            patch.sf = None
            patch.ori = 0.00
        else: # USE GABORS
            patch = visual.GratingStim(win=win, tex='sin', mask='raisedCos', size=patch_size, units='deg') #gives a 'Gabor'
            patch.sf = 0.08
            patch.ori = stims[0] # horizontal is 90, vertical is 0
            patch.setAutoDraw(False)

    # DISPLAY LOOP:
    win.flip() # first clear everything
    #time.sleep(duration) # wait a sec

    frame_counter = 0
    sidx = 0
    curr_frame = 0

    print "DUR:", duration*num_cycles[curr_cond]
    clock = core.Clock()
    while clock.getTime()<=duration*num_cycles[curr_cond]: #frame_counter < frames_per_cycle*num_seq_reps: #endPoint - posLinear <= dist: #frame_counter <= frames_per_cycle*num_seq_reps: 
        t = globalClock.getTime()
        
        if int(curr_cond) > 0:
            if not use_images:
                patch.phase = 1 - clock.getTime() * driftFrequency

            if nframes % (dwell_time*fps) == 0:
                print "STIM idx:", sidx
                sidx += 1
                if use_images:
                    patch = textures[stimIdxs[sidx]]
                    patch.sf = None
                else:
                    patch.ori = stims[stimIdxs[sidx]]

            path_pos = ( ( clock.getTime() % duration ) / duration) * 360
            patch.pos = pol2cart(path_pos, path_diam, units='deg') #pol2cart(path_pos[curr_frame], path_diam, units='deg')

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
                fdict['condName'] = condLabels[int(curr_cond)] #condLabel[int(condType)-1]
                fdict['frame'] = frame_counter
                fdict['cycleidx'] = sidx
                fdict['stim'] = re.sub('[^%s]' % allow, '', str(stims[stimIdxs[sidx]]))
                # print 'frame #....', frame_counter
                fdict['time'] = datetime.now().strftime(FORMAT)
                fdict['pos'] = patch.pos

                im_queue.put(fdict)


        if nframes % report_period == 0:
            if last_t is not None:
                print('avg frame rate: %f' % (report_period / (t - last_t)))
            last_t = t

        nframes += 1
        frame_counter += 1
        curr_frame += 1

        # Break out of the while loop if these keys are registered
        if event.getKeys(keyList=['escape', 'q']):
            getout = 1
            break  

    #print "TOTAL COND TIME: " + str(clock.getTime())

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
    

