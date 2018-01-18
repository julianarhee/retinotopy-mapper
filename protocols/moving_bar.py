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
from serial import Serial
from libtiff import TIFF

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

# ser = Serial('/dev/ttyACM0', 9600,timeout=2) # Establish the connection on a specific port

#%%
monitor_list = monitors.getAllMonitors()

parser = optparse.OptionParser()

parser.add_option('-R', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")


parser.add_option('--no-camera', action="store_false", dest="acquire_images", default=True, help="just run PsychoPy protocol")
parser.add_option('--save-images', action="store_true", dest="save_images", default=False, help="save camera frames to disk")
parser.add_option('--output-format', action="store", dest="output_format", type="choice", choices=['tif', 'png', 'npz', 'pkl'], default='tif', help="out file format, tif | png | npz | pkl [default: png]")
parser.add_option('--use-pvapi', action="store_true", dest="use_pvapi", default=True, help="use the pvapi")
parser.add_option('--use-opencv', action="store_false", dest="use_pvapi", help="use some other camera")
parser.add_option('--fullscreen', action="store_true", dest="fullscreen", default=True, help="display full screen [defaut: True]")
parser.add_option('--debug-window', action="store_false", dest="fullscreen", help="don't display full screen, debug mode")
parser.add_option('--write-process', action="store_true", dest="save_in_separate_process", default=True, help="spawn process for disk-writer [default: True]")
parser.add_option('--write-thread', action="store_false", dest="save_in_separate_process", help="spawn threads for disk-writer")
parser.add_option('--monitor', action="store", dest="monitor", default="testMonitor", help=str(monitor_list))

# Stimulus params:
parser.add_option('-w', '--width', action="store", dest="bar_deg", default=1, help="Bar size in degrees (default: 1 deg)")
parser.add_option('-f', '--freq', action="store", dest="target_freq", default=0.13, help="stimulation frequency (default: 0.13)")
parser.add_option('-c', '--ncycles', action="store", dest="ncycles", default=20, help="Num cycles to show (default: 20)")
parser.add_option('-a', '--fps', action="store", dest="acquisition_rate", default=30., help="Acquisition rate of camera (default: 30)")
parser.add_option('--flash', action="store_true", dest="flash", default=False, help="Flash checkerboard inside bar?")
parser.add_option('--conds', action="store", dest="cond_str", default="", help="Comma-separated list of conds to run (default: all conditions)")
parser.add_option('-t', '--nreps', action="store", dest="nreps", default=3, help="Num reps per condition to run (default: 3)")



parser.add_option('--short-axis', action="store_false", dest="use_long_axis", default=True, help="Use short axis instead?")


# parser.add_option('--run-num', action="store", dest="run_num", default="1", help="run number for condition X")
(options, args) = parser.parse_args()

rootdir = options.rootdir
animalid = options.animalid
session = options.session
acquisition = options.acquisition

acquire_images = options.acquire_images
save_images = options.save_images
output_format = options.output_format
save_in_separate_process = options.save_in_separate_process
fullscreen = options.fullscreen
curr_monitor = options.monitor

nreps_per_cond = int(options.nreps)
cond_str = options.cond_str
cyc_per_sec = options.target_freq
flash = options.flash
ncycles = int(options.ncycles) # how many times to do the cycle of 1 condition
fps = float(options.acquisition_rate)
bar_width = options.bar_deg #8 #2 # bar width in degrees
flashPeriod = 0.2 #1.0 #0.2#0.2 #amount of time it takes for a full cycle (on + off)
dutyCycle = 0.5 #1.0 #0.5#0.5 #Amount of time flash bar is "on" vs "off". 0.5 will be 50% of the time.


#%%

# -------------------------------------------------------------
# Monitor params:
# -------------------------------------------------------------
if not fullscreen:
    winsize = [800, 600]
else:
    winsize = monitors.Monitor(curr_monitor).getSizePix()
use_pvapi = options.use_pvapi

print "WIN SIZE: ", winsize

screen_width_cm = monitors.Monitor(curr_monitor).getWidth()
screen_height_cm = (float(screen_width_cm)/monitors.Monitor(curr_monitor).getSizePix()[0])*monitors.Monitor(curr_monitor).getSizePix()[1]

screen_width_deg = tools.monitorunittools.cm2deg(screen_width_cm, monitors.Monitor(curr_monitor))
screen_height_deg = tools.monitorunittools.cm2deg(screen_height_cm, monitors.Monitor(curr_monitor))

use_width = options.use_long_axis #True
if use_width:
    total_length = max([screen_width_cm, screen_height_cm])
else:
    total_length = min([screen_width_cm, screen_height_cm])
print "Base Length (screen dim, cm):  ", total_length

total_length_deg = tools.monitorunittools.cm2deg(total_length, monitors.Monitor(curr_monitor))
distance = monitors.Monitor(curr_monitor).getDistance()

frames_per_cycle = fps/cyc_per_sec
bar_width_cm = tools.monitorunittools.deg2cm(bar_width, monitors.Monitor(curr_monitor))
print "Distance from monitor (cm): ", distance
print "Bar width (deg | cm): ", bar_width, ' | ', bar_width_cm

#%%
# -------------------------------------------------------------
# STIMULUS params by conditions:
# -------------------------------------------------------------
cond_labels = ['blank','left','right','top','bottom']

stimconfigs = dict((c, dict()) for c in cond_labels)
stimconfigs['left'] = {'condnum': 1,
                        'start_sign': -1, # start from LEFT/bottom (neg --> pos)
                        'angle': 90,      # 90 deg is vertical
                        'longside_cm': screen_height_cm,
                        'longside':  screen_height_deg,
                        'start_pos': left_start_pos,
                        'bar_color': 1
                        }
stimconfigs['right'] = {'condnum': 2,
                        'start_sign': 1, # start from RIGHT/top (pos --> neg)
                        'angle': 90,      # 90 deg is vertical
                        'longside_cm': screen_height_cm,
                        'longside':  screen_height_deg,
                        'start_pos': right_start_pos,
                        'bar_color': 1
                        }
stimconfigs['top'] = {'condnum': 3,
                        'start_sign': 1, # start from right/TOP (pos --> neg)
                        'angle': 0,      # 90 deg is vertical
                        'longside_cm': screen_width_cm,
                        'longside':  screen_width_deg,
                        'start_pos': top_start_pos,
                        'bar_color': 1
                        }
stimconfigs['bottom'] = {'condnum': 4,
                        'start_sign': -1, # start from left/BOTTOM (neg --> pos)
                        'angle': 0,       # 90 deg is vertical
                        'longside_cm': screen_width_cm,
                        'longside':  screen_width_deg,
                        'start_pos': bottom_start_pos,
                        'bar_color': 1
                        }
stimconfigs['blank'] = {'condnum': 0,
                        'start_sign': -1, # start from LEFT/bottom (neg --> pos)
                        'angle': 90,      # 90 deg is vertical
                        'longside_cm': screen_height_cm,
                        'longside':  screen_height_deg,
                        'start_pos': left_start_pos,
                        'bar_color': -1
                        }
for cond in cond_names:
    if stimconfig[cond]['start_pos'] is None:
        angle = stimconfigs[cond]['angle']
        longside = stimconfigs[cond]['long_side']
        start_sign = stimconfigs[cond]['start_sign']
        stim_size = (stimconfigs[cond]['longside'], bar_width) # First number is longer dimension no matter what the orientation is.
        unsign_start_point = (total_length_deg*0.5) + bar_width*0.5 # half the screen-size, plus hal bar-width to start with bar OFF screen
        start_point = start_sign * unsign_start_point
        end_point = -1 * start_point
        stimconfig[cond]['start_pos'] = start_point
        stimconfig[cond]['end_pos'] = end_point


# -----------------------------------------------------------------------------
# Create trial mat:
# -----------------------------------------------------------------------------
if len(cond_str) == 0:
    conds_to_run = cond_labels.copy()
else:
    conds_to_run = [i for i in cond_str.split(',')]
print "RUNNING CONDITIONS:"
for cond in conds_to_run:
    print stimconfig[cond]['condnum'], cond

condnums_to_run = [stimconfig[cond]['condnum'] for cond in conds_to_run]

trialmat = np.tile(condsnums_to_run, nreps_per_cond)


# -----------------------------------------------------------------------------
# Output Setup
# -----------------------------------------------------------------------------
if flash is True:
    bar_type = 'flash'
else:
    bar_type = 'solid'
curr_run = 'bar_%sHz_%s' % (str(cyc_per_sec).replace('.',''), bar_type)

acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)
if not os.path.exists(acquisition_dir):
    os.makedirs(acquisition_dir)

run_path = os.path.join(acquisition_dir, curr_run)
if not os.path.exists(output_path):
    os.makedirs(run_path)

print output_format
save_as_tif = False
save_as_png = False
save_as_npz = False
if output_format == 'png':
    save_as_png = True
elif output_format == 'npz':
    save_as_npz = True
else:
    save_as_tif = True

# -----------------------------------------------------------------------------
# Camera Setup
# -----------------------------------------------------------------------------
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




win_flag = 0
trial_idx = 0
while True:
    time.sleep(2)
    curr_trial = "trial%03d" % int(trial_idx+1)
    condnum = trialmat[trial_idx]
    curr_cond = cond_label[int(condnum)]

    print "Starting trial %i: %s (%s)" % (trial_idx, curr_trial, curr_cond)

    user_input=raw_input("\nEnter <s> to start, or 'exit':\n")
    if user_input=='exit':
        break
    time.sleep(2)#a couple of seconds for user to switch windows


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

        # Make the output path if it doesn't already exist
        trial_path = os.path.join(run_path, 'raw', curr_trial)
        frame_path = os.path.join(trial_path, 'frames')
        if not os.path.exists(frame_path):
            os.makedirs(frame_path)

        frame_log_file = open(os.path.join(trial_path, 'frame_info.txt')) #'framelog_%s_%s.txt') % (currdict['condname'], str(run_num)), 'w')
        frame_log_file.write('idx\tframenum\tcycnum\ttrial\tcurrcond\txpos\typos\ttstamp\n')

        while currdict is not None:

            frame_log_file.write(
                '%i\t%i\t%i\t%s\t%s\t%i\t%.4f\t%.4f\t%.4f\t%str\n' % (n, int(currdict['frame_num'], int(currdict['cycle_num'],
                                                                 curr_trial, currdict['curr_cond'], currdict['cond_num'],
                                                                 currdict['xpos'], currdict['ypos'], currdict['pos_linear'],
                                                                 str(currdict['time'])
            if save_as_npz:
                np.savez_compressed(os.path.join(trial_path, '%i.npz' % n), currdict['im'])
            else:
                if save_as_png:
                    fname = '%i.png' % n
                elif save_as_tif:
                    fname = '%i.tif' % n
                tiff = TIFF.open(os.path.join(frame_path, fname), mode='w')
                tiff.write_image(currdict['im'])
                tiff.close()

            n += 1
            currdict = im_queue.get()

        disk_writer_alive = False
        print('Disk-saving thread inactive...')
        # disk_writer_alive = False

    if save_in_separate_process:
        disk_writer = mp.Process(target=save_images_to_disk)
    else:
        disk_writer = threading.Thread(target=save_images_to_disk)

    # disk_writer.daemon = True

    if save_images:
        disk_writer.daemon = True
        disk_writer.start()


    # -------------------------------------------------------------
    # Psychopy stuff here (just lifted from a demo)
    # -------------------------------------------------------------
    globalClock = core.Clock()

    #make a window
    flash=options.flash
    if win_flag==0:
        if flash:
            # win = visual.Window(fullscr=fullscreen, color=(.5,.5,.5), size=winsize, units='deg', monitor=whichMonitor)
            win = visual.Window(fullscr=fullscreen, color=(-1,-1,-1), size=winsize, units='deg', monitor=curr_monitor)

        else:
            win = visual.Window(fullscr=fullscreen, color=(-1,-1,-1), size=winsize, units='deg', monitor=curr_monitor)

        win_flag=1


    t=0
    nframes = 0.
    frame_accumulator = 0
    flash_count = 0
    last_t = None

    report_period = 60 # frames
    refresh_rate = 60.000 #60.000

    if acquire_images:
        # Start acquiring
        win.flip()
        time.sleep(0.002)
        # camera.capture_start(frame_rate)
        camera.capture_start()
        camera.queue_frame()

    # RUN:
    getout = 0
    tstamps = []

# SPECIFICY CONDITION TYPES:

    center_point = [0,0] #center of screen is [0,0] (degrees).

    start_point = stimconfig[curr_cond]['start_pos']
    end_point = stimconfig[curr_cond]['end_pos'] #-1 * start_point
    start_to_end = end_point - start_point

    print "Cycle Travel LENGTH (deg): ", start_to_end
    print "START: ", start_point
    print "END: ", end_point
    print "Degrees per cycle: ", start_to_end # center-to-center #abs(start_point)*2. + bar_width
    SF = 1./(abs(start_point)*2. + bar_width)
    print "Calc SF (cpd): ", SF
    # cyc = 0

    # Movie params:
    cycle_duration = start_to_end / (start_to_end*cyc_per_sec)
    total_duration = cycle_duration * ncycles
    print "Cycle Travel TIME (s): ", cycle_duration
    print "TOTAL DUR: ", total_duration

    # 1. bar moves to this far from centerPoint (in degrees)
    # 2. bar starts & ends OFF the screen

    # CREATE THE STIMULUS:
    if flash is True:
        barmask = np.ones([1,1]) * bar_color
        bar1 = visual.GratingStim(win=win,tex='sqrXsqr', sf=.1, color=1*bar_color, mask=barmask,units='deg',pos=center_point,size=stim_size,ori=angle)
        bar2 = visual.GratingStim(win=win,tex='sqrXsqr', sf=.1, color=-1, mask=barmask,units='deg',pos=center_point,size=stim_size,ori=angle)

    else:
        bartex = np.ones([256,256,3])*bar_color;
        #bartex[:,:,0] = 0. # turn OFF red channel
        barStim = visual.PatchStim(win=win,tex=bartex,mask='none',units='deg',pos=center_point,size=stim_size,ori=angle)
        barStim.setAutoDraw(False)

    # DISPLAY LOOP:
    win.flip() # first clear everything
    # time.sleep(0.001) # wait a sec

    FORMAT = '%Y%m%d_%H%_M%_S_%f'
    frame_counter = 0
    cycnum = 0
    clock = core.Clock()
    while clock.getTime()<=total_duration: #frame_counter < frames_per_cycle*num_seq_reps: #endPoint - posLinear <= dist: #frame_counter <= frames_per_cycle*num_seq_reps:
        t = globalClock.getTime()

        if flash is True:
            if (clock.getTime()/flashPeriod) % (1.0) < dutyCycle:
                barStim = bar1
            else:
                barStim = bar2

        pos_linear = (clock.getTime() % cycle_duration) / cycle_duration * (end_point-start_point) + start_point #what pos we are at in degrees
        #print posLinear
        posX = pos_linear*math.sin(angle*math.pi/180)+center_point[0]
        posY = pos_linear*math.cos(angle*math.pi/180)+center_point[1]
        barStim.setPos([posX,posY])
        barStim.draw()
        win.flip()

        if acquire_images:
            im_array = camera.capture_wait()
            camera.queue_frame()
        else:
            im_array = np.zeros((winsize[0], winsize[1]))

        if save_images:
            fdict = dict()
            fdict['im'] = im_array
            fdict['cond_num'] = condnum
            fdict['cond_name'] = cond_label[int(condnum)]
            fdict['frame_num'] = frame_counter #nframes
            fdict['cycle_num'] = cycnum
            fdict['time'] = datetime.now().strftime(FORMAT)
            fdict['pos_linear'] = pos_linear
            fdict['xpos'] = posX
            fdict['ypos'] = posY

            im_queue.put(fdict)

        if nframes % report_period == 0:
        #if frame_accumulator % report_period == 0:
            if last_t is not None:
                print('avg frame rate: %f' % (report_period / (t - last_t)))
            last_t = t

        frame_counter += 1
        flash_count += 1
        cycnum += 1

        # Break out of the while loop if these keys are registered
        if event.getKeys(keyList=['escape', 'q']):
            getout = 1
            break

        print cycnum

    win.clearBuffer()
    win.flip()

    #print "TOTAL COND TIME: " + str(clock.getTime())
    # Break out of the FOR loop if these keys are registered
    if getout==1:
        break
    else:
        pass

    if acquire_images:
        camera.capture_end()
        # camera.close()

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

if acquire_images:
    camera.close()

win.close()
