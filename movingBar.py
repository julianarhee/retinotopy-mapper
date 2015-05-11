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
    # sample_seq = seq + seq        
    random.shuffle(sample_seq)    
    while not valid_duplicate_spacing(sample_seq, nconds):
        random.shuffle(sample_seq)
    return sample_seqgithub



monitor_list = monitors.getAllMonitors()

parser = optparse.OptionParser()
parser.add_option('--no-camera', action="store_false", dest="acquire_images", default=True, help="just run PsychoPy protocol")
parser.add_option('--save-images', action="store_true", dest="save_images", default=False, help="save camera frames to disk")
parser.add_option('--output-path', action="store", dest="output_path", default="/tmp/frames", help="out path directory [default: /tmp/frames]")
parser.add_option('--output-format', action="store", dest="output_format", type="choice", choices=['png', 'npz'], default='png', help="out file format, png or npz [default: png]")
parser.add_option('--use-pvapi', action="store_true", dest="use_pvapi", default=True, help="use the pvapi")
parser.add_option('--use-opencv', action="store_false", dest="use_pvapi", help="use some other camera")
parser.add_option('--fullscreen', action="store_true", dest="fullscreen", default=True, help="display full screen [defaut: True]")
parser.add_option('--debug-window', action="store_false", dest="fullscreen", help="don't display full screen, debug mode")
parser.add_option('--write-process', action="store_true", dest="save_in_separate_process", default=True, help="spawn process for disk-writer [default: True]")
parser.add_option('--write-thread', action="store_false", dest="save_in_separate_process", help="spawn threads for disk-writer")
parser.add_option('--monitor', action="store", dest="whichMonitor", default="testMonitor", help=str(monitor_list))
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

if not acquire_images:
    save_images = False

save_as_png = False
save_as_npz = False
if output_format == 'png':
    save_as_png = True
elif output_format == 'npz':
    save_as_npz = True

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

# TODO:  fix this so that I can create new processes (image queue / disk-writer)
# for each condition, since it's easier to read in data corresponding to condition X that way.

if save_in_separate_process:
    im_queue = mp.Queue()
else:
    im_queue = Queue()

disk_writer_alive = True

def save_images_to_disk(output_path):
    print('Disk-saving thread active...')
    n = 0
    while disk_writer_alive: 
        if not im_queue.empty():
            im_array = im_queue.get()
            if save_as_png:
                imsave('%s/test%d.png' % (output_path, n), im_array)
            else:
                np.savez_compressed('%s/test%d.npz' % (output_path, n), im_array)
            n += 1
    print('Disk-saving thread inactive...')


if save_in_separate_process:
    disk_writer = mp.Process(target=save_images_to_disk, args=(output_path,))
else:
    disk_writer = threading.Thread(target=save_images_to_disk, args=(output_path,))

# disk_writer.daemon = True

if save_images:
    disk_writer.daemon = True
    disk_writer.start()


# -------------------------------------------------------------
# Psychopy stuff here (just lifted from a demo)
# -------------------------------------------------------------

globalClock = core.Clock()

#make a window
win = visual.Window(fullscr=fullscreen, size=winsize, units='deg', monitor=whichMonitor)

# SET CONDITIONS:
num_cond_reps = 2 # how many times to run each condition
num_seq_reps = 1 # how many times to do the cycle of 1 condition
conditionTypes = ['1', '2', '3', '4']
# conditionTypes = ['1']
condLabel = ['V-Left','V-Right','H-Down','H-Up']
# conditionMatrix = sample_permutations_with_duplicate_spacing(conditionTypes, len(conditionTypes), num_cond_reps) # constrain so that at least 2 diff conditions separate repeats
conditionMatrix = []
for i in conditionTypes:
    conditionMatrix.append([np.tile(i, num_cond_reps)])
conditionMatrix = list(itertools.chain(*conditionMatrix))
conditionMatrix = sorted(list(itertools.chain(*conditionMatrix)), key=natural_keys)
print conditionMatrix


#input parameters 
cyc_per_sec = 0.1 # 
screen_width_cm = monitors.Monitor(whichMonitor).getWidth()
screen_height_cm = (float(screen_width_cm)/monitors.Monitor(whichMonitor).getSizePix()[0])*monitors.Monitor(whichMonitor).getSizePix()[1]
total_length = max([screen_width_cm, screen_height_cm])
print screen_width_cm
print screen_height_cm
print total_length

fps = 60.
total_time = total_length/(total_length*cyc_per_sec) #how long it takes for a bar to move from startPoint to endPoint
frames_per_cycle = fps/cyc_per_sec
distance = monitors.Monitor(whichMonitor).getDistance()

#time parameters
duration = total_time*num_seq_reps; #how long to run the same condition for (seconds)

# # serial port / trigger info
# useSerialTrigger = 0 #0=run now, 1=wait for serial port trigger
# ser = None
# if useSerialTrigger==1:
#     ser = serial.Serial('/dev/tty.pci-serial1', 9600, timeout=1) 
#     bytes = "1" 
#     while bytes: #burn up any old bits that might be lying around in the serial buffer
#         bytes = ser.read() 


t=0
nframes = 0
frame_accumulator = 0
last_t = None

report_period = 60 # frames

if acquire_images:
    # Start acquiring
    win.flip()
    time.sleep(0.002)
    camera.capture_start()
    camera.queue_frame()


# #wait for serial
# if useSerialTrigger==1:
#     bytes = ""
#     while(bytes == ""):
#         bytes = ser.read(1)
        

# RUN:
getout = 0
tstamps = []
for condType in conditionMatrix:
    print condType
    print condLabel[int(condType)-1]

    # SAVE EACH CONDITION TO SEPARATE DIRECTORY, MAKE NEW PROCESS:
    # if save_images:
    #     currDir = output_path + '/' + condLabel[int(condType)-1]
    #     print currDir
    #     if not os.path.exists(currDir):
    #         os.mkdir(currDir)

    #         if save_in_separate_process:
    #             disk_writer = mp.Process(target=save_images_to_disk, args=(currDir,))
    #         else:
    #             disk_writer = threading.Thread(target=save_images_to_disk, args=(currDir,))

    #         # disk_writer.daemon = True

    #         if save_images:
    #             disk_writer.daemon = True
    #             disk_writer.start()


    # SPECIFICY CONDITION TYPES:
    if condType == '1':
        orientation = 1 # 1 = VERTICAL, 0 = horizontal
        direction = 1 # 1 = start from LEFT or BOTTOM (neg-->pos), 0 = start RIGHT or TOP (pos-->neg)


    elif condType == '2':
        orientation = 1 # vertical
        direction = 0 # start from RIGHT

    elif condType == '3':
        orientation = 0 # horizontal
        direction = 0 # start from TOP

    elif condType == '4':
        orientation = 0 # horizontal
        direction = 1 # start from BOTTOM

    # SPECIFY STIM PARAMETERS
    barColor = 1 # 1 for white, -1 for black, 0.5 for low contrast white, etc.
    barWidth = 1 # bar width in degrees 
    if orientation==1:
        angle = 90 #0 is horizontal, 90 is vertical. 45 goes from up-left to down-right.
        longside = tools.monitorunittools.cm2deg(screen_height_cm, monitors.Monitor(whichMonitor)) #screen_height_cm
        # travelDist = screen_width_cm*0.5 # Half the travel distance (magnitude, no sign)
        width_deg = tools.monitorunittools.cm2deg(screen_width_cm, monitors.Monitor(whichMonitor))
        travelDist = width_deg*0.5
    else:
        angle = 0
        longside = tools.monitorunittools.cm2deg(screen_width_cm, monitors.Monitor(whichMonitor)) #screen_width_cm
        # travelDist = screen_height_cm*0.5 # Half the travel distance (magnitude, no sign)
        height_deg = tools.monitorunittools.cm2deg(screen_height_cm, monitors.Monitor(whichMonitor))
        travelDist = height_deg*0.5

    # uStartPoint = tools.monitorunittools.cm2deg(travelDist, monitors.Monitor(whichMonitor)) + barWidth*0.5
    total_length_deg = tools.monitorunittools.cm2deg(total_length, monitors.Monitor(whichMonitor))
    stimSize = (longside,barWidth) # First number is longer dimension no matter what the orientation is.
    # uStartPoint = travelDist + barWidth*0.5 
    uStartPoint = travelDist

    #position parameters
    centerPoint = [0,0] #center of screen is [0,0] (degrees).
    if direction==1: # START FROM NEG, go POS (start left-->right, or start bottom-->top)
        startSign = -1
    else:
        startSign = 1
    startPoint = startSign*uStartPoint; #bar starts this far from centerPoint (in degrees)
    # currently, the endPoint is set s.t. the same total distance is traveled regardless of V or H bar
    # endPoint = -1*(startPoint + startSign*(total_length_deg*0.5-uStartPoint+barWidth*0.5))
    endPoint = -1*(startPoint + startSign*(total_length_deg*0.5-uStartPoint))
    dist = endPoint - startPoint
    print dist
    # 1. bar moves to this far from centerPoint (in degrees)
    # 2. bar starts & ends OFF the screen

    # CREATE THE STIMULUS:
    barTexture = numpy.ones([256,256,3])*barColor;
    barStim = visual.PatchStim(win=win,tex=barTexture,mask='none',units='deg',pos=centerPoint,size=stimSize,ori=angle)
    barStim.setAutoDraw(False)


    # DISPLAY LOOP:
    win.flip() # first clear everything
    # time.sleep(0.001) # wait a sec

    clock = core.Clock()
    frame_counter = 0
    # posLinear = startPoint
    # print posLinear
    # print endPoint

    while clock.getTime()<=duration: #frame_counter < frames_per_cycle*num_seq_reps: #endPoint - posLinear <= dist: #frame_counter <= frames_per_cycle*num_seq_reps: 
        t = globalClock.getTime()

        posLinear = (clock.getTime() % total_time) / total_time * (endPoint-startPoint) + startPoint; #what pos we are at in degrees
        # print posLinear
        posX = posLinear*math.sin(angle*math.pi/180)+centerPoint[0]
        posY = posLinear*math.cos(angle*math.pi/180)+centerPoint[1]
        barStim.setPos([posX,posY])
        barStim.draw()
        win.flip()
        # dt = datetime.now()
        tstamps.append(datetime.now())

        if acquire_images:
            im_array = camera.capture_wait()
            camera.queue_frame()

            if save_images:
                im_queue.put(im_array)

        if nframes % report_period == 0:
            if last_t is not None:
                print('avg frame rate: %f' % (report_period / (t - last_t)))
            last_t = t

        nframes += 1
        frame_counter += 1

        # Break out of the while loop if these keys are registered
        if event.getKeys(keyList=['escape', 'q']):
            getout = 1
            break  

    print "TOTAL COND TIME: " + str(clock.getTime())
    # Break out of the FOR loop if these keys are registered        
    if getout==1:
        break
    else:
        continue

win.close() 
# pfile = open('/Volumes/MAC/data/timestamps.pkl', 'wb')
# pkl.dump(tstamps, pfile)
# pfile.close()


if acquire_images:
    camera.capture_end()
    camera.close()


if save_images:
    hang_time = time.time()
    nag_time = 0.05

    sys.stdout.write('Waiting for disk writer to catch up (this may take a while)...')
    sys.stdout.flush()
    waits = 0
    while not im_queue.empty():
        now = time.time()
        if (now - hang_time) > nag_time:
            sys.stdout.write('.')
            sys.stdout.flush()
            hang_time = now
            waits += 1

    print waits
    print("\n")

    if not im_queue.empty():
        print("WARNING: not all images have been saved to disk!")

    disk_writer_alive = False

    if save_in_separate_process and disk_writer is not None:
        print("Terminating disk writer...")
        disk_writer.join()
        disk_writer.terminate()
    
    # disk_writer.join()
    print('Disk writer terminated')
    

