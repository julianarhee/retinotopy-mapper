#!/usr/bin/env python2
#rotate flashing wedge
from psychopy import visual, event, core, monitors, logging
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

if save_in_separate_process:
    im_queue = mp.Queue()
else:
    im_queue = Queue()

disk_writer_alive = True

def save_images_to_disk():
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
win = visual.Window(fullscr=fullscreen, size=winsize, units='deg', monitor=whichMonitor)

#input parameters 

#time parameters
duration = 60*20; #how long to run the stim for (seconds)
movementPeriod = 30; #how long it takes for a bar to move from startPoint to endPoint

#bar parameters
orientation = 90 #0 is horizontal, 90 is vertical. 45 goes from up-left to down-right.
barColor = 1 #1 for white, -1 for black, 0.5 for low contrast white, etc.

#position parameters
centerPoint = [0,0] #center of screen is [0,0] (degrees).
startPoint = -45; #bar starts this far from centerPoint (in degrees)
endPoint = 45; #bar moves to this far from centerPoint (in degrees)
stimSize = (180,4) #First number is longer dimension no matter what the orientation is.

useSerialTrigger = 0 #0=run now, 1=wait for serial port trigger

#serial port info
ser = None
if useSerialTrigger==1:
    ser = serial.Serial('/dev/tty.pci-serial1', 9600, timeout=1) 
    bytes = "1" 
    while bytes: #burn up any old bits that might be lying around in the serial buffer
        bytes = ser.read() 

#create bar stim
barTexture = numpy.ones([256,256,3])*barColor;
barStim = visual.PatchStim(win=win,tex=barTexture,mask='none',units='deg',pos=centerPoint,size=stimSize,ori=orientation)
barStim.setAutoDraw(True)


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


#wait for serial
if useSerialTrigger==1:
    bytes = ""
    while(bytes == ""):
        bytes = ser.read(1)
        

#run
clock = core.Clock()
while clock.getTime()<duration:

    t = globalClock.getTime()

    posLinear = (clock.getTime() % movementPeriod) / movementPeriod * (endPoint-startPoint) + startPoint; #what pos we are at in degrees
    posX = posLinear*math.sin(orientation*math.pi/180)+centerPoint[0]
    posY = posLinear*math.cos(orientation*math.pi/180)+centerPoint[1]
    barStim.setPos([posX,posY])
    win.flip()

    nframes += 1

    if acquire_images:
        im_array = camera.capture_wait()
        camera.queue_frame()

        if save_images:
            im_queue.put(im_array)

    if nframes % report_period == 0:
        if last_t is not None:
            print('avg frame rate: %f' % (report_period / (t - last_t)))
        last_t = t

    # Break out of the while loop if these keys are registered
    if event.getKeys(keyList=['escape', 'q']):
        break           


win.close() 


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
        

