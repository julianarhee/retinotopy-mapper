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
parser.add_option('--output-format', action="store", dest="output_format", type="choice", choices=['png', 'npz', 'pkl'], default='pkl', help="out file format, png | npz | pkl [default: png]")
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
print output_format

if not acquire_images:
    save_images = False

save_as_png = False
save_as_npz = False
save_as_dict = False
if output_format == 'png':
    save_as_png = True
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
    while currdict is not None:
        if save_as_png:
            # Make the output path if it doesn't already exist
            currpath = '%s/%s/' % (output_path, currdict['condName'])
            if not os.path.exists(currpath):
                os.mkdir(currpath)

            fname = '%s/%s/00%i_%i_%i_%i_%s_%s.png' % (output_path, currdict['condName'], int(currdict['flashRate']), int(currdict['time']), int(currdict['frame']), int(n), str(currdict['barWidth']), str(currdict['contrast1_2']))
            #img = scipy.misc.toimage(currdict['im'], high=65536, low=0, mode='I')
            #img.save(fname)
            tiff = TIFF.open(fname, mode='w')
            tiff.write_image(currdict['im'])
            tiff.close()

            # imsave(fname, currdict['im'])
            # imsave('%s/test%d.png' % (output_path, n), currdict['im'])
        elif save_as_npz:
            np.savez_compressed('%s/test%d.npz' % (output_path, n), currdict['im'])
            # np.savez_compressed('%s/test%d.npz' % (output_path, n), fdict['im'])
        else:
            # Make the output path if it doesn't already exist
            currpath = '%s/%s/' % (output_path, currdict['condName'])
            if not os.path.exists(currpath):
                os.mkdir(currpath)

            fname = '%s/%s/00%i_%i_%i_%i.pkl' % (output_path, currdict['condName'], int(currdict['condNum']), int(currdict['time']), int(currdict['frame']), int(n))
            with open(fname, 'wb') as f:
                pkl.dump(currdict, f, protocol=pkl.HIGHEST_PROTOCOL) #protocol=pkl.HIGHEST_PROTOCOL)

        #if n % 100 == 0:
        #print 'DONE SAVING FRAME: ', currdict['frame'], n #fdict
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
win = visual.Window(fullscr=fullscreen, size=winsize, units='deg', monitor=whichMonitor)

# SET CONDITIONS:
num_cond_reps = 1 #20 # 8 how many times to run each condition
num_seq_reps = 20 # how many times to do the cycle of 1 condition
# conditionTypes = ['1', '2', '3', '4']
conditionTypes = ['1']
condLabel = ['flashbars'] #['V-Left','V-Right','H-Down','H-Up']
# conditionMatrix = sample_permutations_with_duplicate_spacing(conditionTypes, len(conditionTypes), num_cond_reps) # constrain so that at least 2 diff conditions separate repeats
conditionMatrix = []
for i in conditionTypes:
    conditionMatrix.append([np.tile(i, num_cond_reps)])
conditionMatrix = list(itertools.chain(*conditionMatrix))
conditionMatrix = sorted(list(itertools.chain(*conditionMatrix)), key=natural_keys)
print conditionMatrix


#input parameters 
cyc_per_sec = 0.5 # for flashBar, this just makes the run 10sec*numReps (20)
screen_width_cm = monitors.Monitor(whichMonitor).getWidth()
screen_height_cm = (float(screen_width_cm)/monitors.Monitor(whichMonitor).getSizePix()[0])*monitors.Monitor(whichMonitor).getSizePix()[1]
total_length = max([screen_width_cm, screen_height_cm])
print screen_width_cm
print screen_height_cm
print total_length


#time parameters
fps = 60.
total_time = total_length/(total_length*cyc_per_sec) #how long it takes for a bar to move from startPoint to endPoint
frames_per_cycle = fps/cyc_per_sec
distance = monitors.Monitor(whichMonitor).getDistance()

duration = total_time*num_seq_reps; #how long to run the same condition for (seconds)

#flashing parameters
flashPeriod = 0.5 #0.2#0.2 #amount of time it takes for a full cycle (on + off) #seconds for one B-W cycle (ie 1/Hz)

#flashFrequency = 5 # number of flashes per second
dutyCycle = 0.5 #0.5#0.5 #Amount of time flash bar is "on" vs "off". 0.5 will be 50% of the time.

# SPECIFY STIM PARAMETERS
barColor = 1 # starting 1 for white, -1 for black, 0.5 for low contrast white, etc.
barWidth = 1 # bar width in degrees 

blackBar = (0,0,0)
whiteBar = (255,255,255)

# # serial port / trigger info
# useSerialTrigger = 0 #0=run now, 1=wait for serial port trigger
# ser = None
# if useSerialTrigger==1:
#     ser = serial.Serial('/dev/tty.pci-serial1', 9600, timeout=1) 
#     bytes = "1" 
#     while bytes: #burn up any old bits that might be lying around in the serial buffer
#         bytes = ser.read() 


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


# #wait for serial
# if useSerialTrigger==1:
#     bytes = ""
#     while(bytes == ""):
#         bytes = ser.read(1)
        

# RUN:
getout = 0
cyc = 1
for condType in conditionMatrix:
        #print condType
        #print condLabel[int(condType)-1]

    if cyc == 1:
        # SPECIFICY CONDITION TYPES:
        angle = 90

        #position parameters
        centerPoint = [0,0] #center of screen is [0,0] (degrees).

        # CREATE THE STIMULUS:
        screen_width_deg = tools.monitorunittools.cm2deg(screen_width_cm, monitors.Monitor(whichMonitor))
        screen_height_deg = tools.monitorunittools.cm2deg(screen_height_cm, monitors.Monitor(whichMonitor))
        
        flashFrequency = 8.
        stimSize = (screen_height_deg*0.5, screen_width_deg*0.1)
        print stimSize[0]
        print stimSize[1]
        stepsize = np.diff(np.linspace(1, 0, frames_per_cycle*0.5, endpoint=True))[0]

        contrast1 = 1.
        contrast2 = -1.
        barTexture = numpy.ones([256,256,3]);
        barTexture[:,:,0] = 0.
        centerLeft = [centerPoint[0]-int(stimSize[1])*4, 0]
        centerRight = [centerPoint[0]+int(stimSize[1])*4, 0]
        barStim1 = visual.PatchStim(win=win,tex=barTexture,mask='none',units='deg',pos=centerLeft,size=stimSize,ori=angle)
        barStim1.setAutoDraw(False)
        barStim2 = visual.PatchStim(win=win,tex=barTexture,mask='none',units='deg',pos=centerRight,size=stimSize,ori=angle)
        barStim2.setAutoDraw(False)

        # DISPLAY LOOP:
        win.flip() # first clear everything
        # time.sleep(0.001) # wait a sec

    # clock = core.Clock()
    frame_counter = 0
    # posLinear = startPoint
    # print posLinear
    # print endPoint
    FORMAT = '%Y%m%d%H%M%S%f'

    hits = []
    print duration
    clock = core.Clock()
    while clock.getTime()<=duration: #frame_counter < frames_per_cycle*num_seq_reps: #endPoint - posLinear <= dist: #frame_counter <= frames_per_cycle*num_seq_reps: 
        t = globalClock.getTime()
        
        # if (clock.getTime()*1) % (1.0) < switchFrequency:
        #     contrast1 = 1.0
        #     contrast2 = 0.0
        # else:
        #     contrast1 = 0.5
        #     contrast2 = 0.5

        # get_contrast = (clock.getTime() % total_time)/ total_time 
        # curr_contrast =  2*(1 - (0.5+get_contrast)) 
        # # if contrast1 == 1.:
        # #     hits.append(clock.getTime())

        # if (clock.getTime()*flashFrequency) % (1.0) < 0.5:
        #     #barStim.setContrast(contrast1)
        #     barStim.setContrast( curr_contrast )
        # else:
        #     #barStim.setContrast( -1* contrast1 )
        #     barStim.setContrast( -1 * curr_contrast )
        # #print [curr_contrast, -1*curr_contrast]

        if (clock.getTime()/flashPeriod) % (1.0) < dutyCycle:
            # barStim.setContrast(1)
            contrast1 = -1
            contrast2 = 1 #*contrast1
            # barStim1.setContrast(-1)
            # barStim2.setContrast(1)
        else:
            # barStim.setContrast(0)
            contrast1 = 1
            contrast2 = -1 #*contrast1
            # barStim1.setContrast(1)
            # barStim2.setContrast(-1)

        barStim1.setContrast(contrast1)
        barStim2.setContrast(contrast2)
        #posLinear = (clock.getTime() % total_time) / total_time * (endPoint-startPoint) + startPoint; #what pos we are at in degrees
        # print posLinear
        #posX = posLinear*math.sin(angle*math.pi/180)+centerPoint[0]
        #posY = posLinear*math.cos(angle*math.pi/180)+centerPoint[1]
        #barStim.setPos([posX,posY])
        barStim1.draw()
        barStim2.draw()
        win.flip()

        if acquire_images:
            im_array = camera.capture_wait()
            camera.queue_frame()

            if save_images:
                fdict = dict()
                fdict['im'] = im_array
                fdict['barWidth'] = stimSize
                fdict['flashRate'] = 1./flashPeriod
                fdict['condName'] = 'barflash' #condLabel[int(condType)-1]
                fdict['frame'] = frame_counter
                # print 'frame #....', frame_counter
                fdict['time'] = datetime.now().strftime(FORMAT)
                fdict['contrast1_2'] = [contrast1, contrast2]


                im_queue.put(fdict)
                # if save_as_dict:
                #     fdict['im'] = im_array
                #     im_queue.put(fdict)
                # else:
                #     im_queue.put(im_array)


        if nframes % report_period == 0:
            if last_t is not None:
                print('avg frame rate: %f' % (report_period / (t - last_t)))
            last_t = t

        # if (contrast1+stepsize) < -1:
        #     contrast1 = -1 * contrast1
        #     hits.append(clock.getTime())
        # contrast1 += stepsize
        # # contrast2 = -1 * contrast1

        nframes += 1
        frame_counter += 1
        flash_count += 1

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

    cyc += 1

print hits

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
    

