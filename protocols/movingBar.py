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

#from skimage import io, exposure, img_as_uint
#io.use_plugin('freeimage')

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
parser.add_option('--output-format', action="store", dest="output_format", type="choice", choices=['png', 'npz', 'pkl', 'tif'], default='tif', help="out file format, tif | png | npz | pkl [default: tif]")
parser.add_option('--use-pvapi', action="store_true", dest="use_pvapi", default=True, help="use the pvapi")
parser.add_option('--use-opencv', action="store_false", dest="use_pvapi", help="use some other camera")
parser.add_option('--fullscreen', action="store_true", dest="fullscreen", default=True, help="display full screen [defaut: True]")
parser.add_option('--debug-window', action="store_false", dest="fullscreen", help="don't display full screen, debug mode")
parser.add_option('--write-process', action="store_true", dest="save_in_separate_process", default=True, help="spawn process for disk-writer [default: True]")
parser.add_option('--write-thread', action="store_false", dest="save_in_separate_process", help="spawn threads for disk-writer")
parser.add_option('--monitor', action="store", dest="whichMonitor", default="testMonitor", help=str(monitor_list))
parser.add_option('--run-num', action="store", dest="run_num", default="0", help="run number for condition X")
(options, args) = parser.parse_args()

acquire_images = options.acquire_images
save_images = options.save_images
output_path = options.output_path
output_format = options.output_format
run_num = options.run_num
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
    # while disk_writer_alive: 
    #     if not im_queue.empty():

    #         currdict = im_queue.get()
    #         currpath = '%s/%s/' % (output_path, currdict['condName'])

    #         if save_as_png:
    #             imsave('%s/test%d.png' % (output_path, n), currdict['im'])
    #         elif save_as_npz:
    #             np.savez_compressed('%s/test%d.npz' % (output_path, n), currdict['im'])
    #         else:
    #             # Make the output path if it doesn't already exist
    #             currpath = '%s/%s/' % (output_path, currdict['condName'])
    #             if not os.path.exists(currpath):
    #                 os.mkdir(currpath)

    #             # # Make the output path if it doesn't already exist
    #             # currpath = '%s/%s/' % (output_path, fdict['condName'])
    #             # try:
    #             #     os.mkdir(currpath)
    #             # except OSError, e:
    #             #     if e.errno != errno.EEXIST:
    #             #         raise e
    #             #     pass

    #             # fname = '%s/%s/00%i_%i.pkl' % (output_path, fdict['condName'], int(fdict['condNum']), int(fdict['reltime']))
    #             fname = '%s/%s/00%i_%i_%i_%i.pkl' % (output_path, currdict['condName'], int(currdict['condNum']), int(currdict['time']), int(currdict['frame']), int(n))

    #             with open(fname, 'wb') as f:
    #                 pkl.dump(fdict['im'], f)

    #         print 'DONE SAVING FRAME: ', currdict['frame'], n #fdict
    #         n += 1

    currdict = im_queue.get()
    # Make the output path if it doesn't already exist
    currpath = '%s/%s_%s/' % (output_path, currdict['condName'], str(run_num))
    if not os.path.exists(currpath):
	os.mkdir(currpath)

    while currdict is not None:
        if save_as_png:
#            # Make the output path if it doesn't already exist
#            currpath = '%s/%s_%s/' % (output_path, currdict['condName'], str(run_num))
#            if not os.path.exists(currpath):
#                os.mkdir(currpath)

            # fname = '%s/%s/00%i_%i_%i_%i_%ideg_%s.png' % (output_path, currdict['condName'], int(currdict['condNum']), int(currdict['time']), int(currdict['frame']), int(n), int(currdict['barWidth']), str(currdict['stimPos']))
            # img = scipy.misc.toimage(currdict['im'], high=65536, low=0, mode='I')
            # img.save(fname)

            fname = '%s/%s_%s/00%i_%i_%i_%i_%i_%ideg_%s.tif' % (output_path, currdict['condName'], str(run_num), int(currdict['condNum']), int(currdict['time']), int(currdict['frame']), int(n), int(currdict['frame_count']), int(currdict['barWidth']), str(currdict['stimPos']))
            #img = img_as_uint(currdict['im'])
            #io.imsave(fname, img)
            #img = scipy.misc.toimage(currdict['im'], cmax=65535, cmin=0, mode='I')
            
            #img = scipy.misc.toimage(currdict['im'], high=np.max(currdict['im']), low=np.min(currdict['im']), mode='I')
            #img.save(fname)
            tiff = TIFF.open(fname, mode='w')
            tiff.write_image(currdict['im'])
            tiff.close()

            # imsave(fname, currdict['im'])
            # imsave('%s/test%d.png' % (output_path, n), currdict['im'])

	elif save_as_tif:
            fname = '%s/%s_%s/00%i_%i_%i_%i_%i_%ideg_%s.tif' % (output_path, currdict['condName'], str(run_num), int(currdict['condNum']), int(currdict['time']), int(currdict['frame']), int(n), int(currdict['frame_count']), int(currdict['barWidth']), str(currdict['stimPos']))

            tiff = TIFF.open(fname, mode='w')
            tiff.write_image(currdict['im'])
            tiff.close()


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
win = visual.Window(fullscr=fullscreen, rgb=-1, size=winsize, units='deg', monitor=whichMonitor)

# SET CONDITIONS:
num_cond_reps = 20 #20 # 8 how many times to run each condition
num_seq_reps = 1 # how many times to do the cycle of 1 condition
# conditionTypes = ['1', '2', '3', '4']
# can either run 1 cycle many times, or repmat:
conditionTypes = ['4']
condLabel = ['V-Left','V-Right','H-Down','H-Up']
# conditionMatrix = sample_permutations_with_duplicate_spacing(conditionTypes, len(conditionTypes), num_cond_reps) # constrain so that at least 2 diff conditions separate repeats
conditionMatrix = []
for i in conditionTypes:
    conditionMatrix.append([np.tile(i, num_cond_reps)])
conditionMatrix = list(itertools.chain(*conditionMatrix))
conditionMatrix = sorted(list(itertools.chain(*conditionMatrix)), key=natural_keys)
# print conditionMatrix


#input parameters 
cyc_per_sec = 0.08 
screen_width_cm = monitors.Monitor(whichMonitor).getWidth()
screen_height_cm = (float(screen_width_cm)/monitors.Monitor(whichMonitor).getSizePix()[0])*monitors.Monitor(whichMonitor).getSizePix()[1]
total_length = max([screen_width_cm, screen_height_cm])
#print screen_width_cm
#print screen_height_cm
#print total_length


#time parameters
fps = 60.
total_time = total_length/(total_length*cyc_per_sec) #how long it takes for a bar to move from startPoint to endPoint
frames_per_cycle = fps/cyc_per_sec
distance = monitors.Monitor(whichMonitor).getDistance()

duration = total_time*num_seq_reps; #how long to run the same condition for (seconds)

#flashing parameters
flashPeriod = 1. #0.2#0.2 #amount of time it takes for a full cycle (on + off)
dutyCycle = 1. #0.5#0.5 #Amount of time flash bar is "on" vs "off". 0.5 will be 50% of the time.

# SPECIFY STIM PARAMETERS
barColor = 1 # starting 1 for white, -1 for black, 0.5 for low contrast white, etc.
barWidth = 1 # bar width in degrees 

#blackBar = (0,0,0)
#whiteBar = (255,255,255)

print "DUR: ", duration

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
frame_rate = 180.000
refresh_rate = 60.000

if acquire_images:
    # Start acquiring
    win.flip()
    time.sleep(0.002)
    camera.capture_start(frame_rate)
    camera.queue_frame()


# #wait for serial
# if useSerialTrigger==1:
#     bytes = ""
#     while(bytes == ""):
#         bytes = ser.read(1)
        

# RUN:
getout = 0
tstamps = []
cyc = 1
for condType in conditionMatrix:
    # print condType
    # print condLabel[int(condType)-1]

    if cyc == 1:
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
        # barColor = 1 # 1 for white, -1 for black, 0.5 for low contrast white, etc.
        # barWidth = 1 # bar width in degrees 
        # fdict['barWidth'] = barWidth

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

        # startPoint = startSign*uStartPoint; #bar starts this far from centerPoint (in degrees)
        # # currently, the endPoint is set s.t. the same total distance is traveled regardless of V or H bar
        # # endPoint = -1*(startPoint + startSign*(total_length_deg*0.5-uStartPoint+barWidth*0.5))
        # endPoint = -1*(startPoint + startSign*(total_length_deg*0.5-uStartPoint))
        # dist = endPoint - startPoint
        # #print dist
        # # 1. bar moves to this far from centerPoint (in degrees)
        # # 2. bar starts & ends OFF the screen

        # CREATE THE STIMULUS:
        # barTexture = numpy.ones([256,256,3])*barColor;
        # barStim = visual.PatchStim(win=win,tex=barTexture,mask='none',units='deg',pos=centerPoint,size=stimSize,ori=angle)
        # barStim.setAutoDraw(False)

        # DISPLAY LOOP:
        # win.flip() # first clear everything
        # # time.sleep(0.001) # wait a sec

        startPoint = startSign*uStartPoint +barWidth*0.5; #bar starts this far from centerPoint (in degrees)
        # currently, the endPoint is set s.t. the same total distance is traveled regardless of V or H bar
        # endPoint = -1*(startPoint + startSign*(total_length_deg*0.5-uStartPoint+barWidth*0.5))
        endPoint = -1*(startPoint + startSign*(total_length_deg*0.5-uStartPoint - barWidth*0.5))
        dist = endPoint - startPoint
        print "DIST: ", dist
        cyc = 0

    # 1. bar moves to this far from centerPoint (in degrees)
    # 2. bar starts & ends OFF the screen

    # CREATE THE STIMULUS:
    barTexture = numpy.ones([256,256,3])*barColor;
    barTexture[:,:,0] = 0. # IS THIS HERE?!
    barStim = visual.PatchStim(win=win,tex=barTexture,mask='none',units='deg',pos=centerPoint,size=stimSize,ori=angle)
    barStim.setAutoDraw(False)


    # DISPLAY LOOP:
    win.flip() # first clear everything
    # time.sleep(0.001) # wait a sec

    FORMAT = '%Y%m%d%H%M%S%f'
    clock = core.Clock()
    frame_counter = 0
    count_frames = 0
    # posLinear = startPoint
    # print posLinear
    # print endPoint
    # FORMAT = '%Y%m%d%H%M%S%f'
    # datetime.now().strftime(FORMAT)

    while clock.getTime()<=duration: #frame_counter < frames_per_cycle*num_seq_reps: #endPoint - posLinear <= dist: #frame_counter <= frames_per_cycle*num_seq_reps: 
        t = globalClock.getTime()

        # if (clock.getTime()/flashPeriod) % (1.0) < dutyCycle:
        #     barStim.setContrast(1)
        #     #barStim.setColor(whiteBar, 'rgb255')
        # else:
        #     barStim.setContrast(0)
        #     #barStim.setColor(blackBar, 'rgb255')

        posLinear = (clock.getTime() % total_time) / total_time * (endPoint-startPoint) + startPoint; #what pos we are at in degrees
        # print posLinear
        posX = posLinear*math.sin(angle*math.pi/180)+centerPoint[0]
        posY = posLinear*math.cos(angle*math.pi/180)+centerPoint[1]
        barStim.setPos([posX,posY])
        barStim.draw()
        win.flip()

        lastT = clock.getTime()

        fdict = dict()
    	#fdict['im'] = []

        if acquire_images:
            #fdict = dict()
    #	   while clock.getTime() - lastT + 1./refresh_rate <= 1./refresh_rate:
    #		fdict = dict()
    #		fdict['im'] = camera.capture_wait() #append(camera.capture_wait())
    #		camera.queue_frame()
    #		count_frames += 1


            while (clock.getTime() - lastT + (1./frame_rate)) <= (1./refresh_rate):
            #for fr_idx in range(int(frame_rate/refresh_rate)):
            #count_frames += 1
            #print clock.getTime() - lastT
                fdict['im'] = camera.capture_wait() #.append(camera.capture_wait())
                camera.queue_frame()

                #                try:
                #                    fdict['im'].append(camera.capture_wait())
                #                except KeyError:
                #                    fdict['im'] = [camera.capture_wait()]
                #            #im_array = #camera.capture_wait()
                #            camera.queue_frame()
                #print count_frames

                if save_images:
                    #fdict = dict()
                    #fdict['im'] = im_array
                    fdict['barWidth'] = barWidth
                    fdict['condNum'] = condType
                    fdict['condName'] = condLabel[int(condType)-1]
                    fdict['frame'] = frame_counter #nframes
                    fdict['frame_count'] = count_frames
                    #print 'frame #....', frame_counter
                    fdict['time'] = datetime.now().strftime(FORMAT)
                    fdict['stimPos'] = [posX,posY]

                    im_queue.put(fdict)

                    frame_accumulator += 1

                count_frames += 1

            #print fr
            # if save_as_dict:
            #     fdict['im'] = im_array
            #     im_queue.put(fdict)
            # else:
            #     im_queue.put(im_array)



        #if count_frames % report_period == 0:
        if frame_accumulator % report_period == 0:
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

    #print "TOTAL COND TIME: " + str(clock.getTime())
    # Break out of the FOR loop if these keys are registered        
    if getout==1:
        break
    else:
        continue

    cyc += 1

win.close() 

if acquire_images:
    camera.capture_end()
    camera.close()

# fdict['im'] = None
# fdict = None
print "GOT HERE"
im_queue.put(None)



# if save_images:
#     hang_time = time.time()
#     nag_time = 0.05

#     sys.stdout.write('Waiting for disk writer to catch up (this may take a while)...')
#     sys.stdout.flush()
#     waits = 0
#     while not im_queue.empty():
#         now = time.time()
#         if (now - hang_time) > nag_time:
#             sys.stdout.write('.')
#             sys.stdout.flush()
#             hang_time = now
#             waits += 1

#     print waits
#     print("\n")

#     if not im_queue.empty():
#         print("WARNING: not all images have been saved to disk!")

#     disk_writer_alive = False

#     if save_in_separate_process and disk_writer is not None:
#         print("Terminating disk writer...")
#         disk_writer.join()
#         disk_writer.terminate()
    
#     # disk_writer.join()
#     print('Disk writer terminated')





if save_images:
    hang_time = time.time()
    # nag_time = 0.05
    nag_time = 2.0

    sys.stdout.write('Waiting for disk writer to catch up (this may take a while)...')
    sys.stdout.flush()
    # waits = 0
    # while not im_queue.empty():
    #     now = time.time()
    #     if (now - hang_time) > nag_time:
    #         sys.stdout.write('.')
    #         sys.stdout.flush()
    #         hang_time = now
    #         waits += 1

    # print waits

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
    

