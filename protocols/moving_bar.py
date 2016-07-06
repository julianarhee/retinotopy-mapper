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


# ser = Serial('/dev/ttyACM0', 9600,timeout=2) # Establish the connection on a specific port

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
# parser.add_option('--run-num', action="store", dest="run_num", default="1", help="run number for condition X")
(options, args) = parser.parse_args()

acquire_images = options.acquire_images
save_images = options.save_images
output_path = options.output_path
output_format = options.output_format
# run_num = options.run_num
save_in_separate_process = options.save_in_separate_process
fullscreen = options.fullscreen
whichMonitor = options.whichMonitor
if not fullscreen:
    winsize = [800, 600]
else:
    winsize = monitors.Monitor(whichMonitor).getSizePix()
use_pvapi = options.use_pvapi

print "WIN SIZE: ", winsize
print output_format

if not acquire_images:
    save_images = False

save_as_tif = False
save_as_png = False
save_as_npz = False
save_as_dict = False
if output_format == 'png':
    save_as_png = True
if output_format == 'tif':
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

win_flag = 0

user_input=raw_input("\nEnter subject ID:\n")
if user_input=='':
    subID='test'
subID=user_input

user_input=raw_input("\nEnter stimulation frequency (Hz):\n")
if user_input=='':
    cyc_per_sec = 0.05
cyc_per_sec = float(user_input)

while True:
    time.sleep(2)
    user_input=raw_input("\nEnter COND num [1=V-Left, 2=V-Right, 3=H-Down, 4=H-Up]  to continue or 'exit':\n")
    if user_input=='exit':
        break

    # conditionTypes = ['1']
    condnum = int(user_input)
    cond_label = ['Left','Right','Top','Bottom']
    condname = cond_label[int(condnum)-1]

    user_input=raw_input("\nEnter RUN num to continue or 'exit':\n")
    if user_input=='exit':
        break
    run_num = user_input    

    runID = condname + '_run' + str(run_num) # i.e., JR009W_Left0
    str_cyc = str(cyc_per_sec)
    exptID = str(subID) + '_bar_' + str_cyc.replace('.', '') + 'Hz'

    run_path = os.path.join(output_path, exptID)
    if not os.path.exists(run_path):
        os.mkdir(run_path)

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
        currpath = '%s/%s/%s/' % (output_path, exptID, runID) #(output_path, currdict['condname'], str(run_num))
        if not os.path.exists(currpath):
            os.mkdir(currpath)

        frame_log_file = open(os.path.join(output_path, exptID, 'framelog_%s_%s.txt') % (currdict['condname'], str(run_num)), 'w')
        frame_log_file.write('tstamp\t n\t framenum\t currcond\t runnum\t xpos\t ypos\t stimsize\t cycledur\t cyclelen\t degpercyc\t TF\t SF\n ')

        while currdict is not None:

            frame_log_file.write('%i\t %i\t %i\t %s\t %i\t %10.4f\t %10.4f\t %s\t %10.4f\t %10.4f\t %10.4f\t %10.4f\t %10.4f\n' % (int(currdict['time']), n, int(currdict['frame']), str(currdict['condname']), int(run_num), currdict['xpos'], currdict['ypos'], str(currdict['stim_size']), currdict['cycledur'], currdict['cyclelen'], currdict['degpercyc'], currdict['TF'], currdict['SF']))

            if save_as_png:

                fname = '%s/%s/%s/00%i_%i_%i_%i_%ideg_%s.png' % (output_path, exptID, runID, int(currdict['condnum']), int(currdict['time']), int(currdict['frame']), int(n), int(currdict['bar_width']), str(currdict['stim_pos']))

                tiff = TIFF.open(fname, mode='w')
                tiff.write_image(currdict['im'])
                tiff.close()

            elif save_as_tif:

                fname = '%s/%s/%s/00%i_%i_%i_%i_%ideg_%s.tif' % (output_path, exptID, runID, int(currdict['condnum']), int(currdict['time']), int(currdict['frame']), int(n), int(currdict['bar_width']), str(currdict['stim_pos']))

                tiff = TIFF.open(fname, mode='w')
                tiff.write_image(currdict['im'])
                tiff.close()

            elif save_as_npz:
                np.savez_compressed('%s/test%d.npz' % (output_path, n), currdict['im'])

            else:

                fname = '%s/%s/00%i_%i_%i_%i.pkl' % (output_path, currdict['condname'], int(currdict['condnum']), int(currdict['time']), int(currdict['frame']), int(n))
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
    flash=0
    if win_flag==0:
        if flash:
            win = visual.Window(fullscr=fullscreen, color=(.5,.5,.5), size=winsize, units='deg', monitor=whichMonitor)
        else:
            win = visual.Window(fullscr=fullscreen, color=(-1,-1,-1), size=winsize, units='deg', monitor=whichMonitor)
        win_flag=1

    # STIMULUS PARAMS:
    # conditionTypes = ['1']
    # condLabel = ['V-Left','V-Right','H-Down','H-Up']

    # cyc_per_sec = 0.13 # 

    flashPeriod = 1.0 #0.2#0.2 #amount of time it takes for a full cycle (on + off)
    dutyCycle = 1.0 #0.5#0.5 #Amount of time flash bar is "on" vs "off". 0.5 will be 50% of the time.

    bar_color = 1 # starting 1 for white, -1 for black, 0.5 for low contrast white, etc.
    bar_width = 1 # bar width in degrees 

    # SCREEN PARAMS:
    screen_width_cm = monitors.Monitor(whichMonitor).getWidth()
    screen_height_cm = (float(screen_width_cm)/monitors.Monitor(whichMonitor).getSizePix()[0])*monitors.Monitor(whichMonitor).getSizePix()[1]

    width_deg = tools.monitorunittools.cm2deg(screen_width_cm, monitors.Monitor(whichMonitor))
    height_deg = tools.monitorunittools.cm2deg(screen_height_cm, monitors.Monitor(whichMonitor))

    use_width = True
    if use_width:
        total_length = max([screen_width_cm, screen_height_cm])
    else:
        total_length = min([screen_width_cm, screen_height_cm])
    print "Base Length (screen dim, cm):  ", total_length

    total_length_deg = tools.monitorunittools.cm2deg(total_length, monitors.Monitor(whichMonitor))

    # TIMING PARAMS:
    num_cycles = 20 # how many times to do the cycle of 1 condition

    fps = 60.
    # total_time = total_length/(total_length*cyc_per_sec) #how long it takes for a bar to move from startPoint to endPoint
    # print "Cycle Travel TIME (s): ", total_time

    frames_per_cycle = fps/cyc_per_sec
    distance = monitors.Monitor(whichMonitor).getDistance()
    bar_width_cm = tools.monitorunittools.deg2cm(bar_width, monitors.Monitor(whichMonitor))
    print "Distance from monitor (cm): ", distance
    print "Bar width (deg | cm): ", bar_width, ' | ', bar_width_cm
    # duration = total_time*num_cycles; #how long to run the same condition for (seconds)
    # print "TOTAL DUR: ", duration


    t=0
    nframes = 0.
    frame_accumulator = 0
    flash_count = 0
    last_t = None

    report_period = 60 # frames
    frame_rate = 60.000
    refresh_rate = 60.000

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
    # cyc = 1
    # condnums = [condnum]
    # for cond in conditionTypes:
    # print condType
    # print condLabel[int(condType)-1]

    # if cyc == 1:
    # SPECIFICY CONDITION TYPES:
    if condnum == 1:
        orientation = 1 # 1 = VERTICAL, 0 = horizontal
        direction = 1 # 1 = start from LEFT or BOTTOM (neg-->pos), 0 = start RIGHT or TOP (pos-->neg)

    elif condnum == 2:
        orientation = 1 # vertical
        direction = 0 # start from RIGHT

    elif condnum == 3:
        orientation = 0 # horizontal
        direction = 0 # start from TOP

    elif condnum == 4:
        orientation = 0 # horizontal
        direction = 1 # start from BOTTOM


    if orientation==1:
        angle = 90 #0 is horizontal, 90 is vertical. 45 goes from up-left to down-right.
        longside = tools.monitorunittools.cm2deg(screen_height_cm, monitors.Monitor(whichMonitor)) #screen_height_cm

    else:
        angle = 0
        longside = tools.monitorunittools.cm2deg(screen_width_cm, monitors.Monitor(whichMonitor)) #screen_width_cm

    stim_size = (longside,bar_width) # First number is longer dimension no matter what the orientation is.

    unsign_start_point = (total_length_deg*0.5) + bar_width*0.5 # half the screen-size, plus hal bar-width to start with bar OFF screen

    #position parameters
    center_point = [0,0] #center of screen is [0,0] (degrees).
    if direction==1: # START FROM NEG, go POS (start left-->right, or start bottom-->top)
        start_sign = -1
    else:
        start_sign = 1

    # startPoint = startSign*uStartPoint; #bar starts this far from centerPoint (in degrees)
    # # currently, the endPoint is set s.t. the same total distance is traveled regardless of V or H bar
    # # endPoint = -1*(startPoint + startSign*(total_length_deg*0.5-uStartPoint+barWidth*0.5))
    # endPoint = -1*(startPoint + startSign*(total_length_deg*0.5-uStartPoint))
    # dist = endPoint - startPoint
    # #print dist
    # # 1. bar moves to this far from centerPoint (in degrees)
    # # 2. bar starts & ends OFF the screen

    start_point = start_sign * unsign_start_point

    # currently, the endPoint is set s.t. the same total distance is traveled regardless of V or H bar
    # endPoint = -1*(startPoint + startSign*(total_length_deg*0.5-uStartPoint+barWidth*0.5))
    # endPoint = -1*(startPoint + startSign*(total_length_deg*0.5-uStartPoint - barWidth*0.5))
    end_point = -1 * start_point
    start_to_end = end_point - start_point
    print "Cycle Travel LENGTH (deg): ", start_to_end
    print "START: ", start_point
    print "END: ", end_point
    print "Degrees per cycle: ", start_to_end # center-to-center #abs(start_point)*2. + bar_width
    SF = 1./(abs(start_point)*2. + bar_width)
    print "Calc SF (cpd): ", SF
    # cyc = 0
    cycle_duration = start_to_end / (start_to_end*cyc_per_sec)
    total_duration = cycle_duration * num_cycles
    print "Cycle Travel TIME (s): ", cycle_duration
    print "TOTAL DUR: ", total_duration

    # 1. bar moves to this far from centerPoint (in degrees)
    # 2. bar starts & ends OFF the screen

    # CREATE THE STIMULUS:
    bartex = np.ones([256,256,3])*bar_color;
    bartex[:,:,0] = 0. # turn OFF red channel
    barStim = visual.PatchStim(win=win,tex=bartex,mask='none',units='deg',pos=center_point,size=stim_size,ori=angle)
    barStim.setAutoDraw(False)

    # DISPLAY LOOP:
    win.flip() # first clear everything
    # time.sleep(0.001) # wait a sec

    FORMAT = '%Y%m%d%H%M%S%f'
    frame_counter = 0

    cyc = 0
    clock = core.Clock()
    while clock.getTime()<=total_duration: #frame_counter < frames_per_cycle*num_seq_reps: #endPoint - posLinear <= dist: #frame_counter <= frames_per_cycle*num_seq_reps: 
        t = globalClock.getTime()

        # if (clock.getTime()/flashPeriod) % (1.0) < dutyCycle:
        #     barStim.setContrast(1)
        #     #barStim.setColor(whiteBar, 'rgb255')
        # else:
        #     barStim.setContrast(0)
        #     #barStim.setColor(blackBar, 'rgb255')

        posLinear = (clock.getTime() % cycle_duration) / cycle_duration * (end_point-start_point) + start_point #what pos we are at in degrees
        print posLinear
        posX = posLinear*math.sin(angle*math.pi/180)+center_point[0]
        posY = posLinear*math.cos(angle*math.pi/180)+center_point[1]
        barStim.setPos([posX,posY])
        barStim.draw()
        win.flip()

        if acquire_images:
            # fdict = dict()
            #for fr_idx in range(int(frame_rate/refresh_rate)):
                # try:
                #     fdict['im'].append(camera.capture_wait())
                # except KeyError:
                #     fdict['im'] = [camera.capture.wait()]
            im_array = camera.capture_wait()
            camera.queue_frame()

        if save_images:
            fdict = dict()
            fdict['im'] = im_array
            fdict['bar_width'] = bar_width
            fdict['condnum'] = condnum
            fdict['condname'] = cond_label[int(condnum)-1]
            fdict['frame'] = frame_counter #nframes
            #print 'frame #....', frame_counter
            fdict['time'] = datetime.now().strftime(FORMAT)
            fdict['stim_pos'] = [posX,posY]

            fdict['xpos'] = posX
            fdict['ypos'] = posY
            fdict['degpercyc'] = abs(start_point)*2. + bar_width
            fdict['cycledur'] = cycle_duration # cycle_duration = start_to_end / (start_to_end*cyc_per_sec)
            fdict['cyclelen'] = start_to_end
            fdict['total_duration'] = total_duration # total_duration = cycle_duration * num_cycles
            fdict['SF'] = SF # SF = 1./(abs(start_point)*2. + bar_width)
            fdict['TF'] = cyc_per_sec
            fdict['stim_size'] = stim_size

            im_queue.put(fdict)

            frame_accumulator += 1


        if nframes % report_period == 0:
        #if frame_accumulator % report_period == 0:
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

        # cyc += 1
        # print cyc

    # make sure stimulus ends up OFF screen...
    # posLinear = (cycle_duration % cycle_duration) / cycle_duration * (end_point-start_point) + start_point #what pos we are at in degrees
    # posX = posLinear*math.sin(angle*math.pi/180)+center_point[0]
    # posY = posLinear*math.cos(angle*math.pi/180)+center_point[1]
    # barStim.setPos([posX,posY])
    # barStim.draw()
    # win.flip()

    #print "TOTAL COND TIME: " + str(clock.getTime())
    # Break out of the FOR loop if these keys are registered        
    if getout==1:
        break
    else:
        pass

    # cyc += 1
    # print cyc

    if acquire_images:
        camera.capture_end()
        # camera.close()

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
