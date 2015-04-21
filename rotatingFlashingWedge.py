#!/usr/bin/env python2
#rotate flashing wedge
from psychopy import visual, event, core
from pvapi import PvAPI, Camera
import time
from scipy.misc import imsave
import numpy as np
import multiprocessing as mp
import threading
from Queue import Queue
import sys


acquire_images = True
save_images = True
save_as_png = True
save_in_separate_process = False
fullscreen = False

pvapi_retries = 50


# -------------------------------------------------------------
# Camera Setup
# -------------------------------------------------------------

print('Finding camera...')

# try PvAPI
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

    print("Unable to find PvAPI camera")
    print(e)

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
                imsave('/tmp/frames/test%d.png' % n, im_array)
            else:
                np.savez_compressed('/tmp/frames/test%d.npz' % n, im_array)
            n += 1
    print('Disk-saving thread inactive...')


if save_in_separate_process:
    disk_writer = mp.Process(target=save_images_to_disk)
else:
    disk_writer = threading.Thread(target=save_images_to_disk)

disk_writer.daemon = True

if save_images:
    disk_writer.start()


# -------------------------------------------------------------
# Psychopy stuff here (just lifted from a demo)
# -------------------------------------------------------------

globalClock = core.Clock()
win = visual.Window(fullscr=fullscreen, size=[1680, 1050], units='deg', monitor='testMonitor')
sq_size = [15., 15.]
#make two wedges (in opposite contrast) and alternate them for flashing
wedge1 = visual.RadialStim(win, tex='sqrXsqr', color=1, size=sq_size,
    visibleWedge=[0, 45], radialCycles=4, angularCycles=8, interpolate=False,
    autoLog=False) #this stim changes too much for autologging to be useful
wedge2 = visual.RadialStim(win, tex='sqrXsqr', color=-1, size=sq_size,
    visibleWedge=[0, 45], radialCycles=4, angularCycles=8, interpolate=False,
    autoLog=False) #this stim changes too much for autologging to be useful


t=0

rotationRate = 0.1 #revs per sec
flashPeriod = 0.1 #seconds for one B-W cycle (ie 1/Hz)
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

while True:
    t = globalClock.getTime()

    if (nframes/4) % 2 == 0:
    # if (t%flashPeriod) < (flashPeriod/2.0):# (NB more accurate to use number of frames)
        stim = wedge1
    else:
        stim = wedge2
    
    stim.ori = t * rotationRate * 360.0  # set new rotation
    stim.draw()
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

camera.capture_end()
camera.close()

hang_time = time.time()
nag_time = 2.0

sys.stdout.write('Waiting for disk writer to catch up (this may take a while)...')
sys.stdout.flush()
waits = 0
while (not im_queue.empty()):
    
    now = time.time()
    if (now - hang_time) > nag_time:
        sys.stdout.write('.')
        sys.stdout.flush()
        hang_time = now
        waits += 1

print("\n")

if not im_queue.empty():
    print("WARNING: not all images have been saved to disk!")

disk_writer_alive = False

if save_in_separate_process and disk_writer is not None:
    disk_writer.terminate()

disk_writer.join()
print('Disk writer terminated')
    

# if save_images:
#     exit(0)  # pull the plug, otherwise it hangs (hacky)


