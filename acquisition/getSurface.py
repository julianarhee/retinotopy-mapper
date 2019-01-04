#!/usr/bin/env python2
#TODO: INSERT HEADER 
#blocked design
from psychopy import visual, event, core, monitors, logging

from pvapi import PvAPI, Camera
import time
from scipy.misc import imsave
import scipy.misc
import numpy as np
import multiprocessing as mp
import threading

from Queue import Queue
import sys
import errno

import os
import optparse
import StringIO

from libtiff import TIFF


#from PIL import Image, ImageSequence
#import Image, ImageSequence
import re
import glob
#import h5py
import cPickle as pkl

from datetime import datetime
logging.console.setLevel(logging.CRITICAL)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

monitor_list = monitors.getAllMonitors()

parser = optparse.OptionParser()
parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD')
	
parser.add_option('--no-camera', action="store_false", dest="acquire_images", default=True, help="just run PsychoPy protocol")
parser.add_option('--save-images', action="store_true", dest="save_images", default=False, help="save camera frames to disk")
parser.add_option('--output-path', action="store", dest="output_path", default="/tmp/", help="out path directory [default: /tmp/frames]")
parser.add_option('--output-format', action="store", dest="output_format", type="choice", choices=['tiff', 'npz'], default='tiff', help="out file format, tiff or npz [default: tiff]")
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
session = options.session
animalid = options.animalid
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
if output_format == 'tiff':
    save_as_tiff = True
elif output_format == 'npz':
    save_as_npz = True
    


# Make the output path if it doesn't already exist
dateFormat = '%Y%m%d%H%M%S%f'
tStamp=datetime.now().strftime(dateFormat)

	

try:
    os.mkdir(output_path)
except OSError, e:
    if e.errno != errno.EEXIST:
        raise e
    pass


subjectDir=os.path.join(output_path,animalid)

try:
    os.mkdir(subjectDir)
except OSError, e:
    if e.errno != errno.EEXIST:
        raise e
    pass

subjectDir=os.path.join(subjectDir,session)

try:
    os.mkdir(subjectDir)
except OSError, e:
    if e.errno != errno.EEXIST:
        raise e
    pass

subjectPath='%s/%s_%s_surface_%s'%(subjectDir,animalid,session,tStamp)




try:
    os.mkdir(subjectPath)
except OSError, e:
    if e.errno != errno.EEXIST:
        raise e
    pass

dataOutputPath=subjectPath+'Surface/'



try:
    os.mkdir(dataOutputPath)
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
		if save_as_tiff:
			fname = '%s/frame%i.tiff' % (dataOutputPath,int(currdict['frame']))
			tiff = TIFF.open(fname, mode='w')
			tiff.write_image(currdict['im'])
			tiff.close()
	
		elif save_as_npz:
			np.savez_compressed('%s/test%d.npz' % (output_path, n), currdict['im'])
		else:
			fname = '%s/frame%i.tiff' % (dataOutputPath,int(currdict['frame']),)
			with open(fname, 'wb') as f:
				pkl.dump(currdict, f, protocol=pkl.HIGHEST_PROTOCOL)
			
#		print 'DONE SAVING FRAME: ', currdict['frame'], n #fdict
		n += 1
		currdict = im_queue.get()
		
	disk_writer_alive = False
	#frameTimeOutputFile.close()
	print('Disk-saving thread inactive...')


if save_in_separate_process:
    disk_writer = mp.Process(target=save_images_to_disk)
else:
    disk_writer = threading.Thread(target=save_images_to_disk)

if save_images:
	frame_counter = 0
	disk_writer.daemon = True
	disk_writer.start()

# -------------------------------------------------------------
# FRAME ACQUISITION
# -------------------------------------------------------------

nFrames=10

globalClock = core.Clock()



if acquire_images:
	# Start acquiring
	time.sleep(0.002)
	camera.capture_start()
	#ser.write('1')#TRIGGER
	camera.queue_frame()
for f in range(0,nFrames):
	print(f)
	if acquire_images:
		im_array = camera.capture_wait()
		camera.queue_frame()
	if save_images:
		fdict = dict()
		fdict['im'] = im_array
		fdict['frame'] = f
		im_queue.put(fdict)
			
if acquire_images:
    camera.capture_end()
    camera.close()

#ser.write('2')
#flushBuffer()		
print("FINISHED")
im_queue.put(None)



if save_images:
	
	hang_time = time.time()
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
	
	
	disk_writer.join()
	print('Disk writer terminated')
    