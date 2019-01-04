#!/usr/bin/env python2
#blocked design
from __future__ import division

from psychopy import visual, event, core, monitors, logging
from pvapi import PvAPI, Camera
import time
import scipy.misc
import numpy as np
import multiprocessing as mp
import threading
from Queue import Queue
import sys

import errno
import os
import optparse
import math
import StringIO
from libtiff import TIFF
from PIL import Image, ImageSequence

import re
import glob

import cPickle as pkl
from time import sleep
from serial import Serial

from datetime import datetime

logging.console.setLevel(logging.CRITICAL)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

monitor_list = monitors.getAllMonitors()
print(monitor_list)
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
parser.add_option('--monitor', action="store", dest="whichMonitor", default='testMonitor', help=str(monitor_list))
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
print(whichMonitor)
if not fullscreen:
    winsize = [1280, 720]
else:
    winsize = monitors.Monitor(whichMonitor).getSizePix()
use_pvapi = options.use_pvapi

if not acquire_images:
    save_images = False

save_as_tiff = False
save_as_npz = False
if output_format == 'tiff':
    save_as_tiff = True
elif output_format == 'npz':
    save_as_npz = True
 

szX=1360
szY=768
nx, ny = (szX, szY)

x = np.linspace(1, nx, nx)
y = np.linspace(1, ny, ny)
xGrid, yGrid = np.meshgrid(x, y)




# --EXPERIMENTAL VARIABLES  --
barSize=125
effectiveSzX=szX+barSize
effectiveSzY=szY+barSize
rotationRate = 0.13# revs per sec
rotationPeriod = 1/rotationRate
flashRate = 0.13
flashPeriod = 1/flashRate  
numCycles = 12


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


# --GET FILENAMES OF ALL THE STIMULI--

imList=[]
sourceFolder='./Stimuli/1036x768/'
imList.append(sorted((fn for fn in os.listdir(sourceFolder) if fn.endswith('.tif')), key=natural_keys))

#---LOAD ALL IMAGES---
stimArray=[]
for im in imList[0]:
	imName = sourceFolder+im
	imframe = scipy.misc.imread(imName)
	imframe = (imframe/127)-1
	szY,szX = np.shape(imframe)
	stimArray.append(imframe)
print("ALL IMAGES LOADED!")

winFlag=0




userInput=raw_input("\nEnter run ID  to continue:\n")

runID=userInput

condNum=raw_input("\nEnter condition [1=V-Left, 2=V-Right, 3=H-Down, 4=H-Up] to continue or 'exit':\n")

condNum=int(condNum)
if winFlag==0:
	win = visual.Window(fullscr=fullscreen, size=winsize, units='pix', monitor=whichMonitor, color = (-.5,-.5,-.5))
	winFlag=1
	time.sleep(3)

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

subjectPath='%s/%s_%s_%s_%s/'%(subjectDir,animalid,session,runID,tStamp)


try:
    os.mkdir(subjectPath)
except OSError, e:
    if e.errno != errno.EEXIST:
        raise e
    pass

dataOutputPath=subjectPath+'frames/'
planOutputPath=subjectPath+'plan/'



try:
    os.mkdir(dataOutputPath)
except OSError, e:
    if e.errno != errno.EEXIST:
        raise e
    pass


try:
    os.mkdir(planOutputPath)
except OSError, e:
    if e.errno != errno.EEXIST:
        raise e
    pass


# -------------------------------------------------------------
# WRITE GENERAL EXPERIMENT PARAMETERS TO DISK
# -------------------------------------------------------------
paramOutputFile = open(planOutputPath+'parameters.txt','w')
paramOutputFile.write('Screen Size (pixels) X: %i \n'%szX)
paramOutputFile.write('Screen Size (pixels) Y: %i \n'%szY)
paramOutputFile.write('Bar Width (pixels): %i \n'%barSize)
paramOutputFile.write('Cycle Rate (Hz): %10.4f \n'%rotationRate)
paramOutputFile.write('Cycle Period (s): %10.4f \n'%rotationPeriod)
paramOutputFile.write('Flash Rate (Hz): %10.4f \n'%flashRate)
paramOutputFile.write('Flash Period (s): %10.4f \n'%flashPeriod)
paramOutputFile.write('Number of Cycles: %i \n'%numCycles)
paramOutputFile.close()

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
	frameTimeOutputFile = open(planOutputPath+'frameTimes.txt','w')
	frameTimeOutputFile.write('frameCount\t n\t frameCond\t stimPosition\t experimentTime\t experimentInterval\t frameT\t interval\n')
	currdict = im_queue.get()
	while currdict is not None:
		frameTimeOutputFile.write('%i\t %i\t %i\t %s\t %s\t %s\t %s\t %s\n' % \
			(int(currdict['frame']),n,currdict['frameCond'],currdict['stimPos'],\
				currdict['experimentT'],currdict['experimentInterval'],\
				currdict['time'],currdict['interval']))
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
# Stimulus Presentation (by C.E.)
# -------------------------------------------------------------

# --EXPERIMENTAL VARIABLES  --
expDuration = rotationPeriod * numCycles

stimPermList=np.random.randint(0,len(stimArray),int(expDuration/flashPeriod)*2)
#--STIMULI PRESENTATION--

from psychopy.tools import imagetools

globalClock = core.Clock()


#To Break Out of Loop
class BreakIt(Exception): pass

#To keep track of frame acquisition
nframes = 0
oldFrameT = 0
last_t = None
report_period = 6
imOld=None

reportPlaybackPeriod = 24
refreshPeriod=np.true_divide(1,24)

frameRate=camera.attr_float32_get('FrameRate')
framePeriod=np.around(np.true_divide(1,frameRate),3)#secs

#Presentation Routine
try:

	if acquire_images:
		# Start acquiring
		win.flip()
		time.sleep(0.002)
		camera.capture_start()
		camera.queue_frame()
		camera.capture_wait()
		camera.queue_frame()

	tStart = globalClock.getTime()
	while globalClock.getTime()-tStart < expDuration:
		win.flip()
		flipT = globalClock.getTime()
		drawT=flipT+refreshPeriod-tStart

		imCurrent=int(math.floor((globalClock.getTime()-tStart)/flashPeriod))

		if imCurrent != imOld:
			im0=np.copy(stimArray[stimPermList[imCurrent]])
			if condNum==1 or condNum==2:
				#startBar=(szX/2.0)-(barSize/2.0)
				startBar=np.random.randint(0,szX-barSize)
				endBar=startBar+barSize
				barImg=im0[:,startBar:endBar]
			elif condNum==3 or condNum==4:
				#startBar=(szY/2.0)-(barSize/2.0)
				startBar=np.random.randint(0,szY-barSize)
				endBar=startBar+barSize
				barImg=im0[startBar:endBar,:]

		imOld=imCurrent

		#update position
		if condNum==1:
			endBar=np.ceil(((((drawT)/rotationPeriod)*effectiveSzX))%(effectiveSzX))
			startBar=endBar-barSize
			stimPos=startBar
		elif condNum==2:
			endBar=np.ceil((((1-(drawT)/rotationPeriod)*effectiveSzX))%(effectiveSzX))
			startBar=endBar-barSize
			stimPos=startBar
		elif condNum==3:
			endBar=np.ceil(((((drawT)/rotationPeriod)*effectiveSzY))%(effectiveSzY))
			startBar=endBar-barSize
			stimPos=startBar
		elif condNum==4:
			endBar=np.ceil((((1-(drawT)/rotationPeriod)*effectiveSzY))%(effectiveSzY))
			startBar=endBar-barSize
			stimPos=startBar
			
		
		#draw image within aperture
		displayImg=np.ones(np.shape(xGrid))*-.5
		if condNum==1 or condNum==2:	
			if startBar<1:
				startBarImg=int(abs(startBar))
				endBarImg=int(barSize)
				colInd=np.squeeze(np.where(np.logical_and(xGrid[0,:]>=startBar,xGrid[0,:]<=endBar)))
			elif startBar>(szX-barSize):
				startBarImg=0
				endBarImg=int((szX-startBar))
				colInd=np.squeeze(np.where(np.logical_and(xGrid[0,:]>startBar,xGrid[0,:]<=endBar)))
			else:
				startBarImg=0
				endBarImg=int(barSize)
				colInd=np.squeeze(np.where(np.logical_and(xGrid[0,:]>=startBar,xGrid[0,:]<endBar)))
		#	print(colInd.size,barImg.shape,startBarImg,endBarImg,startBar,endBar)
			if colInd.size>0:
				displayImg[:,colInd]=np.squeeze(barImg[:,startBarImg:endBarImg])
		elif condNum==3 or condNum==4:
			if startBar<1:
				startBarImg=int(abs(startBar))
				endBarImg=int(barSize)
				colInd=np.squeeze(np.where(np.logical_and(yGrid[:,0]>=startBar,yGrid[:,0]<=endBar)))
			elif startBar>(szY-barSize):
				startBarImg=0
				endBarImg=int(szY-startBar)
				colInd=np.squeeze(np.where(np.logical_and(yGrid[:,0]>startBar,yGrid[:,0]<=endBar)))
			else:
				startBarImg=0
				endBarImg=int(barSize)
				colInd=np.squeeze(np.where(np.logical_and(yGrid[:,0]>=startBar,yGrid[:,0]<endBar)))
		#	print(colInd.size,barImg.shape,startBarImg,endBarImg,startBar,endBar)
			if colInd.size>0:
				displayImg[colInd,:]=np.squeeze(barImg[startBarImg:endBarImg,:])

		stim = visual.ImageStim(win, image=displayImg,size=(szX,-szY),units='pix')
		stim.draw()
		while globalClock.getTime()-flipT < refreshPeriod:
			if acquire_images:
				im_array = camera.capture_wait()
				camera.queue_frame()
				frameT=globalClock.getTime()-tStart
				frameInt=frameT-oldFrameT
				oldFrameT=frameT
				if nframes % report_period == 0:#REPORT ACQUISITION FRAME RATE
					if last_t is not None:
						print('avg frame rate: %f' % (report_period / (frameT - last_t)))
					last_t = frameT
			if save_images:
				fdict = dict()
				fdict['im'] = im_array
				fdict['frame'] = frame_counter
				fdict['frameCond']=condNum
				fdict['stimPos']=stimPos
				fdict['experimentT']=frameT
				fdict['experimentInterval']=frameInt
				fdict['time'] = ((frame_counter+1)*framePeriod)#assume camera has constant rate
				fdict['interval'] = framePeriod

				im_queue.put(fdict)
				nframes += 1
				frame_counter += 1
			# Break out of the while loop if these keys are registered
			if event.getKeys(keyList=['escape', 'q']):
				win.flip()
				win.flip()
				raise BreakIt						
	win.flip()
	win.flip()
	expDur=globalClock.getTime()-tStart
	print(expDur)
	expectedFrameCount=int(np.floor(np.around(expDur,2)/framePeriod))
	print('Actual Frame Count = '+str(frame_counter))
	print('Expected Frame Count = '+str(expectedFrameCount))

except BreakIt:
    pass    

if acquire_images:
	camera.clear_queue()
	camera.capture_end()
print "FINISHED"
im_queue.put(None)

# -------------------------------------------------------------
# WRITE PERFORMANCE REPORT TO FILE
# -------------------------------------------------------------
performOutputFile = open(planOutputPath+'performance.txt','w')
performOutputFile.write('frameRate\t framePeriod\t expDuration\t frameCount\t expectedFrameCount\t missingFrames\n')
performOutputFile.write('%10.4f\t %10.4f\t %10.4f\t %i\t %i\t %i\n'%\
	(frameRate,framePeriod,expDur,frame_counter,expectedFrameCount,expectedFrameCount-frame_counter))
performOutputFile.close()



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

if acquire_images:
    camera.close()
win.close()
core.quit()