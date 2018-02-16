#!/usr/bin/env python2
# wrapper script for calling stim presentation
import os
import time
# from serial import Serial
#ser = Serial('/dev/ttyACM0', 9600) # Establish the connection on a specific port

# ANIMAL INFO:
# rootdir = '/media/labuser/dixie/volume1/widefield/data'
rootdir = '/home/labuser/Documents/macro-data'
rootdir='/tmp/macro-data'
animalid = 'test'
session = '20180211'
acquisition = 'macro_fullfov'
save_images = True

# MONITOR INFO:
monitor = 'LGHDTV_macro1'
fullscreen = True
# ------------------
left = -10
right = 50
top = -10
bottom = -20
# ------------------

# STIMULUS INFO:
flash = True
bar_width = 2
target_freq = 0.13
ncycles = 10
acquisition_rate = 30.
nreps_per = 5

conds='right,bottom'


if fullscreen is True:
	os.system("python moving_bar.py \
		--root=%s\
		--animalid=%s\
		--session=%s\
		--acq=%s\
		--monitor=%s\
		--width=%i\
		--freq=%f\
		--ncycles=%i\
		--fps=%f\
		--nreps=%i\
		--conds=%s\
	    --save-images\
	    --flash" % (rootdir, animalid, session, acquisition, 
	    			monitor, bar_width, target_freq, ncycles, 
	    			acquisition_rate, nreps_per, conds)
	    			)
else:
	os.system("python moving_bar.py \
		--root=%s\
		--animalid=%s\
		--session=%s\
		--acq=%s\
		--monitor=%s\
		--width=%i\
		--freq=%f\
		--ncycles=%i\
		--fps=%f\
		--nreps=%i\
		--conds=%s\
		--left=%f\
		--right=%f\
		--bottom=%f\
		--top=%f\
	    --save-images\
	    --flash" % (rootdir, animalid, session, acquisition, 
	    			monitor, bar_width, target_freq, ncycles, 
	    			acquisition_rate, nreps_per, conds,
	    			left, right, bottom, top)
	    			)
print("Done")
#ser.write('2')#Turn off camera