#!/usr/bin/env python2
#wrapper script for calling stim presentation
import os
import time
# from serial import Serial
#ser = Serial('/dev/ttyACM0', 9600) # Establish the connection on a specific port

# runStart=1;
# runEnd=2;
#outPath='./outputFiles/'
# outPath='/media/labuser/IMDATA1/widefield/JR015W/20160803'
# outPath='/media/labuser/dixie/volume1/widefield/data/JR029W/20161201'
# outPath='/media/labuser/dixie/volume1/widefield/data/JR037Wp/20161211'
# outPath='/media/labuser/dixie/volume1/widefield/data/CE024/20161218'
# outPath='/media/labuser/dixie/volume1/widefield/data/JR030W/20161222'
# outPath='/media/labuser/dixie/volume1/widefield/data/JR041W/20170404'
# outPath='/media/labuser/dixie/volume1/widefield/data/JR039W/20170506'
# outPath = '/media/labuser/dixie/volume1/widefield/data/JR042W/20170621'
# outPath = '/media/labuser/dixie/volume1/widefield/data/JR044W/20170703'
# outPath = '/media/labuser/dixie/volume1/widefield/data/JR046W/20170711'
# outPath = '/media/labuser/dixie/volume1/widefield/data/JR050W/20170804'
#outPath = '/media/labuser/dixie/volume1/widefield/data/JR059W/20171003'


# ANIMAL INFO:
rootdir = '/media/labuser/dixie/volume1/widefield/data'
animalid = 'TEST'
session = '20170118'
acquisition = 'flash'
save_images = True

# MONITOR INFO:
monitor = 'AQUOS'
short_axis = False
# ------------------
fullscreen = False
left = -10
right = 50
top = -10
bottom = -20
# ------------------

# STIMULUS INFO:
flash = True
bar_width = 3
target_freq = 0.28
ncycles = 2
acquisition_rate = 30.
nreps_per = 2

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

# outPath='/media/labuser/IMDATA2/TEFO/retinomapping/JR027W/20161117'
# outPath='/home/labuser/Desktop/TefoTest/20161118'


#23751f0f95a53fcd8b7418c1c4d6f043a74ec806

# outPath='/home/labuser/Desktop/test'

# os.system("python getSurface.py \
#   --save-images\
#    --output-path "+outPath)
#for run in range(runStart,runEnd+1):

# os.system("python moving_bar.py \
#     --save-images \
#     --monitor='AQUOS' \
#     --output-path="+outPath)

# os.system("python moving_bar_flash.py \
#     --save-images \
#     --monitor='AQUOS' \
#     --flash\
#     --short-axis\
#     --output-path="+outPath)
#



# os.system("python moving_bar_flash.py \
#     --save-images \
#     --monitor='AQUOS' \
#     --flash\
#     --output-path="+outPath)


# os.system("python moving_bar_flash.py \
#     --save-images \
#     --monitor='AQUOS' \
#     --output-path="+outPath)

# os.system("python moving_bar_flash.py \
#     --save-images \
#     --short-axis \
#     --flash\
#     --monitor='AQUOS' \
#     --output-path="+outPath)

# -- far monitor -----------------

# os.system("python moving_bar_flash.py \
#     --save-images \
#     --monitor='AQUOS_far' \
#     --fps=30.0 \
#     --flash\
#     --output-path="+outPath)

# os.system("python moving_bar_flash.py \
#     --save-images \
#     --monitor='AQUOS_far' \
#     --fps=30.0 \
#     --output-path="+outPath)

# -----------------------------------


# os.system("python moving_bar_flash.py \
#     --no-camera \
#     --save-images \
#     --ncycles=1000 \
#     --monitor='DELL22' \
#     --output-path="+outPath)

# Just run on DELL for TeFo test:
# os.system("python moving_bar_flash.py \
#     --no-camera \
#     --ncycles=1000 \
#     --monitor='DELL22' \
#     --output-path="+outPath)


# os.system("python stimCircle.py \
#     --save-images \
#     --monitor='AQUOS' \
#     --output-path="+outPath)


#   time.sleep(60)#wait a minute
print("Done")
#ser.write('2')#Turn off camera