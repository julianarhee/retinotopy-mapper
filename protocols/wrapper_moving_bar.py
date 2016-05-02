#!/usr/bin/env python2
#wrapper script for calling stim presentation
import os
import time
# from serial import Serial
#ser = Serial('/dev/ttyACM0', 9600) # Establish the connection on a specific port
​
# runStart=1;
# runEnd=2;
#outPath='./outputFiles/'
outPath='/media/juliana/IMDATA/widefield/TEST/20160502/anesthetized_bar_0013Hz'
​
# os.system("python getSurface.py \
# 	--save-images\
# 	 --output-path "+outPath)
#for run in range(runStart,runEnd+1):
​
os.system("python moving_bar.py \
	--save-images \
	--monitor='AQUOS' \
	--output-path="+outPath)
#	time.sleep(60)#wait a minute
print("Done")
#ser.write('2')#Turn off camera