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
outPath='/media/labuser/dixie_widefield/data/JR016W/20160815'

# outPath='/home/labuser/Desktop/test'

# os.system("python getSurface.py \
#   --save-images\
#    --output-path "+outPath)
#for run in range(runStart,runEnd+1):

# os.system("python moving_bar.py \
#     --save-images \
#     --monitor='AQUOS' \
#     --output-path="+outPath)

# os.system("python gabor_circle.py \
#     --save-images \
#     --monitor='AQUOS' \
#     --use-images\
#     --output-path="+outPath)

os.system("python gabor_circle.py \
    --monitor='AQUOS' \
    --save-images\
    --output-path="+outPath)



#   time.sleep(60)#wait a minute
print("Done")
#ser.write('2')#Turn off camera




