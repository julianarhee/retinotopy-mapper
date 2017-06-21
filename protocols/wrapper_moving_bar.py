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
outPath = '/media/labuser/dixie/volume1/widefield/data/JR042W/20170621'

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

os.system("python moving_bar_flash.py \
    --save-images \
    --monitor='AQUOS' \
    --flash\
    --output-path="+outPath)


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