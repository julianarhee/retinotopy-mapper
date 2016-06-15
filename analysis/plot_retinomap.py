#!/usr/bin/env python2

'''

This script analyzes data acquired using movingBar_tmp.py.

It is the longer way of doing FFT (akin to fiedmap_demodulate_orig.py)

It creates maps based on reversal directions of vertical and horizontal bars.

Run:  python make+maps.py /path/to/imaging/directory

It will output change in response to...

'''

import numpy as np
import os
from skimage.measure import block_reduce
from scipy.misc import imread
import cPickle as pkl
import scipy.signal
import numpy.fft as fft
import sys
import optparse
from libtiff import TIFF
from PIL import Image
import re
import itertools
from scipy import ndimage

import math
import matplotlib as mpl
import matplotlib.pylab as plt
import matplotlib.cm as cm
import pandas as pd

from datetime import datetime

scale = 1

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'valid')


parser = optparse.OptionParser()
parser.add_option('--headless', action="store_true", dest="headless", default=False, help="run in headless mode, no figs")
parser.add_option('--reduce', action="store", dest="reduce_val", default="2", help="block_reduce value")
parser.add_option('--sigma', action="store", dest="gauss_kernel", default="0", help="size of Gaussian kernel for smoothing")
parser.add_option('--format', action="store", dest="im_format", default="tif", help="saved image format")
parser.add_option('--bar', action="store_true", dest="bar", default=True, help="moving bar stimulus or not")
parser.add_option('--custom', action="store_true", dest="custom_keys", default=False, help="custom keys for condition runs or not")
parser.add_option('--run', action="store", dest="run_num", default=0, help="run number for current condition set")

parser.add_option('--up', action="store", dest="up_key", default=1, help="if more than one run, run number")
parser.add_option('--down', action="store", dest="down_key", default=1, help="if more than one run, run number")
parser.add_option('--left', action="store", dest="left_key", default=1, help="if more than one run, run number")
parser.add_option('--right', action="store", dest="right_key", default=1, help="if more than one run, run number")
parser.add_option('--append', action="store",
                  dest="append_name", default="", help="append string to saved file name")

(options, args) = parser.parse_args()

bar = options.bar
im_format = '.'+options.im_format
headless = options.headless
reduce_factor = (int(options.reduce_val), int(options.reduce_val))
if reduce_factor[0] > 0:
    reduceit=1
else:
    reduceit=0
gsigma = int(options.gauss_kernel)
if headless:
    mpl.use('Agg')

custom_keys = options.custom_keys

up_key = str(options.up_key)
down_key = str(options.down_key)
left_key = str(options.left_key)
right_key = str(options.right_key)

run_num = str(options.run_num)

append_to_name = str(options.append_name)

#################################################################################
# GET PATH INFO:
#################################################################################
outdir = sys.argv[1]
# files = os.listdir(outdir)
# files = [f for f in files if os.path.splitext(f)[1] == '.pkl']

rundir = os.path.split(outdir)[0]
sessiondir = os.path.split(rundir)[0]


#################################################################################
# GET BLOOD VESSEL IMAGE:
#################################################################################

folders = os.listdir(sessiondir)
figpath = [f for f in folders if f == 'figures']

if figpath:
    figdir = figpath[0]
    # ims = os.listdir(os.path.join(sessiondir, figdir))
    tmp_ims = os.listdir(os.path.join(sessiondir, figdir))
    ims = [i for i in tmp_ims if 'surface' in i or 'green' in i or 'GREEN' in i]
    print ims
    impath = os.path.join(sessiondir, figdir, ims[0])
    print impath
    # image = Image.open(impath) #.convert('L')
    # imarray = np.asarray(image)
    print os.path.splitext(impath)[1]
    if os.path.splitext(impath)[1] == '.tif':
        tiff = TIFF.open(impath, mode='r')
        imarray = tiff.read_image().astype('float')
        tiff.close()
        plt.imshow(imarray)
    else:
        image = Image.open(impath) #.convert('L')
        imarray = np.asarray(image)

else: # NO BLOOD VESSEL IMAGE...
    imarray = np.zeros([200,300])

# ims = os.listdir(os.path.join(sessiondir, figdir))
# print ims
# impath = os.path.join(sessiondir, figdir, ims[0])
# # image = Image.open(impath) #.convert('L')
# # imarray = np.asarray(image)
# print os.path.splitext(impath)[1]
# if os.path.splitext(impath)[1] == '.tif':
#   tiff = TIFF.open(impath, mode='r')
#   imarray = tiff.read_image().astype('float')
#   tiff.close()
#   plt.imshow(imarray)
# else:
#   image = Image.open(impath) #.convert('L')
#   imarray = np.asarray(image)

# files = os.listdir(outdir)

# # GET BLOOD VESSEL IMAGE:
# ims = [f for f in files if os.path.splitext(f)[1] == str(im_format)]
# print ims
# impath = os.path.join(outdir, ims[0])
# image = Image.open(impath).convert('L')
# imarray = np.asarray(image)

# GET DATA STRUCT FILES:
# sessions = [f for f in flist if os.path.splitext(f)[1] != '.png']
# session_path = os.path.join(outdir, sessions[int(0)]) ## LOOP THIS


#################################################################################
# GET DATA STRUCT FILES:
#################################################################################

files = os.listdir(outdir)
files = [f for f in files if os.path.splitext(f)[1] == '.pkl']
dstructs = [f for f in files if 'D_target_FFT' in f and str(reduce_factor) in f]
if not dstructs:
    dstructs = [f for f in files if 'D_target' in f and str(reduce_factor) in f] # address older analysis formats

print dstructs

D = dict()
for f in dstructs:
    outfile = os.path.join(outdir, f)
    with open(outfile,'rb') as fp:
        D[f] = pkl.load(fp)
# close

split = 1
# MATCH ELEV vs. AZIM conditions:
outshape = D[D.keys()[0]]['ft_real'].shape
for curr_key in D.keys():
    print curr_key, outshape
    reals = D[curr_key]['ft_real'].ravel()
    imags = D[curr_key]['ft_imag'].ravel()
    ft = [complex(x[0], x[1]) for x in zip(reals, imags)]
    # if 'Down' in curr_key:
    #     imags = imags*-1
    # if 'Right' in curr_key:
    #     imags = imags*-1
    # ftmap[curr_key] = [complex(x[0], x[1]) for x in zip(reals, imags)]
    # ftmap[curr_key] = np.reshape(np.array(ftmap[curr_key]), outshape)
    if split:
        if "Left" in curr_key:
            leftmap = np.reshape(np.array(ft), outshape)
        elif "Right" in curr_key:
            rightmap = np.reshape(np.array(ft), outshape)
        elif "Down" in curr_key:
            downmap = np.reshape(np.array(ft), outshape)
        elif "Up" in curr_key:
            upmap = np.reshape(np.array(ft), outshape)
    else:

        if "Left" in curr_key:
            leftmap = D[curr_key]['ft']
        elif "Right" in curr_key:
            rightmap = D[curr_key]['ft']
        elif "Down" in curr_key:
            downmap = D[curr_key]['ft']
        elif "Up" in curr_key:
            upmap = D[curr_key]['ft']


if bar:
    print "BAR!!!", D.keys()
    if custom_keys:

        upkey = [k for k in D.keys() if 'H' in k and '_'+up_key in k and 'Up' in k][0]
        downkey = [k for k in D.keys() if 'H' in k and '_'+down_key in k and 'Down' in k][0]

        leftkey = [k for k in D.keys() if 'V' in k and '_'+left_key in k and 'Left' in k][0]
        rightkey = [k for k in D.keys() if 'V' in k and '_'+right_key in k and 'Right' in k][0]

        H_keys = [upkey, downkey]
        V_keys = [leftkey, rightkey]

    else:
        # upkey = [k for k in ftmap.keys() if 'H' in k and '_'+run_num in k and 'Up' in k][0]
        # downkey = [k for k in ftmap.keys() if 'H' in k and '_'+run_num in k and 'Down' in k][0]

        # leftkey = [k for k in ftmap.keys() if 'V' in k and '_'+run_num in k and 'Left' in k][0]
        # rightkey = [k for k in ftmap.keys() if 'V' in k and '_'+run_num in k and 'Right' in k][0]
        upkey = [k for k in D.keys() if 'H' in k and '_'+run_num in k and 'Up' in k][0]
        downkey = [k for k in D.keys() if 'H' in k and '_'+run_num in k and 'Down' in k][0]

        leftkey = [k for k in D.keys() if 'V' in k and '_'+run_num in k and 'Left' in k][0]
        rightkey = [k for k in D.keys() if 'V' in k and '_'+run_num in k and 'Right' in k][0]

        H_keys = [upkey, downkey]
        V_keys = [leftkey, rightkey]
        # print H_keys
        # print V_keys
        # V_keys = [k for k in ftmap.keys() if 'V' in k and '_'+run_num in k]
        # H_keys = [k for k in ftmap.keys() if 'H' in k and '_'+run_num in k]

    # azimuth_phase = (np.angle(ftmap[V_keys[0]]) + np.angle(ftmap[V_keys[1]])) / 2. #* (180./math.pi)
    # elevation_phase = (np.angle(ftmap[H_keys[0]]) + np.angle(ftmap[H_keys[1]])) / 2.  #* (180./math.pi)

    print "AZ keys: ", V_keys
    print "EL keys: ", H_keys

    # azimuth_phase = np.angle(ftmap[V_keys[0]] / ftmap[V_keys[1]]) / 2.  #* (180./math.pi)
    # elevation_phase = np.angle(ftmap[H_keys[0]] / ftmap[H_keys[1]]) / 2. #* (180./math.pi)

    # azimuth_phase = np.angle(ftmap[V_keys[0]]) - np.angle(ftmap[V_keys[1]]) #* (180./math.pi)
    # elevation_phase = np.angle(ftmap[H_keys[0]]) - np.angle(ftmap[H_keys[1]]) #* (180./math.pi)

    
    # azimuth_phase = 2 * (np.angle(ftmap[V_keys[0]]) - np.angle(ftmap[V_keys[1]])) #* (180./math.pi)
    # elevation_phase = 2 * (np.angle(ftmap[H_keys[0]]) - np.angle(ftmap[H_keys[1]])) #* (180./math.pi)



    #azimuth_phase = np.angle(ftmap[V_keys[0]] * ftmap[V_keys[1]]) / 2.
    #elevation_phase = np.angle(ftmap[H_keys[0]] * ftmap[H_keys[1]]) / 2.

    # azimuth_phase = np.angle(ftmap[V_keys[0]]) + np.angle(ftmap[V_keys[1]]) 
    # elevation_phase = np.angle(ftmap[H_keys[0]]) + np.angle(ftmap[H_keys[1]])

    # azimuth_phase = ( np.angle(ftmap[V_keys[0]]) + np.angle(ftmap[V_keys[1]]) ) / 2.
    # elevation_phase = ( np.angle(ftmap[H_keys[0]]) + np.angle(ftmap[H_keys[1]]) ) / 2.

    delay_vert = np.angle(leftmap * rightmap) / 2.
    delay_horiz = np.angle(downmap * upmap) / 2.

    # azimuth_phase = ( np.angle(leftmap) - np.angle(rightmap) ) / 2.
    # elevation_phase = ( np.angle(downmap) - np.angle(upmap) ) / 2.

    azimuth_phase = np.angle(leftmap) - delay_vert
    elevation_phase = np.angle(downmap) - delay_horiz



    aztitle = 'azimuth'
    eltitle = 'elevation'

else:

    blank_key = [k for k in ftmap.keys() if 'blank' in k]
    stim_key = [k for k in ftmap.keys() if 'stimulus' in k]
    
    azimuth_phase = np.angle(ftmap[stim_key[0]])
    elevation_phase = np.angle(ftmap[blank_key[0]])

    aztitle = 'stimulus'
    eltitle = 'blank'



# freqs = D[V_keys[0]]['freqs']
# target_freq = D[V_keys[0]]['target_freq']
# target_bin = D[V_keys[0]]['target_bin']


#################################################################################
# PLOT IT:
#################################################################################
plt.figure()

scale = 1
maptype = 'spectral' #hsv' #'spectral'
valmin = -1*math.pi
valmax = math.pi#1*math.pi

plt.subplot(1,3,1) # GREEN LED image
plt.imshow(imarray,cmap=cm.Greys_r)

# azimuth_phase = azimuth_phase + math.pi
# elevation_phase = elevation_phase + math.pi


plt.subplot(1,3,2) # ABS PHASE -- elevation
if scale:
    fig = plt.imshow(elevation_phase, cmap=maptype, vmin=valmin, vmax=valmax)
else:
    fig = plt.imshow(elevation_phase, cmap=maptype)
# plt.colorbar()
cbar = plt.colorbar(fig) 
#cbar.ax.invert_yaxis() 
plt.title(eltitle)

plt.subplot(1,3,3) # ABS PHASE -- azimuth
if scale:
    fig = plt.imshow(azimuth_phase, cmap=maptype, vmin=valmin, vmax=valmax)
else:
    fig = plt.imshow(azimuth_phase, cmap=maptype)
cbar = plt.colorbar(fig) 
plt.title(aztitle)

# SAVE FIG 1
sessionpath = os.path.split(outdir)[0]
plt.suptitle(sessionpath)

FORMAT = '%Y%m%d%H%M%S%f'
currT = datetime.now().strftime(FORMAT)

outdirs = os.path.join(sessionpath, 'figures')
which_sesh = os.path.split(sessionpath)[1]
print outdirs
if not os.path.exists(outdirs):
    os.makedirs(outdirs)
# imname = which_sesh  + '_mainmaps_' + str(reduce_factor) + '.svg'
# plt.savefig(outdirs + '/' + imname, format='svg', dpi=1200)
if custom_keys:
    # imname = ' '.join([which_sesh, '_mainmaps_', str(reduce_factor), ])
    imname = "%s_mainmaps_%s_U%s_D%s_L%s_R%s_%s.jpg" % (which_sesh, str(reduce_factor), up_key, down_key, left_key, right_key, currT)

    # which_sesh  + '_mainmaps_' + str(reduce_factor) + '_U' + up_key + '_D' + down_key + '_L' + left_key + '_R' + right_key + '_' + currT + '.jpg' #'.png'
else:
    imname = "%s_mainmaps_%s_run%s_%s.jpg" % (which_sesh, str(reduce_factor), str(run_num), currT)
    # imname = which_sesh  + '_mainmaps_' + str(reduce_factor) + '_' + currT + '.png'

# plt.savefig(outdirs + '/' + imname, format='png')
# plt.savefig(outdirs + '/' + imname)
print outdirs + '/' + imname


print "AZ conditions: ", D[V_keys[0]]['target_freq'], D[V_keys[0]]['target_bin']
print "EL conditions: ", D[H_keys[0]]['target_freq'], D[H_keys[0]]['target_bin']

plt.show()


# GET ALL RELATIVE CONDITIONS:
plt.figure()

if bar: 

    plt.subplot(3,4,1) # GREEN LED image
    plt.imshow(imarray,cmap=cm.Greys_r)

    scale = 1
    plt.subplot(3,4,2) # ABS PHASE -- elevation
    if scale:
        fig = plt.imshow(elevation_phase, cmap=maptype, vmin=valmin, vmax=valmax)
    else:
        fig = plt.imshow(elevation_phase, cmap=maptype)
    cbar = plt.colorbar(fig) 
    # cbar.ax.invert_yaxis() 
    plt.title(eltitle)


    plt.subplot(3, 4, 3) # ABS PHASE -- azimuth
    scale = 1
    if scale:
        fig = plt.imshow(azimuth_phase, cmap=maptype, vmin=valmin, vmax=valmax)
    else:
        fig = plt.imshow(azimuth_phase, cmap=maptype)
    # plt.colorbar()
    cbar = plt.colorbar(fig) 
    plt.title(aztitle)

    # plt.show()

    # PHASE:
    scale = 1
    for i,k in enumerate(H_keys): #enumerate(ftmap.keys()):
        plt.subplot(3,4,i+5)
        if 'Down' in k:
            phase_map = np.angle(downmap)
        else:
            phase_map = np.angle(upmap)

        # phase_map = np.angle(D[k]['ft']) #np.angle(complex(D[k]['ft_real'], D[k]['ft_imag']))
        #plt.figure()
        # if 'Down' in k:
        #     phase_map = -1*phase_map
        if scale:
            fig = plt.imshow(phase_map, cmap=maptype, vmin=valmin, vmax=valmax)
        else:
            fig = plt.imshow(phase_map, cmap=maptype)
        plt.title(k)
        
        # if 'Up' in k:
        if 'Down' in k:
            cbar = plt.colorbar(fig) 
            # cbar.ax.invert_yaxis()
        else:
            plt.colorbar() 

    scale = 1
    for i,k in enumerate(V_keys): #enumerate(ftmap.keys()):
        plt.subplot(3,4,i+7)
        if 'Right' in k:
            phase_map = np.angle(rightmap)
        else:
            phase_map = np.angle(leftmap)

        # phase_map = np.angle(D[k]['ft']) #np.angle(complex(D[k]['ft_real'], D[k]['ft_imag']))
        #plt.figure()
        # if 'Right' in k:
        #     phase_map = -1*phase_map
        if scale:
            fig = plt.imshow(phase_map, cmap=maptype, vmin=valmin, vmax=valmax)
        else:
            fig = plt.imshow(phase_map, cmap=maptype)
        plt.title(k)

        if 'Right' in k:
            cbar = plt.colorbar(fig) 
            # cbar.ax.invert_yaxis()
        else:
            plt.colorbar() 

    # MAG:
    for i,k in enumerate(H_keys): #enumerate(D.keys()):
        plt.subplot(3,4,i+9)
        mag_map = D[k]['mag_map']
        #mag_norm = (mag_map - mag_map.min()) / (mag_map.max() - mag_map.min())

        fig = plt.imshow(mag_map, cmap=cm.Greys_r)
        plt.title(k)
        plt.colorbar()

    for i,k in enumerate(V_keys): #enumerate(D.keys()):
        plt.subplot(3,4,i+11)
        mag_map = D[k]['mag_map']
        #mag_norm = (mag_map - mag_map.min()) / (mag_map.max() - mag_map.min())

        fig = plt.imshow(mag_map, cmap=cm.Greys_r)
        plt.title(k)
        plt.colorbar()


    #plt.suptitle(session_path)


    # sessionpath = os.path.split(outdir)[0]
    # plt.suptitle(sessionpath)

    # SAVE FIG
    # outdirs = os.path.join(sessionpath, 'figures')
    # which_sesh = os.path.split(sessionpath)[1]
    # print outdirs
    # if not os.path.exists(outdirs):
    #   os.makedirs(outdirs)

    if custom_keys:
        imname = "%s_allmaps_%s_U%s_D%s_L%s_R%s_%s_%s" % (which_sesh, str(reduce_factor), up_key, down_key, left_key, right_key, append_to_name, currT)

        # imname = which_sesh  + '_allmaps_run' + str(run_num) + '_' + str(reduce_factor) + '_U' + up_key + '_D' + down_key + '_L' + left_key + '_R' + right_key + '_' + currT + '.svg'
    else:
        # imname = which_sesh  + '_allmaps_run' + str(run_num) + '_' + str(reduce_factor) + '_' + currT + '.svg'
        imname = "%s_allmaps_%s_run%s_%s_%s" % (which_sesh, str(reduce_factor), str(run_num), append_to_name, currT)

    # plt.savefig(outdirs + '/' + imname + '.svg', format='svg', dpi=1200)
#   print outdirs + '/' + imname
#   plt.show()
    # plt.savefig(outdirs + '/' + imname + '.png', format='png')


    plt.savefig(os.path.join(outdirs, imname+'.svg'), format='svg', dpi=1200)
    plt.savefig(os.path.join(outdirs, imname+'.png'), format='png')



    print outdirs + '/' + imname
    plt.show()




    # # MASK
    # threshold = 0.5
    # plt.figure()
    # for i,k in enumerate(H_keys): #enumerate(D.keys()):
    #   plt.subplot(2,2,i+1)
    #   phase_map = np.angle(ftmap[k])
    #   mag_map = D[k]['mag_map']
    #   #mag_norm = (mag_map - mag_map.min()) / (mag_map.max() - mag_map.min())

    #   [x, y] = np.where(mag_map >= threshold*mag_map.max())
    #   phase_mask = np.ones(mag_map.shape)* 100
    #   phase_mask[x, y] = phase_map[x, y]

    #   [nullx, nully] = np.where(phase_mask == 100)
    #   print nullx, nully
    #   phase_mask[nullx, nully] = np.nan
    #   phase_mask = np.ma.array(phase_mask)

    #   fig = plt.imshow(phase_mask, cmap=maptype, vmin=-1*math.pi, vmax=math.pi)
    #   plt.title(k)
    #   plt.colorbar()

    # for i,k in enumerate(V_keys): #enumerate(D.keys()):
    #   plt.subplot(2,2,i+3)
    #   phase_map = np.angle(ftmap[k])
    #   mag_map = D[k]['mag_map']
    #   #mag_norm = (mag_map - mag_map.min()) / (mag_map.max() - mag_map.min())

    #   [x, y] = np.where(mag_map >= threshold*mag_map.max())
    #   phase_mask = np.ones(mag_map.shape)* 100
    #   phase_mask[x, y] = phase_map[x, y]

    #   [nullx, nully] = np.where(phase_mask == 100)
    #   print nullx, nully
    #   phase_mask[nullx, nully] = np.nan
    #   phase_mask = np.ma.array(phase_mask)

    #   fig = plt.imshow(phase_mask, cmap=maptype, vmin=-1*math.pi, vmax=math.pi)
    #   plt.title(k)
    #   plt.colorbar()

    plt.show()
