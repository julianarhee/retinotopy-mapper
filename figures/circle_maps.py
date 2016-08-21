#!/usr/bin/env python2
# coding: utf-8

# FROM circle_maps.ipnb (JNB)

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


def  cart2pol(x,y, units='deg'):
    """Convert from cartesian to polar coordinates

    :usage:

        theta, radius = pol2cart(x, y, units='deg')

    units refers to the units (rad or deg) for theta that should be returned
    """
    radius= np.hypot(x,y)
    theta= np.arctan2(y,x)
    if units in ['deg', 'degs']:
        theta=theta*180/np.pi
    return theta, radius


def pol2cart(theta, radius, units='deg'):
    """Convert from polar to cartesian coordinates

    usage::

        x,y = pol2cart(theta, radius, units='deg')

    """
    if units in ['deg', 'degs']:
        theta = theta*np.pi/180.0
    xx = radius*np.cos(theta)
    yy = radius*np.sin(theta)

    return xx,yy


parser = optparse.OptionParser()
parser.add_option('--headless', action="store_true", dest="headless", default=False, help="run in headless mode, no figs")
parser.add_option('--reduce', action="store", dest="reduce_val", default="1", help="block_reduce value")
parser.add_option('--path', action="store", dest="path", default="", help="path to data directory")
parser.add_option('-t', '--thresh', action="store", dest="threshold", default=0.5, help="cutoff threshold value")
parser.add_option('-r', '--run', action="store", dest="run", default=1, help="cutoff threshold value")
parser.add_option('--append', action="store", dest="append", default="", help="appended label for analysis structs")
parser.add_option('--mask', action="store", dest="mask", type="choice", choices=['DC', 'blank', 'magmax'], default='DC', help="mag map to use for thresholding: DC | blank | magmax [default: DC]")
parser.add_option('--cmap', action="store", dest="cmap", default='spectral', help="colormap for summary figures [default: spectral]")
parser.add_option('--use-norm', action="store_true", dest="use_norm", default=False, help="compare normalized blank to condition")

(options, args) = parser.parse_args()

use_norm = options.use_norm
headless = options.headless
reduce_factor = (int(options.reduce_val), int(options.reduce_val))
if reduce_factor[0] > 1:
    reduceit=1
else:
    reduceit=0
if headless:
    mpl.use('Agg')
    
colormap = options.cmap

threshold = float(options.threshold)
outdir = options.path
run_num = options.run

exptdir = os.path.split(outdir)[0]
sessiondir = os.path.split(exptdir)[0]

savedir = os.path.split(outdir)[0]
figdir = os.path.join(savedir, 'figures')
if not os.path.exists(figdir):
    os.makedirs(figdir)


#################################################################################
# GET BLOOD VESSEL IMAGE:
#################################################################################
folders = os.listdir(sessiondir)
# figpath = [f for f in folders if f == 'figures']
figpath = [f for f in folders if f == 'surface'][0]
print "EXPT: ", exptdir
print "SESSION: ", sessiondir
print "path to surface: ", figpath

if figpath:
    # figdir = figpath[0]
    tmp_ims = os.listdir(os.path.join(sessiondir, figpath))
    surface_words = ['surface', 'GREEN', 'green', 'Surface', 'Surf']
    ims = [i for i in tmp_ims if any([word in i for word in surface_words])]
    ims = [i for i in ims if '_' in i]
    print ims
    if ims:
        impath = os.path.join(sessiondir, figpath, ims[0])
        # image = Image.open(impath) #.convert('L')
        # imarray = np.asarray(image)
        print os.path.splitext(impath)[1]
        if os.path.splitext(impath)[1] == '.tif':
            tiff = TIFF.open(impath, mode='r')
            surface = tiff.read_image().astype('float')
            tiff.close()
            plt.imshow(surface)
        else:
            image = Image.open(impath) #.convert('L')
            surface = np.asarray(image)
    else:
        surface = np.zeros([200,300])
else: # NO BLOOD VESSEL IMAGE...
    surface = np.zeros([200,300])

if reduceit:
    surface = block_reduce(surface, reduce_factor, func=np.mean)

# plt.imshow(surface, cmap='gray')


#################################################################################
# GET DATA STRUCT FILES:
#################################################################################
# append = 'detrend'
append = options.append

files = os.listdir(outdir)
# print "all files: ", files
files = [f for f in files if os.path.splitext(f)[1] == '.pkl']
print "ALL files: ", files

# dstructs = [f for f in files if 'Target_fft' in f and str(reduce_factor) and key in f and append in f]

dstructs = [f for f in files if 'Target_fft' in f and str(reduce_factor) and append in f]
cw_key = [k for k in dstructs if 'CCW_' not in k and 'blank' not in k]
ccw_key = [k for k in dstructs if 'CCW_' in k]
blank_key = [k for k in dstructs if 'blank_' in k]
cond_keys = [cw_key, ccw_key, blank_key]


print "COND KEYS: "
print "CW: ", cw_key
print "CCW: ", ccw_key
print "N conds: ", len(cond_keys)

D = dict()
for f in dstructs:
	outfile = os.path.join(outdir, f)
	with open(outfile,'rb') as fp:
		D[f] = pkl.load(fp)

for key in range(len(cond_keys)):
    cond_set = cond_keys[key]
    print "RUNNING: ", cond_set


    for curr_key in cond_set:

        if 'blank' in curr_key:
            CW = False # This is just true bec of how protocol originally written (blank has "CCW" running underneath)
            is_blank = True

        else:
            is_blank = False

            if 'CCW_' in curr_key:
                CW = False
            else:
                CW = True

            # try:
            #     curr_key = curr_cond[0]
            # except IndexError, e:
            #     print e
            #     print "Structs found only for condition: ", cond_keys
            #     pass

        print curr_key

        threshold_type = options.mask #'blank'
        threshold = float(options.threshold)


        Ny = len(D[curr_key]['freqs'])/2.
        mag_map = D[curr_key]['mag_map']/Ny
        phase_map = D[curr_key]['phase_map']

        DC_mag_map = D[curr_key]['DC_mag']/Ny
        if threshold_type=='blank':
            blank_mag_map = D[blank_key[0]]['mag_map']/Ny

        date = os.path.split(os.path.split(os.path.split(outdir)[0])[0])[1]
        experiment = os.path.split(os.path.split(outdir)[0])[1]

        #-----------------------------------------------------------------
        # Overlaid Phase-map, thresholded
        #-----------------------------------------------------------------

        fig = plt.figure(figsize=(10,10))

        fig.add_subplot(2,3,1)
        plt.imshow(surface,cmap=cm.Greys_r)
        plt.axis('off')

        fig.add_subplot(2,3,2)
        plt.imshow(phase_map, cmap=colormap)
        plt.axis('off')
        plt.title('phase')

        fig.add_subplot(2,3,3)
        plt.imshow(surface,cmap=cm.Greys_r)
        print surface.shape

        old_min = mag_map.min()
        old_max = mag_map.max()
        new_min = 0
        new_max = 1
        normed_mag_map = np.zeros(mag_map.shape)
        for x in range(mag_map.shape[0]):
            for y in range(mag_map.shape[1]):
                old_val = mag_map[x, y]
                normed_mag_map[x, y] = (((old_val - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min

                
        if threshold_type=='DC':
            thresh_map = DC_mag_map
        elif threshold_type=='blank':
            thresh_map = blank_mag_map

        old_min = thresh_map.min()
        old_max = thresh_map.max()
        new_min = 0
        new_max = 1
        normed_thresh_map = np.zeros(thresh_map.shape)
        for x in range(mag_map.shape[0]):
            for y in range(thresh_map.shape[1]):
                old_val = thresh_map[x, y]
                normed_thresh_map[x, y] = (((old_val - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min

        if use_norm is True:
            [x, y] = np.where(normed_mag_map >= threshold*normed_thresh_map)
            tit = 'Threshold, %.2f of %s normed magnitude' % (threshold, threshold_type)
        else:
            [x, y] = np.where(mag_map >= threshold*thresh_map)
            tit = 'Threshold, %.2f of %s magnitude' % (threshold, threshold_type)


        phase_mask = np.ones(thresh_map.shape) * 100
        phase_mask[x, y] = phase_map[x, y]
        # tit = 'Threshold, %.2f of %s magnitude' % (threshold, threshold_type)
        # threshold_type = 'DCmag'
            
        [nullx, nully] = np.where(phase_mask == 100)
        phase_mask[nullx, nully] = np.nan
        phase_mask = np.ma.array(phase_mask)
        plt.imshow(phase_mask, cmap=colormap, vmin=-1*math.pi, vmax=math.pi)
        plt.axis('off')
        plt.title(tit)

        #-----------------------------------------------------------------
        # LEGEND
        #-----------------------------------------------------------------

        ax = fig.add_subplot(2,3,6, projection='polar')
        ax.set_theta_zero_location('W') # W puts 0 on RIGHT side...

        if CW is True:
            ax._direction = 2*np.pi # object moves toward bottom first (CW)
        else:
            ax._direction = -2*np.pi # objecct moves toward top first (CCW)

        norm = mpl.colors.Normalize(vmax=1*np.pi, vmin=-1*np.pi)
        #norm = mpl.colors.Normalize(vmax=2*np.pi, vmin=0)
        cb = mpl.colorbar.ColorbarBase(ax, cmap=cm.get_cmap(colormap),
                                        norm=norm, orientation='horizontal')
        # cb.ax.invert_xaxis()
        # cb.outline.set_visible(False)
        # ax.set_axis_off()
        ax.set_rlim([-1, 1])
        ax.axis('off')


        fig.add_subplot(2,3,4) # MEAN INTENSITY:
        mean_intensity = D[curr_key]['mean_intensity']
        plt.imshow(mean_intensity, cmap='hot')
        plt.axis('off')
        plt.colorbar()
        plt.title('mean intensity')

        fig.add_subplot(2,3,5) # MAGNITUDE
        plt.imshow(mag_map, cmap='gray')
        plt.colorbar()
        plt.axis('off')
        plt.title('magnitude')

        plt.tight_layout()

        plt.suptitle([date, experiment, curr_key])

        if is_blank is True:
            imname = '%s_phase_overlay_withkey_%s_threshold%.2f' % (os.path.splitext(curr_key)[0], threshold_type, threshold)
        else:
            if CW is True:
                imname = '%s_phase_overlay_withkey_%s_threshold%.2f' % (os.path.splitext(curr_key)[0], threshold_type, threshold)
            else:
                imname = '%s_phase_overlay_withkey_%s_threshold%.2f' % (os.path.splitext(curr_key)[0], threshold_type, threshold)
        # impath = os.path.join(outdir, imname+'.eps')
        # plt.savefig(impath, format='svg', dpi=1200)

        # impath = os.path.join(savedir, imname+'.png')
        # plt.savefig(impath, format='png')


        if use_norm:
            norm_flag = 'normed_'
        else:
            norm_flag = 'actual_'

        impath = os.path.join(figdir, colormap+'_'+norm_flag+imname+'.png')
        plt.savefig(impath, format='png')
        print impath

        plt.show()

        # ------------------------------------------------------------------
        # HSV PLOT
        # ------------------------------------------------------------------

        # THIS SEEMS TO WORK BETTER (FROM BOTTOM OF CIRC jnb)

        # thresh_method = 'magmax'
        # thresh_method = 'DC'
        # thresh_method = 'blank'

        # cutoff_val = 2

        # use_mag_max = 0
        # use_DC = 0
        # use_blank = 1

        # curr_key = d_key

        Ny = len(D[curr_key]['freqs'])/2.

        fig = plt.figure()
        mag_map = D[curr_key]['mag_map'] / Ny
        phase_map = D[curr_key]['phase_map']

        DC_map = D[curr_key]['DC_mag']/Ny
        if threshold_type=='blank':
            blank_map = D[blank_key[0]]['mag_map']/Ny

        print "mag range: ", mag_map.min(), mag_map.max()
        print "phase range: ", phase_map.min(), phase_map.max()


        # Get normed PHASE map for stimulation condN for HSV composite:

        old_min = -math.pi #phase_map.min()
        old_max = math.pi #phase_map.max()
        new_min = 0
        new_max = 1
        normed_phase_map = np.zeros(phase_map.shape)
        for x in range(phase_map.shape[0]):
            for y in range(phase_map.shape[1]):
                old_val = phase_map[x, y]
                normed_phase_map[x, y] = (((old_val - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min


        # Get normed MAGNITUDE map for stimulation condN for HSV composite:
        old_min = mag_map.min()
        old_max = mag_map.max()
        new_min = 0
        new_max = 1
        normed_mag_map = np.zeros(mag_map.shape)
        for x in range(mag_map.shape[0]):
            for y in range(mag_map.shape[1]):
                old_val = mag_map[x, y]
                normed_mag_map[x, y] = (((old_val - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min

                
        # CHOOSE MAG METHOD TO THRESHOLD OFF OF:

        import copy

        if threshold_type=='magmax':
            thresh_map = copy.deepcopy(mag_map)
            plot_title = 'masked, >= %s of mag max' % str(threshold)
        elif threshold_type=='DC':
            thresh_map = copy.deepcopy(DC_map)
            plot_title = 'masked, >= %s of DC mag' % str(threshold)
        elif threshold_type=='blank':
            thresh_map = copy.deepcopy(blank_map)
            plot_title = 'masked, >= %s of blank mag' % str(threshold)

            
        # # NORMALIZE THERSHOLD MAP???
        old_min = thresh_map.min()
        old_max = thresh_map.max()
        new_min = 0
        new_max = 1
        normed_thresh_map = np.zeros(thresh_map.shape)
        for x in range(thresh_map.shape[0]):
            for y in range(thresh_map.shape[1]):
                old_val = thresh_map[x, y]
                normed_thresh_map[x, y] = (((old_val - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min


        hue = normed_phase_map
        sat = np.ones(hue.shape)
        val = normed_mag_map

        print "HUE range: ", hue.min(), hue.max()
        print "VAL range: ", val.min(), val.max()

        HSV = np.ones(val.shape + (3,))
        HSV[...,0] = hue
        HSV[...,2] = sat * 1
        HSV[...,1] = val

        # plt.imshow(normed_phase_map, cmap='hsv')
        # plt.colorbar()

        # print normed_mag_map.min()
        # print normed_mag_map.max()


        ## REMOVE BELOW THRESH:

        print "Cutoff at: ", threshold

        nons = []
        for x in range(mag_map.shape[0]):
            for y in range(mag_map.shape[1]):
        # #         if mag_map[x, y] < thresh_map[x, y]*cutoff_val:
        # #         if normed_mag_map[x, y] < normed_thresh_map[x, y]*cutoff_val:
        #         if not normed_mag_map[x, y] >= normed_thresh_map[x, y]*threshold:
        #             nons.append([x,y])

                if use_norm is True:
                    if normed_mag_map[x, y] < normed_thresh_map[x, y]*threshold:
                        nons.append([x,y])
                else:
                    if mag_map[x, y] < thresh_map[x, y]*threshold:
                        nons.append([x,y])
                

        # NOTE ON THRESHOLDING:
        # If use normed-mag against normed-threshold-map, get good removal of baddies.
        # BUT, if use actual mag-map values against actual blank/DC map conditions, too much stuff gets included...

                        
        print "N pixels below threshold:  ", len(nons)

        ##
        # HSV TO RGB:

        import colorsys
        convmap = np.empty(HSV.shape)

        for i in range(HSV.shape[0]):
            for j in range(HSV.shape[1]):

                    convmap[i, j, :] = colorsys.hsv_to_rgb(HSV[i,j,:][0], HSV[i,j,:][1], HSV[i,j,:][2])
                    
        print "HSV range: ", HSV.min(), HSV.max()
        # print convmap[i,j,:]
        # print convmap.min()

        ##
        # MASK:

        alpha_channel = np.ones(convmap[:,:,1].shape)
        # print alpha_channel.shape
        for i in nons:
            alpha_channel[i[0], i[1]] = 0

        composite = np.empty((alpha_channel.shape[0], alpha_channel.shape[1], 4))
        composite[:,:,0:3] = convmap[:,:,:]

        composite[:,:,3] = alpha_channel


        # PLOT:


        # MAKE AND SAVE FIGURE:

        date = os.path.split(os.path.split(os.path.split(outdir)[0])[0])[1]
        experiment = os.path.split(os.path.split(outdir)[0])[1]
                
        # fig = plt.figure(figsize=(10,10))

        plt.subplot(1,3,1)
        plt.imshow(surface, 'gray')
        plt.axis('off')

        plt.subplot(1,3,2)
        plt.imshow(surface, 'gray')
        plt.imshow(composite, 'hsv')
        plt.axis('off')
        plt.title(plot_title)
        # plt.colorbar()

        # plt.subplot(1,3,3)
        # plt.imshow(legend, cmap='hsv')
        # plt.axis('off')
        ax = fig.add_subplot(1,3,3, projection='polar')
        ax.set_theta_zero_location('W') # W puts 0 on RIGHT side...

        if CW is True:
            ax._direction = 2*np.pi # object moves toward bottom first (CW)
        else:
            ax._direction = -2*np.pi # objecct moves toward top first (CCW)

        norm = mpl.colors.Normalize(vmax=1*np.pi, vmin=-1*np.pi)
        #norm = mpl.colors.Normalize(vmax=2*np.pi, vmin=0)
        cb = mpl.colorbar.ColorbarBase(ax, cmap=cm.get_cmap('hsv'),
                                        norm=norm, orientation='horizontal')
        # cb.ax.invert_xaxis()
        # cb.outline.set_visible(False)
        # ax.set_axis_off()
        ax.set_rlim([-1, 1])
        ax.axis('off')


        plt.suptitle([date, experiment, curr_key])

        plt.tight_layout()
            
        # impath = os.path.join(outdir, imname+'.svg')
        # plt.savefig(impath, format='svg', dpi=1200)


        imname = "pastel_%s_thresh%s%%_MASKED_HSV_%s" % (threshold_type, str(int(threshold*100)), os.path.splitext(curr_key)[0])

        savedir = os.path.split(outdir)[0]
        figdir = os.path.join(savedir, 'figures')
        if not os.path.exists(figdir):
            os.makedirs(figdir)
            
        impath = os.path.join(figdir, norm_flag+imname+'.png')
        plt.savefig(impath, format='png')
        print impath

        plt.show()

        # In[1416]:

        # Checkout the threshold map...
        plt.figure()

        plt.subplot(2,2,1)
        plt.imshow(mag_map, vmin=0, vmax=mag_map.max(), cmap='hot') 
        plt.title('magnitude at target freq')
        plt.colorbar()

        plt.subplot(2,2,2)
        plt.imshow(normed_mag_map, vmin=0, vmax=1, cmap='hot')
        plt.title('normed magnitude')
        plt.colorbar()

        plt.subplot(2,2,3)
        plt.imshow(thresh_map,  vmin=0, vmax=mag_map.max(), cmap='hot')
        if threshold_type=='DC':
            plt.title('DC magnitude')
        elif threshold_type=='blank':
            plt.title('BLANK magnitude')
        elif threshold_type=='magmax':
            plt.title('magnitude map')
        plt.colorbar()

        plt.subplot(2,2,4)
        plt.imshow(normed_thresh_map, vmin=0, vmax=1, cmap='hot')
        plt.title('normed mag %s' % threshold_type)
        plt.colorbar()

        imname = "magnitude_%s_thresh%s%%_%s" % (threshold_type, str(int(threshold*100)), os.path.splitext(curr_key)[0])

        savedir = os.path.split(outdir)[0]
        figdir = os.path.join(savedir, 'figures')
        if not os.path.exists(figdir):
            os.makedirs(figdir)
            
        impath = os.path.join(figdir, imname+'.png')
        plt.savefig(impath, format='png')
        print impath

        plt.show()
