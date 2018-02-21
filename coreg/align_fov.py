#!/usr/bin/env python2

import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import math
import os
import cv2
import h5py
import traceback
import cPickle as pkl
import numpy as np
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
# get_ipython().magic(u'matplotlib inline')
import copy

import optparse



#def click_and_crop(event, x, y, flags, param):
#    # grab references to the global variables
#    global refPt, cropping, refPt_pre, cropRO#I
#
#    # if the left mouse button was clicked, record the starting
#    # (x, y) coordinates and indicate that cropping is being
#    # performed
##     if cropROI is False:
#    if event == cv2.EVENT_LBUTTONDOWN:
#        refPt.append((x, y))
#        cv2.circle(image, refPt[-1], 1, (0,0,255), -1)
#        cv2.imshow("image", image)
#        cv2.putText(image, '%i' % len(refPt), refPt[-1]+5, cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 2,cv2.LINE_AA)

#     else:
#         if event == cv2.EVENT_LBUTTONDOWN:
#             refPt = [(x, y)]
#             cropping = True

#         # check to see if the left mouse button was released
#         elif event == cv2.EVENT_LBUTTONUP:
#             # record the ending (x, y) coordinates and indicate that
#             # the cropping operation is finished
#             refPt.append((x, y))
#             cropping = False

#             # draw a rectangle around the region of interest
#             cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
#             cv2.imshow("image", image)

#         if not refPt == refPt_pre:
#             print refPt
#             refPt_pre = refPt



def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T

    return np.vstack([np.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         np.matrix([0., 0., 1.])])


def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im

#%%

def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping, refPt_pre, cropROI, image

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
#     if cropROI is False:
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append((x, y))
        cv2.circle(image, refPt[-1], 1, (0,0,255), -1)
        cv2.imshow("image", image)
        #cv2.putText(image, '%i' % len(refPt), (refPt[-1][0]-5, refPt[-1][1]+5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)


def get_registration_points(sample):

    #image = copy.copy(sample)
    #refPt = []
    #cropping = False
    global refPt, image

    clone = image.copy()

    cv2.startWindowThread()
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image", click_and_crop)

    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            image = clone.copy()
            refPt = []

        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break

    # close all open windows
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    return image, refPt

def plot_transforms(sampleimg, referenceimg, out, out_path='/tmp'):

    print "Making figure..."
    # plt.figure(figsize=(10,0))
    plt.figure()

    plt.subplot(221)
    plt.imshow(sampleimg, cmap='gray')
    plt.axis('off')
    plt.title('original sample')

    plt.subplot(222)
    plt.imshow(referenceimg, cmap='gray')
    plt.axis('off')
    plt.title('original reference')

    plt.subplot(223)
    plt.imshow(out, cmap='gray')
    plt.axis('off')
    plt.title('warped sample')

    plt.subplot(224)
    # plt.imshow(SAMPLE, cmap='gray')
    # plt.imshow(out, cmap='jet', alpha=0.2)
    merged = np.zeros((reference.shape[0], reference.shape[1], 3), dtype=np.uint8)
    merged[:,:,0] = reference #cv2.cvtColor(reference)#, cv2.COLOR_RGB2GRAY)
    merged[:,:,1] = out #cv2.cvtColor(outi) #, cv2.COLOR_RGB2GRAY)
    plt.imshow(merged)
    plt.axis('off')
    plt.title('combined')

    plt.tight_layout()

    npoints = len(sample_pts)
    # outpath = './output'
    imname = 'warp_transforms_npoints%i.png' % (npoints)
    print imname
    plt.savefig(os.path.join(out_path, imname))
    plt.show()

def plot_merged(reference, out, out_path='/tmp'):
    print "Getting MERGED figure..."

    plt.figure()
    merged = np.zeros((reference.shape[0], reference.shape[1], 3), dtype=np.uint8)
    merged[:,:,0] = reference
    merged[:,:,1] = out
    plt.imshow(merged)
    plt.axis('off')

    imname = 'overlay_npoints%i' % npoints #(sample_fn, reference_fn, npoints)
    print os.path.join(out_path, imname)
    plt.savefig(os.path.join(out_path, imname))

    plt.show()

#%%
parser = optparse.OptionParser()

parser.add_option('-r', '--reference', action="store", dest="reference",
                  default="", help="Path to reference image (to align to)")
parser.add_option('-s', '--sample', action="store", dest="sample",
                  default="", help="Path to sample image (to align to the reference")
parser.add_option('-o', '--outpath', action="store", dest="outpath",
                  default="/tmp", help="Path to the save ROIs")

parser.add_option('--C', '--crop', action="store_true", dest="crop", default=False, help="Path to save ROI")

(options, args) = parser.parse_args()

# Get paths from options:
reference_path = options.reference
sample_path = options.sample
out_path = options.outpath

#%%
#reference_path = '/nas/volume1/2photon/data/CE074/20180215/coregistration/window.tif'
#sample_path = '/nas/volume1/2photon/data/CE074/20180215/coregistration/V1_surface_sum_transformed.tif'
#out_path = '/nas/volume1/2photon/data/CE074/20180215/coregistration/output'

#%%

if not os.path.exists(out_path):
    os.makedirs(out_path)

#%%

# Load images:
reference = cv2.imread(reference_path)

# Make sure images are gray-scale:
if len(reference.shape)==2: # not RGB
    reference = cv2.cvtColor(reference, cv2.COLOR_GRAY2RGB) # make it 3D
    referenceimg = cv2.cvtColor(reference, cv2.COLOR_GRAY2RGB)
else:
    referenceimg = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)
    reference = referenceimg[:,:,1]
print "Reference size is: ", reference.shape



#%% GET REFERENCE POINTS:

refPt = []
image = copy.copy(reference)
ref_pts_img, reference_pts = get_registration_points(reference)
#reference_pts = copy.copy(refPt)
npoints = len(reference_pts)

# DISPLAY REF IMAGE:
print "GOT %i reference test POINTS: " % npoints
print reference_pts
# Save chosen REF points:
ref_points_path = os.path.join(os.path.split(reference_path)[0], 'reference_points.png')
cv2.imwrite(ref_points_path, ref_pts_img)
print "Saved REFERENCE points to:\n%s" % ref_points_path

#%%

sample = cv2.imread(sample_path)
if len(sample.shape)==2:
    sample = cv2.cvtColor(sample, cv2.COLOR_GRAY2RGB)
    sampleimg = cv2.cvtColor(sample, cv2.COLOR_GRAY2RGB)
else:
    sampleimg = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
    sample = sampleimg[:,:,1]

print "Sample that will be aligned to ref is: ", sample.shape

refPt = []
image = copy.copy(sample)
sample_pts_img, sample_pts = get_registration_points(image)
#sample_pts = copy.copy(refPt)
npoints = len(sample_pts)
print "GOT %i SAMPLE POINTS: " % npoints
print sample_pts


# Save chosen SAMPLE points:
sample_points_path = os.path.join(os.path.split(sample_path)[0], 'sample_points.png')
cv2.imwrite(sample_points_path, sample_pts_img)
print "Saved SAMPLE points to:\n%s" % sample_points_path


#%%
# Use SAMPLTE and TEST points to align:
sample_mat = np.matrix([i for i in sample_pts])
reference_mat = np.matrix([i for i in reference_pts])
M = transformation_from_points(reference_mat, sample_mat)

#Re-read sample image as grayscale for warping:
#sampleG = cv2.imread(sample_path, 0)
#out = warp_im(SAMPLE, M, REF.shape)
out = warp_im(sample, M, reference.shape)
#outbig = warp_im(sample, M, [reference.shape[1], reference.shape[1]])
# out_map = warp_im(retinomap, M, SAMPLE.shape)

coreg_info = dict()
coreg_info['reference_file'] = reference_path
coreg_info['sample_file'] = sample_path
coreg_info['reference_points_x'] = tuple(p[0] for p in reference_pts)
coreg_info['reference_points_y'] = tuple(p[1] for p in reference_pts)
coreg_info['sample_points_x'] = tuple(p[0] for p in sample_pts)
coreg_info['sample_points_y'] = tuple(p[1] for p in sample_pts)
#coreg_info['transform_mat'] = M

coreg_hash = hash(frozenset(coreg_info.items()))
print "COREG HASH: %s" % coreg_hash

alignment_filepath = os.path.join(out_path, 'alignment.hdf5')
if os.path.exists(alignment_filepath):
    results = h5py.File(alignment_filepath, 'a')
else:
    results = h5py.File(alignment_filepath, 'w')

try:
    # Add sources:
    if 'sources' not in results.keys():
        sources = results.create_group('sources')
    else:
        sources = results['sources']

    if reference_path not in sources.keys():
        ref = sources.create_dataset('reference', reference.shape, reference.dtype)
        ref[...] = reference
        ref.attrs['filepath'] = reference_path

    sample_name = os.path.split(sample_path)[1]
    if sample_name not in sources.keys():
        sam = sources.create_dataset(sample_name, sample.shape, sample.dtype)
        sam[...] = sample
        sam.attrs['filepath'] = sample_path

    # Add coregistration info and results:
    if 'transforms' not in results.keys():
        transforms = results.create_group('transforms')
    else:
        transforms = results['transforms']

    if coreg_hash not in transforms.keys():
        match = transforms.create_dataset(str(coreg_hash), M.shape, M.dtype)
        match[...] = M
        for info in coreg_info.keys():
            match.attrs[info] = coreg_info[info]
except Exception as e:
    print "ERROR saving results to coreg hdf5."
    print "Sample: %s" % sample_path
    print "Npoints: %i" % npoints
    traceback.print_exc()
finally:
    results.close()


T = dict()
T['tMAT'] = M
T['sample'] = sample_path
T['reference'] = reference_path
#
#mat_fn = 'temp2sample'+'.mat'
## scipy.io.savemat(os.path.join(source_dir, condition, tif_fn), mdict=pydict)
#
#import scipy.io
#scipy.io.savemat(os.path.join(out_path, mat_fn), mdict=T)
#print os.path.join(out_path, 'mw_data', mat_fn)
#
pkl_fn = 'temp2sample'+'.pkl'
with open(os.path.join(out_path,pkl_fn), 'wb') as fn:
    pkl.dump(T, fn)


 #%% PLOT:

plot_transforms(sampleimg, referenceimg, out, out_path=out_path)
plot_merged(reference, out, out_path=out_path)

#%%
## PAD images to the same size so that can draw overlay:
#maxreference = max(reference.shape)
#maxsample = max(sample.shape)
#maxdim = max([maxsample, maxreference])
#print "MAXDIM:", maxdim
#overlay = np.zeros((maxdim, maxdim, 3), dtype=np.uint8)
#
#adjust_dims_reference = maxdim - np.array(reference.shape)
#adjust_dims_reference_idx = np.where(adjust_dims_reference!=0)[0]
#print "Ref, adjust dims by: ", adjust_dims_reference
#print "Ref, adjust dim idxs: ", adjust_dims_reference_idx
#adjust_dims_sample = maxdim - np.array(outbig.shape)
#adjust_dims_sample_idx = np.where(adjust_dims_sample!=0)[0]
#print "Sample, adjust dims by: ", adjust_dims_sample
#print "Sample, adjust dims by: ", adjust_dims_sample_idx
#if adjust_dims_reference_idx==0:
#    temppad = np.pad(reference, ((0, maxdim-reference.shape[0]), (0,0)), 'constant')
#else:
#    temppad = np.pad(reference, ((0,0), (maxdim-reference.shape[1], 0)), 'constant')
#if len(adjust_dims_sample_idx)>0:
#    if adjust_dims_sample_idx==0:
#	outpad = np.pad(outbig, ((0, maxdim-outbig.shape[0]), (0,0)), 'constant')
#    else:
#	outpad = np.pad(outbig, ((0,0), (maxdim-outbig.shape[1], 0)), 'constant')
#else:
#    outpad = np.copy(outbig)
#print "temppad: ", temppad.shape
#print "outpad: ", outpad.shape
#
#overlay[:,:,0] = temppad
#overlay[:,:,1] = outpad
#
#plt.figure()
#plt.imshow(overlay)
#plt.axis('off')
#imname = 'merged_npoints%i.png' % npoints
#
#plt.savefig(os.path.join(out_path, imname))
##temppad = np.pad(temp, ((), ()), 'constant')
#



#
## Overlay phase map
#print "Displaying PHASE map onto figure..."
##out_map = np.ma.masked_where(out == 0, out)
##out_map_mask = np.ma.masked_where(out_map == 0, out_map)
#print rmap.shape
##gray = cv2.cvtColor(outpad, cv2.COLOR_RGB2GRAY)
#
##out_mask = np.full((out.shape[0], out.shape[1]), 0, dtype=np.uint8)
#mask1 = np.full((out.shape[0], out.shape[1]), 0, dtype=np.uint8)
#print "mask1 shape:", mask1.shape
#mask1[np.where(outpad>0)] = 255
#fg = cv2.bitwise_or(out, out, mask=mask1)
#mask1 = cv2.bitwise_not(mask1)
#background = np.full(out.shape, 255, dtype=np.uint8)
#bk = cv2.bitwise_or(background, background, mask=mask1)
#final=cv2.bitwise_or(fg, bk)
#
#plt.figure()
#plt.imshow(reference)
#plt.imshow(final, alpha=.5555)
#if no_map is False:
#    plt.imshow(rmap, cmap=colormap, alpha=0.5)
#plt.show()
##=======
##out_map = np.ma.masked_where(out == 0, out)
##out_map_mask = np.ma.masked_where(out_map == 0, out_map)
##
##plt.figure()
#merged_gray = cv2.cvtColor(merged, cv2.COLOR_RGB2GRAY)
#print "merged map:", merged_gray.shape
#print "retino map: ", rmap.shape
##plt.imshow(reference, cmap='gray', alpha=0.5)
#rmap_rgb = cv2.cvtColor(rmap, cv2.COLOR_RGB2BGR)
##plt.imshow(merged_gray, cmap='gray', alpha=1) #out_map_mask, cmap='gray', alpha=.5)
##plt.imshow(reference, cmap='gray', alpha=1.0)
##plt.imshow(rmap_rgb, cmap=colormap, alpha=0.8, vmin=0, vmax=math.pi*2)
#plt.imshow(out, cmap='gray', alpha=0.7)
##plt.imshow(merged_gray, cmap='gray', alpha=0.5) #out_map_mask, cmap='gray', alpha=.5)
##plt.imshow(rmap_rgb, cmap=colormap, alpha=0.5, vmin=0, vmax=math.pi*2)
##plt.imshow(out_map_mask, cmap='gray', alpha=.75)
#
#plt.axis('off')
#plt.show()
#imname = 'points_svd_S-%s_T-%s_npoints-%i_RETINO' % (sample_fn, reference_fn, npoints)
#print os.path.join(out_path, imname)
#plt.savefig(os.path.join(out_path, imname))
#
