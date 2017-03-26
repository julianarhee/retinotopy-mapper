#!/usr/bin/env python2

import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
# get_ipython().magic(u'matplotlib inline')
import copy

import optparse



def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping, refPt_pre, cropROI

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
#     if cropROI is False:
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append((x, y))
        cv2.circle(image, refPt[-1], 1, (0,0,255), -1)
        cv2.imshow("image", image)

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


import cPickle as pkl
parser = optparse.OptionParser()

parser.add_option('-t', '--template', action="store", dest="template",
                  default="", help="Path to template image (to align to)")
parser.add_option('-s', '--sample', action="store", dest="sample",
                  default="", help="Path to sample image (to align to the template")
parser.add_option('-o', '--outpath', action="store", dest="outpath",
                  default="/tmp", help="Path to the save ROIs")
parser.add_option('-m', '--map', action="store", dest="map",
                  default="", help="Path to retino map for overlay")
<<<<<<< HEAD
parser.add_option('-c', '--cmap', action="store", dest="cmap",
                  default="spectral", help="Colormap for phase map")

parser.add_option('-S', '--struct', action="store_true", dest="use_struct",
                  default=False, help="Use struct from phase-map images instead of image path.")
parser.add_option('-p', '--structpath', action="store", dest="structpath",
                  default="", help="Path to struct.")

parser.add_option('--C', '--crop', action="store_true", dest="crop", default=False, help="Path to save ROI")

parser.add_option('--no-map', action="store_true", dest="no_map", default=False, help="No phase map to overlay.")

options, args) = parser.parse_args()


# Get paths from options:
# EXs:
# map_path = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/COREG/test/avg_az.png'
# template_path = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/COREG/test/widefield_surface.png'
# sample_path = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/COREG/test/tefo.png'
# out_path = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/COREG/test'
no_map = options.no_map
colormap = options.cmap
use_struct = options.use_struct
if use_struct is True:
    print "Using struct..."
    structpath = options.structpath
    import cPickle as pkl

    with open(structpath,'rb') as fp:
        I = pkl.load(fp)

        # I['az_phase'] = az_avg
        # I['vmin'] = vmin_val
        # I['vmax'] = vmax_val
        # I['az_legend'] = AZ_legend
        # I['surface'] = surface

    # template = I['surface']
    # from PIL import Image
    # template = Image.fromarray(template)
    # # image = Image.open(impath) #.convert('L')
    # surface = np.asarray(image)

    rmap = I['az_phase']
    vmin_val = I['vmin']
    vmax_val = I['vmax']
    legend = I['az_legend']

else:
    template_path = options.template
    template = cv2.imread(template_path, 0)
    map_path = options.map
    if map_path:

        rmap = cv2.imread(map_path)
    else:
        rmap = np.zeros(template.shape)




#map_path = options.map
#template_path = options.template
sample_path = options.sample

out_path = options.outpath
if not os.path.exists(out_path):
    os.makedirs(out_path)

# Load images:
emplate = cv2.imread(template_path)
# tiff = TIFF.open(template_path, mode='r')
# template = tiff.read_image().astype('float')
# tiff.close()

sample = cv2.imread(sample_path)
if len(template.shape)==2: # not RGB
    template = cv2.cvtColor(template, cv2.COLOR_GRAY2RGB) # make it 3D
else:
    template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)

if len(sample.shape)==2:
    sample = cv2.cvtColor(sample, cv2.COLOR_GRAY2RGB)
else:
    sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)

# if map_path:
#     rmap = cv2.imread(map_path)
# else:
#     rmap = np.zeros(template.shape)
# # too big:
# new_sz = 1024.0
# r = new_sz / img2.shape[1]
# dim = (int(new_sz), int(img2.shape[0] * r))
 
# # perform the actual resizing of the image and show it
# SAMPLE = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)


print "Template size is: ", template.shape
print "Sample to align is: ", sample.shape
print "MAP size is: ", rmap.shape

# First get SAMPLE points:
image = copy.copy(sample)

refPt = []
refPt_pre = []
cropping = False
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
        REF = clone.copy()
        refPt = []

    # if the 'c' key is pressed, break from the loop
    elif key == ord("c"):
        break

# close all open windows
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)

sample_pts = copy.copy(refPt)
npoints = len(sample_pts)
print "GOT %i SAMPLE POINTS: " % npoints
print sample_pts


# Save chosen SAMPLE points:
# sample_pts = [(int(i[0]), int(i[1])) for i in sample_pts]
# sampleX,sampleY = zip(*sample_pts)
# plt.figure()
# plt.imshow(sample, cmap='gray')
# plt.scatter(sampleX, sampleY, 'r*')
# plt.title('Sample Points')

fim = os.path.split(sample_path)[1]
sample_fn = fim.split('.')[0]

# imname = 'Sample-%s_npoints-%i' % (sample_fn, npoints)
# print imname
# plt.savefig(os.path.join(out_path, imname))

# plt.show()



# NOW, get TEMPLATE:
image = copy.copy(template)

refPt = []
cropping = False

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

template_pts = copy.copy(refPt)
npoints = len(template_pts)

# DISPLAY SAMPLE IMAGE:
print "GOT %i template test POINTS: " % npoints
print template_pts


# Save chosen SAMPLE points:
# templateX, templateY = zip(*template_pts)
# plt.figure()
# plt.imshow(template, cmap='gray')
# plt.scatter(templateX, templateY, 'r*')
# plt.title('Sample Points')

fim = os.path.split(template_path)[1]
template_fn = fim.split('.')[0]

# imname = 'Template-%s_npoints-%i' % (template_fn, npoints)
# print imname
# plt.savefig(os.path.join(out_path, imname))

# plt.show()






# Use SAMPLTE and TEST points to align:
sample_mat = np.matrix([i for i in sample_pts])
template_mat = np.matrix([i for i in template_pts])
M = transformation_from_points(template_mat, sample_mat)

#Re-read sample image as grayscale for warping:
#sampleG = cv2.imread(sample_path, 0)
#out = warp_im(SAMPLE, M, REF.shape)
out = warp_im(sample, M, template.shape)
# out_map = warp_im(retinomap, M, SAMPLE.shape)



T = dict()
T['tMAT'] = M
T['sample'] = sample_path
T['template'] = template_path

mat_fn = 'temp2sample'+'.mat'
# scipy.io.savemat(os.path.join(source_dir, condition, tif_fn), mdict=pydict)

import scipy.io
scipy.io.savemat(os.path.join(out_path, mat_fn), mdict=T)
print os.path.join(out_path, 'mw_data', mat_fn)

pkl_fn = 'temp2sample'+'.pkl'
with open(os.path.join(out_path,pkl_fn), 'wb') as fn:
    pkl.dump(T, fn)



print "Making figure..."

# plt.figure(figsize=(10,0))
plt.figure()

plt.subplot(221)
plt.imshow(sample, cmap='gray')
plt.axis('off')
plt.title('original sample')

plt.subplot(222)
plt.imshow(template, cmap='gray')
plt.axis('off')
plt.title('original template')

plt.subplot(223)
plt.imshow(out, cmap='gray')
plt.axis('off')
plt.title('warped sample')

plt.subplot(224)
# plt.imshow(SAMPLE, cmap='gray')
# plt.imshow(out, cmap='jet', alpha=0.2)
merged = np.zeros((template.shape[0], template.shape[1], 3), dtype=np.uint8)
merged[:,:,0] = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
merged[:,:,1] = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
plt.imshow(merged)
plt.axis('off')
plt.title('combined')

plt.tight_layout()

npoints = len(sample_pts)
# outpath = './output'
imname = 'points_svd_S-%s_T-%s_npoints-%i' % (sample_fn, template_fn, npoints)
print imname
plt.savefig(os.path.join(out_path, imname))

plt.show()



print "Getting MERGED figure..."

plt.figure()
merged = np.zeros((template.shape[0], template.shape[1], 3), dtype=np.uint8)
merged[:,:,0] = template
merged[:,:,1] = out
plt.imshow(merged)
plt.axis('off')

imname = 'overlay_merge_S-%s_T-%s_npoints-%i' % (sample_fn, template_fn, npoints)
print os.path.join(out_path, imname)
plt.savefig(os.path.join(out_path, imname))

plt.show()


# Overlay phase map
print "Displaying PHASE map onto figure..."
#out_map = np.ma.masked_where(out == 0, out)
#out_map_mask = np.ma.masked_where(out_map == 0, out_map)
gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
#out_mask = np.full((out.shape[0], out.shape[1]), 0, dtype=np.uint8)
mask1 = np.full((out.shape[0], out.shape[1]), 0, dtype=np.uint8)
mask1[np.where(gray>0)] = 255
fg = cv2.bitwise_or(out, out, mask=mask1)
mask1 = cv2.bitwise_not(mask1)
background = np.full(out.shape, 255, dtype=np.uint8)
bk = cv2.bitwise_or(background, background, mask=mask1)
final=cv2.bitwise_or(fg, bk)

plt.figure()
plt.imshow(template)
plt.imshow(final, alpha=.5555)
if no_map is False:
    plt.imshow(rmap, cmap=colormap, alpha=0.5)

#=======
#out_map = np.ma.masked_where(out == 0, out)
#out_map_mask = np.ma.masked_where(out_map == 0, out_map)
#
#plt.figure()
plt.imshow(template, cmap='gray')
plt.imshow(out_map_mask, cmap='gray', alpha=.5)
plt.imshow(rmap, cmap='spectral', alpha=0.5)
#plt.imshow(out_map_mask, cmap='gray', alpha=.75)

plt.axis('off')

imname = 'points_svd_S-%s_T-%s_npoints-%i_RETINO' % (sample_fn, template_fn, npoints)
print os.path.join(out_path, imname)
plt.savefig(os.path.join(out_path, imname))

