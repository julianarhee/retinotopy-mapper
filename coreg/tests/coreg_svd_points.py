
# coding: utf-8

# In[ ]:

# SEE:  https://matthewearl.github.io/2015/07/28/switching-eds-with-python/


# In[3]:

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


# In[4]:

import optparse

parser = optparse.OptionParser()

parser.add_option('-i', '--image', action="store", dest="image",
                  default="", help="Path to input image (to be aligned to)")

parser.add_option('-r', '--ref', action="store", dest="ref",
                  default="", help="Path to REF image (source to align)")

parser.add_option('-o', '--outpath', action="store", dest="outpath",
                  default="/tmp", help="Path to the save ROIs")

parser.add_option('-m', '--map', action="store", dest="map",
                  default="", help="Path to retino map for overlay")

parser.add_option('--C', '--crop', action="store_true", dest="crop", default=False, help="Path to save ROI")

(options, args) = parser.parse_args()

# global cropROI

# image = cv2.imread(options.image)
# outpath = options.outpath
# cropROI = options.crop

# In[5]:

map_path = options.map

image_path = options.image
ref_path = options.ref

# image_path = '/media/juliana/IMDATA/TEFO/20161218_CE025/COREG/widefield_surface.png'
# ref_path = '/media/juliana/IMDATA/TEFO/20161218_CE025/COREG/TEFO_surface.png'


# image_path = '/media/juliana/IMDATA/TEFO/20161219_JR030W/COREG/widefield_surface.png'
# ref_path = '/media/juliana/IMDATA/TEFO/20161219_JR030W/COREG/tefo.png'
# map_path = '/media/juliana/IMDATA/TEFO/20161219_JR030W/COREG/avg_az.png'

outpath = options.outpath


# retinomap = cv2.imread(map_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
retinomap = cv2.imread(map_path, 0)

print retinomap.shape
# './JR015W_test/20160906_REF.png'
# img1 = cv2.imread(ref_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
img1 = cv2.imread(ref_path, 0)


# In[6]:

# Get SAMPLE:
# img2 = cv2.imread('./images/tests/lens_BW_500ms.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)

# fim = 'lens_RGB_500ms.png'
# fim = 'surgery_bright.JPG'
# img2 = cv2.imread('./JR015W_test/tests/%s' % fim, cv2.CV_LOAD_IMAGE_GRAYSCALE)


# img2 = cv2.imread(image_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
img2 = cv2.imread(image_path, 0)


fim = os.path.split(image_path)[1]
# In[8]:

REF = img1 #img1
SAMPLE = img2 #img2

# too big:
new_sz = 1088.0
r = new_sz / img2.shape[1]
dim = (int(new_sz), int(img2.shape[0] * r))
 
# perform the actual resizing of the image and show it
SAMPLE = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)

print REF.shape
print SAMPLE.shape


# In[9]:

# refPt = []
# refPt_pre = []
# cropping = False

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


# In[10]:

# First get REF points:

image = copy.copy(REF)


# In[11]:

# GET POINTS:

refPt = []
refPt_pre = []
cropping = False

clone = image.copy()

cv2.startWindowThread()
cv2.namedWindow("image")
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


# In[12]:

# DISPLAY REF IMAGE:
print "GOT REF POINTS: "
print refPt
#plt.imshow(image) #, cmap='gray')

pts1 = copy.copy(refPt)

# for i in refPt:
#     plt.plot(i, 'r*')


# In[13]:

# NOW, get SAMPLE:

image = copy.copy(SAMPLE)

# GET POINTS:

refPt = []
refPt_pre = []
cropping = False

clone = image.copy()

cv2.startWindowThread()
cv2.namedWindow("image")
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


# In[14]:

# DISPLAY SAMPLE IMAGE:
print "GOT sample test POINTS: "
print refPt
#plt.imshow(image) #, cmap='gray')

pts2 = refPt #copy.copy(refPt)

# for i in refPt:
#     plt.plot(i, 'r*')


# In[15]:

# print pts1
# print pts2


# In[ ]:




# In[16]:

# TRY THIS WITH SVD:
# see:  https://matthewearl.github.io/2015/07/28/switching-eds-with-python/

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


# In[17]:


mat1 = np.matrix([i for i in pts1])
mat2 = np.matrix([i for i in pts2])
M = transformation_from_points(mat2, mat1)


# In[18]:

# out = warp_im(SAMPLE, M, REF.shape)
out = warp_im(REF, M, SAMPLE.shape)

out_map = warp_im(retinomap, M, SAMPLE.shape)


# In[23]:

print "Making figure..."

# plt.figure(figsize=(10,0))
plt.figure()

plt.subplot(221)
plt.imshow(REF, cmap='gray')
plt.axis('off')
plt.title('original ref')

plt.subplot(222)
plt.imshow(SAMPLE, cmap='gray')
plt.axis('off')
plt.title('original sample')

plt.subplot(223)
plt.imshow(out, cmap='gray')
plt.axis('off')
plt.title('warped ref')

plt.subplot(224)
# plt.imshow(SAMPLE, cmap='gray')
# plt.imshow(out, cmap='jet', alpha=0.2)
merged = np.zeros((SAMPLE.shape[0], SAMPLE.shape[1], 3), dtype=np.uint8)
merged[:,:,0] = SAMPLE
merged[:,:,1] = out
plt.imshow(merged)
plt.axis('off')
plt.title('combined')

plt.tight_layout()

input_image_name = fim.replace('.', '')
npoints = len(pts1)
# outpath = './output'
imname = 'points_svd_IM-%s_npoints-%i' % (input_image_name, npoints)
print imname
plt.savefig(os.path.join(outpath, imname))

plt.show()

# In[26]:

print "Displaying MASKED figure..."

# I1 = np.zeros((SAMPLE.shape[0], SAMPLE.shape[1], 3))
# I1[:,:,1] = SAMPLE

# I2 = np.zeros((SAMPLE.shape[0], SAMPLE.shape[1], 3))
# I2[:,:,0] = out

# # plt.figure(figsize=(20,10))
# plt.figure()
# plt.imshow(I1, alpha=0.2)
# plt.imshow(I2, alpha=0.2)
# plt.axis('off')
plt.figure()
merged = np.zeros((SAMPLE.shape[0], SAMPLE.shape[1], 3), dtype=np.uint8)
merged[:,:,0] = SAMPLE
merged[:,:,1] = out
plt.imshow(merged)
plt.axis('off')

imname = 'points_svd_IM-%s_npoints-%i_overlay' % (input_image_name, npoints)
print os.path.join(outpath, imname)
plt.savefig(os.path.join(outpath, imname))

plt.show()


# Overlay phase map

print "Displaying PHASE map onto figure..."

out_mask = np.ma.masked_where(out == 0, out)

out_map_mask = np.ma.masked_where(out_map == 0, out_map)

plt.figure()
plt.imshow(SAMPLE, cmap='gray')
plt.imshow(out_mask, cmap='gray', alpha=.75)
plt.imshow(out_map_mask, cmap='spectral', alpha=0.5)
plt.axis('off')

imname = 'points_svd_IM-%s_npoints-%i_RETINO' % (input_image_name, npoints)
print os.path.join(outpath, imname)
plt.savefig(os.path.join(outpath, imname))

