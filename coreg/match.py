#!/usr/bin/env python2

# import the necessary packages
import numpy as np
import argparse
import imutils
import glob
import cv2
import os

import matplotlib.pyplot as plt
# construct the argument parser and parse the arguments

refdir = '/home/juliana/Repositories/retinotopy-mapper/coreg/samples/REF.tif'
# imdir = '/home/juliana/Repositories/retinotopy-mapper/coreg/samples/im_zoom_bw.png'
# imdir = '/home/juliana/Repositories/retinotopy-mapper/coreg/samples/im_zoom_enhanced.png'
imdir = '/home/juliana/Repositories/retinotopy-mapper/coreg/samples/tests/nolens_BW.png'

outdir = '/home/juliana/Repositories/retinotopy-mapper/coreg/samples/figures/'

if not os.path.exists(outdir):
    os.makedirs(outdir)

# ap = argparse.ArgumentParser()
# ap.add_argument("-t", "--template", required=True, help="Path to template image")
# ap.add_argument("-i", "--images", required=True,
# 	help="Path to images where template will be matched")
# ap.add_argument("-v", "--visualize",
# 	help="Flag indicating whether or not to visualize each iteration")
# args = vars(ap.parse_args())
 
# load the image image, convert it to grayscale, and detect edges
# template = cv2.imread(args["template"])
# template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
# template = cv2.Canny(template, 50, 200)
# (tH, tW) = template.shape[:2]
# cv2.imshow("Template", template)


# ------------------------------------------------------------------------------
# CHECK OUT CANNY EDGE DETECTION:
# ------------------------------------------------------------------------------

template = cv2.imread(refdir, 0)
# template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
cn_min = 100
cn_max = 200
template_edges = cv2.Canny(template, cn_min, cn_max)
# plt.imshow(template_edges)
(tH, tW) = template_edges.shape[:2]

bi_min = 150
bi_max = 255
ret,thresh1 = cv2.threshold(template,bi_min,bi_max,cv2.THRESH_BINARY)
# plt.imshow(thresh1)

fig = plt.figure(figsize=(10,5))
plt.subplot(1,3,1)
plt.imshow(template, cmap='gray')
plt.subplot(1,3,2)
plt.imshow(template_edges, cmap='gray')
plt.title('edges [%i %i]' % (cn_min, cn_max))
plt.subplot(1,3,3)
plt.imshow(thresh1, cmap='gray')
plt.title('binary thresh [%i %i]' % (bi_min, bi_max))

plt.suptitle('REFERENCE')
# savedir = '/media/labuser/IMDATA1/widefield/AH03/FIGS'

imname = 'REF_edge.png'
fig.savefig(outdir + '/' + imname)
# fig.savefig(savedir + '/' + imname)
print "FIG 1: ", outdir + '/' + imname


# Compare original, BW, and edges
# plt.figure(figsize=(20,10))
# plt.subplot(1,3,1)
# plt.imshow(template, cmap='gray')
# plt.title("reference image")
# plt.subplot(1,3,2)
# plt.imshow(template_bw, cmap='gray')
# plt.title("BW convert")
# plt.subplot(1,3,3)
# plt.imshow(template_edges)
# plt.title("edges")
# plt.tight_layout()


# quick test of template matching:

image = cv2.imread(imdir, 0)
# template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
cn_min = 50
cn_max = 200
image_edges = cv2.Canny(image, cn_min, cn_max)
plt.imshow(image_edges)

(tH, tW) = image_edges.shape[:2]

bi_min = 150
bi_max = 255
ret,thresh1 = cv2.threshold(image,bi_min,bi_max,cv2.THRESH_BINARY)
# plt.imshow(thresh1)

fig = plt.figure(figsize=(10,5))
plt.subplot(1,3,1)
plt.imshow(image, cmap='gray')
plt.subplot(1,3,2)
plt.imshow(image_edges, cmap='gray')
plt.title('edges [%i %i]' % (cn_min, cn_max))
plt.subplot(1,3,3)
plt.imshow(thresh1, cmap='gray')
plt.title('binary thresh [%i %i]' % (bi_min, bi_max))

plt.suptitle('SAMPLE IMAGE %s' % os.path.split(imdir)[1])

imname = 'SAMPLE_edge_%s.png' % os.path.split(imdir)[1]
fig.savefig(outdir + '/' + imname)
# fig.savefig(savedir + '/' + imname)
print "FIG 1: ", outdir + '/' + imname



# ------------------------------------------------------------------------------
# Gaussian blur??  okay results when using MATLAB's version...
# ------------------------------------------------------------------------------

sigma = 8
im_gauss = scipy.ndimage.filters.gaussian_filter(image, sigma)
plt.imshow(im_gauss, cmap='gray')
im_gauss_edges = cv2.Canny(im_gauss, cn_min, cn_max)

plt.imshow(im_gauss_edges)



# RESOURCE: http://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html#gsc.tab=0


# ------------------------------------------------------------------------------
# ADAPTIVE THRESHOLDING?
# ------------------------------------------------------------------------------

currdir = imdir #refdir 
img = cv2.imread(currdir,0)
# img = cv2.medianBlur(img,5)
img = cv2.medianBlur(img,3)

ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#             cv2.THRESH_BINARY,11,2)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,5)

th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
 
titles = ['Original Image(Median Blur)', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

for i in xrange(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
    plt.show()

plt.suptitle('Adaptive thresholding for binarization')

imname = 'adaptive_thresh_%s.png' % os.path.split(currdir)[1]
fig.savefig(outdir + '/' + imname)
# fig.savefig(savedir + '/' + imname)
print "FIG 1: ", outdir + '/' + imname

plt.show()



# ------------------------------------------------------------------------------
# BW HISTOGRAM? OTSU'S METHOD...
# ------------------------------------------------------------------------------

currdir = imdir #refdir

img = cv2.imread(currdir,0)

# global thresholding
ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

# Otsu's thresholding
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Otsu's thresholding after Gaussian filtering
sig=4
blur = cv2.GaussianBlur(img,(5,5),sig)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# plot all the images and their histograms
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
          'Original Noisy Image','Histogram',"Otsu's Thresholding",
          'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]

for i in xrange(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])

plt.suptitle("Otsu's thresholding with orig and filt images")

imname = 'otsu_hist_%s.png' % os.path.split(currdir)[1]
fig.savefig(outdir + '/' + imname)
# fig.savefig(savedir + '/' + imname)
print "FIG 1: ", outdir + '/' + imname

plt.show()



# ------------------------------------------------------------------------------
# HIGH-PASS?
# ------------------------------------------------------------------------------

currdir = imdir #refdir

img = cv2.imread(currdir,0)

filt = 'median' #'gauss'
# filt = 'gaussian'
sig=3
win=5
sz = 9
if filt=='gaussian':
	blur = cv2.GaussianBlur(img,(win,win),sig)
elif filt=='median':
	blur = cv2.medianBlur(img,sz)

imsub = img - blur

fig = plt.figure(figsize=(10,3))
plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title('original image')
plt.subplot(1,3,2)
plt.imshow(blur, cmap='gray')
if filt=='gaussian':
	plt.title('%s blur, win %i, sig %i' % (filt, win, sig))
else:
	plt.title('%s blur, sz %i' % (filt, sz))

plt.subplot(1,3,3)
plt.imshow(imsub, cmap='gray')
plt.title('high-pass')

if filt=='gaussian':
	imname = '%s_subtract_win%i_sig%i_%s.png' % (filt, win, sig, os.path.split(currdir)[1])
else:
	imname = '%s_subtract_sz%i_%s.png' % (filt, sz, os.path.split(currdir)[1])

fig.savefig(outdir + '/' + imname)
# fig.savefig(savedir + '/' + imname)
print "FIG 1: ", outdir + '/' + imname

plt.show()





# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------




# loop over the images to find the template in
for imagePath in glob.glob(args["images"] + "/*.jpg"):
	# load the image, convert it to grayscale, and initialize the
	# bookkeeping variable to keep track of the matched region
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	found = None

	# loop over the scales of the image
	for scale in np.linspace(0.2, 1.0, 20)[::-1]:
		# resize the image according to the scale, and keep track
		# of the ratio of the resizing
		resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
		r = gray.shape[1] / float(resized.shape[1])

		# if the resized image is smaller than the template, then break
		# from the loop
		if resized.shape[0] < tH or resized.shape[1] < tW:
			break

		# detect edges in the resized, grayscale image and apply template
		# matching to find the template in the image
		edged = cv2.Canny(resized, 50, 200)
		result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
		(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

		# check to see if the iteration should be visualized
		if args.get("visualize", False):
			# draw a bounding box around the detected region
			clone = np.dstack([edged, edged, edged])
			cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
				(maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
			cv2.imshow("Visualize", clone)
			cv2.waitKey(0)

		# if we have found a new maximum correlation value, then ipdate
		# the bookkeeping variable
		if found is None or maxVal > found[0]:
			found = (maxVal, maxLoc, r)

	# unpack the bookkeeping varaible and compute the (x, y) coordinates
	# of the bounding box based on the resized ratio
	(_, maxLoc, r) = found
	(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
	(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

	# draw a bounding box around the detected result and display the image
	cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
	cv2.imshow("Image", image)
	cv2.waitKey(0)

	# detect edges in the resized, grayscale image and apply template
	# matching to find the template in the image
	edged = cv2.Canny(resized, 50, 200)
	result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
	(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

	# check to see if the iteration should be visualized
	if args.get("visualize", False):
		# draw a bounding box around the detected region
		clone = np.dstack([edged, edged, edged])
		cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
			(maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
		cv2.imshow("Visualize", clone)
		cv2.waitKey(0)

	# if we have found a new maximum correlation value, then ipdate
	# the bookkeeping variable
	if found is None or maxVal > found[0]:
		found = (maxVal, maxLoc, r)

# unpack the bookkeeping varaible and compute the (x, y) coordinates
# of the bounding box based on the resized ratio
(_, maxLoc, r) = found
(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

# draw a bounding box around the detected result and display the image
cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
cv2.imshow("Image", image)
cv2.waitKey(0)
