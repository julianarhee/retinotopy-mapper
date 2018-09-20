# import the necessary packages
import sys

sys.path.append('/usr/local/lib/python2.7/site-packages')

import os


import argparse
import cv2


import optparse

parser = optparse.OptionParser()

parser.add_option('--i', '--image', action="store", dest="image",
                  default="", help="Path to the image")

parser.add_option('--O', '--outpath', action="store", dest="outpath",
                  default="/tmp", help="Path to the save ROIs")

parser.add_option('--C', '--crop', action="store_true", dest="crop", default=False, help="Path to save ROI")



(options, args) = parser.parse_args()

global cropROI

image = cv2.imread(options.image)
outpath = options.outpath
cropROI = options.crop


# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
refPt_pre = []
cropping = False

print cropROI

def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping, refPt_pre, cropROI

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if cropROI is False:
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt.append((x, y))
            cv2.circle(image, refPt[-1], 1, (0,0,255), -1)
            cv2.imshow("image", image)

    else:
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt = [(x, y)]
            cropping = True

        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that
            # the cropping operation is finished
            refPt.append((x, y))
            cropping = False

            # draw a rectangle around the region of interest
            cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
            cv2.imshow("image", image)

        if not refPt == refPt_pre:
            print refPt
            refPt_pre = refPt

    # print refPt

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="Path to the image")
# ap.add_argument("-O", "--outpath", required=True, help="Path to save ROI")
# ap.add_argument("-C", "--crop", required=True, default=False, help="Path to save ROI")

# args = vars(ap.parse_args())

# # load the image, clone it, and setup the mouse callback function
# image = cv2.imread(args["image"])
# outpath = args["outpath"]

# import optparse

# parser = optparse.OptionParser()

# parser.add_option('--i', '--image', action="store", dest="image",
#                   default="", help="Path to the image")

# parser.add_option('--O', '--outpath', action="store", dest="outpath",
#                   default="/tmp", help="Path to the save ROIs")

# parser.add_option('--C', '--crop', action="store_true", dest="crop", default=False, help="Path to save ROI")



# (options, args) = parser.parse_args()

# image = cv2.imread(options.image)
# outpath = options.outpath
# cropROI = options.crop

if not os.path.exists(outpath):
    os.makedirs(outpath)

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

    # if the 'c' key is pressed, break from the loop
    elif key == ord("c"):
        break

    # if cv2.WaitKey(10) == 27:
    #     break

if cropROI is True:
    # if there are two reference points, then crop the region of interest
    # from teh image and display it
    if len(refPt) == 2:
        roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        cv2.namedWindow("ROI")
        cv2.imshow("ROI", roi)
        cv2.waitKey(0)
        roi_name = 'ROI_%s.jpg' % str(refPt)
        roi_path = os.path.join(outpath, roi_name)
        cv2.imwrite(roi_path,roi)
else:
    print "NO CROPPING"
    # GET POINTS:
    print refPt


# close all open windows
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)

# cv2.destroyAllWindows()