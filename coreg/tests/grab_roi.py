# import the necessary packages
import sys

sys.path.append('/usr/local/lib/python2.7/site-packages')

import os


import argparse
import cv2

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False

def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
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

    print refPt

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-O", "--outpath", required=True, help="Path to save ROI")


args = vars(ap.parse_args())

# load the image, clone it, and setup the mouse callback function
image = cv2.imread(args["image"])
outpath = args["outpath"]

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


# close all open windows
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)

# cv2.destroyAllWindows()