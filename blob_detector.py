## adapted from https://www.geeksforgeeks.org/find-circles-and-ellipses-in-an-image-using-opencv-python/

import cv2
import numpy as np
import argparse
from skimage import (
    data, restoration, util
)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

# Load image
image = cv2.imread(args["image"])
#image = util.invert(ds)
#image = cv2.imread('C://gfg//images//blobs.jpg', 0)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ds = cv2.pyrDown(gray)
image_inverted = util.invert(ds)
background_inverted = restoration.rolling_ball(image_inverted, radius=15)
filtered_image_inverted = image_inverted - background_inverted
filtered_image = util.invert(filtered_image_inverted)

# Set our filtering parameters
# Initialize parameter setting using cv2.SimpleBlobDetector
params = cv2.SimpleBlobDetector_Params()
# Thresholding Params
params.minThreshold = 1;
params.maxThreshold = 5000

# Set Area filtering parameters
params.filterByArea = True
params.minArea = 200

# Set Circularity filtering parameters
params.filterByCircularity = True
params.minCircularity = 0.5

# Set Convexity filtering parameters
params.filterByConvexity = True
params.minConvexity = 0.01

# Set inertia filtering parameters
params.filterByInertia = False
params.minInertiaRatio = 0.01

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs
keypoints = detector.detect(image)
#keypoints = detector.detect(filtered_image)


# Draw blobs on our image as red circles
blank = np.zeros((1, 1))
#blobs = cv2.drawKeypoints(filtered_image, keypoints, blank, (0, 0, 255),
                          #cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 0, 255),
                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

x1 = keypoints[0].pt[0]
y1 = keypoints[0].pt[1]

number_of_blobs = len(keypoints)
text = "Number of Circular Blobs: " + str(len(keypoints))
cv2.putText(blobs, text, (20, 550),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)


# Show blobs

cv2.imshow("Filtering Circular Blobs Only", blobs)
cv2.waitKey(0)
cv2.destroyAllWindows()
