#adapted from https://pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/
import numpy as np
import argparse
import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pywt
import pandas as pd
#from cv2_rolling_ball import subtract_background_rolling_ball
from skimage import (
    data, restoration, util
)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

# load the image, clone it for output, and then convert it to grayscale
image = cv2.imread(args["image"])
output = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

"""
strategy 1: gaussian denoising and rolling ball background subtraction.
pros: reduce high frequency noise and general messy backgrounds
cons: rolling ball slows things down
"""

#gaussian blur is necessary to get rid of high frequency, noisy components in image

inter = cv2.GaussianBlur(gray, (5, 5), cv2.BORDER_DEFAULT);

"""
#rolling ball background subtraction

background = restoration.rolling_ball(inter)
img = inter - background
#img, background = subtract_background_rolling_ball(inter, 30, light_background=True,use_paraboloid=False, do_presmooth=True)
#opencv rolling ball is too slow!
"""

"""
strategy 2: gaussian pyramid downsampling plus background subtraction

ds = cv2.pyrDown(gray)
print("Size of image after pyrDown: ", ds.shape)

image_inverted = util.invert(ds)

background_inverted = restoration.rolling_ball(image_inverted, radius=15)
filtered_image_inverted = image_inverted - background_inverted
filtered_image = util.invert(filtered_image_inverted)
background = util.invert(background_inverted)
output = filtered_image.copy()

#background = restoration.rolling_ball(ds)
#img = ds - background
"""


# detect circles in the image
#circles = cv2.HoughCircles(inter, cv2.HOUGH_GRADIENT,1.2,60,param1=80,param2=50,minRadius=20,maxRadius=35)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 60, len(gray)/4, 200, 20, maxRadius=150)
# ensure at least some circles were found
if circles is not None:
	# convert the (x, y) coordinates and radius of the circles to integers
	circles = np.round(circles[0, :]).astype("int")
	# loop over the (x, y) coordinates and radius of the circles
	for (x, y, r) in circles:
		# draw the circle in the output image, then draw a rectangle
		# corresponding to the center of the circle
		cv2.circle(output, (x, y), r, (0, 255, 0), 4)
		#d = [gray[x,y]]#eventually try to draw a line profile through the diameter of the cirlce; record the intensities
		cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
	# show the output image
	print(len(circles)) #we're going to want to save this data in .csv format. better load up pandas!
	cv2.imshow("output", np.hstack([image, output]))
	cv2.waitKey(0)