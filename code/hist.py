# USAGE
# python hist.py -i ../data/NDVI_2.jpg

# import packages
import numpy as np
import argparse
import cv2
import imutils
from matplotlib import pyplot as plt

# construct argument parse and parse command line
ap =  argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load image and convert to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# cv2.imshow("Original", image)
# cv2.imshow("Grayed", gray)
hist = cv2.calcHist([gray], [0], None, [256], [0,256])
plt.figure()
plt.xlabel("Bins")
plt.ylabel("% of Pixels")
plt.xlim([0,256])
plt.plot(hist)
plt.show()

# find all contours
# cnts = cv2.findContours(gray.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
clone = image.copy()
cv2.drawContours(clone, cnts, -1, (0, 255, 0), 2)
print("Found {} contours".format(len(cnts)))

# show output
cv2.imshow("All Contours", clone)
cv2.waitKey(0)
