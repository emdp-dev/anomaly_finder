# USAGE
# python workit.py -i ../data/NDVI.jpg


# import packages
import numpy as np
import argparse
import cv2
import imutils

# construct argument parse and parse command line
ap =  argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load image and convert to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)
# binary
(t, binary) = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)

# # dilate
# dilated = binary.copy()
# for i in range(0, 3):
# 	cv2.dilate(dilated, None, iterations = i + 1)

cnts = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
clone = image.copy()

cv2.drawContours(clone, cnts, -1, (255, 0, 0), 2)
print("Found {} contours".format(len(cnts)))

# cv2.imshow("blurred", blurred)
cv2.imshow("Contour", clone)
cv2.waitKey(0)


# # cv2.drawContours(image, cnts, -1, (255, 0, 0), 2)
# print("Found {} contours".format(len(cnts)))

# for (i, c) in enumerate(cnts):
# 	cv2.drawContours(image, [c], -1, (255, 0, 0), 2)
# 	print("contour: {}".format(c[0]))
# 	area = cv2.contourArea(c)
	

# cv2.imshow("Contours", image)
# cv2.waitKey(0)