# USAGE
# python find_contour.py -i ../data/NDVI_2.jpg

# import packages
import numpy as np
import argparse
import cv2
import imutils

# references
# https://www.learnopencv.com/applications-of-foreground-background-separation-with-semantic-segmentation/
# https://www.codepasta.com/computer-vision/2016/11/06/background-segmentation-removal-with-opencv.html

def detect_edge(channel):
	sobelX = cv2.Sobel(channel, cv2.CV_64F, dx=1, dy=0)
	sobelY = cv2.Sobel(channel, cv2.CV_64F, dx=0, dy=1)
	
	sobelX = cv2.convertScaleAbs(sobelX)
	sobelY = cv2.convertScaleAbs(sobelY)

	sobel = cv2.addWeighted(sobelX, 0.5, sobelY, 0.5, 0)
	return sobel

def get_significant_contours(edgeImg):
	cnts = cv2.findContours(edgeImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	too_small = edgeImg.size * 0.1
	significants = []
	# print("C :{}".format(type(cnts)))
	for (i, c) in enumerate(cnts):
		area = cv2.contourArea(c)
		# print("Area {}".format(area))
		# print("shape: {}".format(c.shape))
		# print("C: {}".format(c))
		if area > too_small:
			significants.append([c, area])

		significants.sort(key=lambda r: r[1])
				
	# return [x[0] for x in significants]
	return significants

# construct argument parse and parse command line
ap =  argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load image and convert to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (9, 9), 0)
edgeImg = detect_edge(blurred)
# edgeImg = detect_edge(gray)



significant_contour = get_significant_contours(edgeImg)
print("Found {} contours".format(len(significant_contour)))

# get smallest contour - the actual paddy field we're interested in
# cv2.drawContours(image, significant_contour[0], -1, (255, 0, 0), 2)

# contour smoothing
# epsilon = 0.1 * cv2.arcLength(significant_contour[0], True)
# approx = cv2.approxPolyDP(significant_contour[0], epsilon, True)
# significant_contour[0] = approx

# for i in significant_contour:
# 	print(i[0].shape)

# dilate
dilated = significant_contour[0][0].copy().astype(np.uint8)
dilated = cv2.GaussianBlur(dilated, (9, 9), 0)
# cv2.dilate(dilated, None, 3)
# dilated = dilated[:,0]
# print(a.dtype)
# print(image.dtype)
print("Dilated :{}".format(dilated.shape))
print("Image: {}".format(image.shape))
print(gray.shape)

ne = []
for i in dilated:
	# print(i[0])
	ne.append(i[0])

# print(len(ne))

# print(dilated[0][0].shape)
# print(dilated[0])
# mask
mask = edgeImg.copy()
mask[mask > 0] = 0
# cv2.rectangle(mask, (2,4), (150, 200), (255, 255,255), -1)
cv2.fillConvexPoly(mask, np.int32(ne), 255)
mask = np.logical_not(mask)
image[mask] = 0


# show output
cv2.imshow("All Contours", image)
cv2.waitKey(0)

# # first parameter = image to threshold
# # second parameter = threshold value
# # third parameter = value if greater than threshold
# (T, thresh) = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
# processed = cv2.bitwise_and(gray, gray, mask=thresh)
# # cv2.imshow("Processed", processed)

# # find all contours
# # cnts = cv2.findContours(gray.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
# clone = image.copy()
# cv2.drawContours(clone, cnts, -1, (255, 0, 0), 2)
# print("Found {} contours".format(len(cnts)))

# for (i, c) in enumerate(cnts):
# 	area = cv2.contourArea(c)
# 	print("Area {}".format(area))

# # show output
# cv2.imshow("All Contours", clone)
# cv2.waitKey(0)
