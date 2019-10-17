# USAGE
# python find_contour.py -i ../data/NDVI_2.jpg

# change image to binary
# get contour
# find boundary - paddy field frame
# mask

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

# construct argument parse and parse command line
ap =  argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load image and convert to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (9, 9), 0)
edgeImg = detect_edge(blurred)


# binary
(t, binary) = cv2.threshold(edgeImg, 20, 255, cv2.THRESH_BINARY_INV)
# cv2.imshow("Binary", binary)

# dilate
dilated = binary.copy()
for i in range(0, 3):
	cv2.dilate(dilated, None, iterations = i + 1)

# # morphological gradient - to get the outside boundary
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# opening = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel)

# cv2.imshow("Opening", opening)
# cv2.imshow("Edge", edgeImg)
# cv2.imshow("Dilated", dilated)
# cv2.waitKey(0)

cnts = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
# cv2.drawContours(image, cnts, -1, (255, 0, 0), 2)
print("1st contour: {}".format(cnts[0].shape))

mask_boundaries = []

for (i, c) in enumerate(cnts):
	# print("Drawing contour #{}".format(i + 1))
	cv2.drawContours(image, [c], -1, (255, 0, 0), 2)
	# print("3rd contour: {}".format(c[0]))
	area = cv2.contourArea(c)
	if area > 300000.0 and area < 500000.0:
		# cv2.drawContours(image, [c], -1, (255, 0, 0), 2)
		# print("Drawing contour #{}".format(i + 1))
		# print("Area {}".format(area))
		mask_boundaries.append(c)
		# print(c.dtype)

# cv2.imshow("Contours", image)
# cv2.waitKey(0)

# mask
mask = np.zeros(image.shape[:2], dtype="uint8")
cv2.fillPoly(mask, [mask_boundaries[0]], 255)
masked = cv2.bitwise_and(image, image, mask=mask)

# cv2.imshow("Masked", masked)
# cv2.waitKey(0)

# cv2.imwrite("NDVI.jpg", masked)

# now we have our POI, get to work
gray_masked = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
blurred_masked = cv2.GaussianBlur(gray_masked, (3, 3), 0)

cont = cv2.findContours(blurred_masked, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cont = imutils.grab_contours(cont)
print("2nd contour: {}".format(cont[0].shape))

cv2.drawContours(image, cont, -1, (255, 0, 0), 2)

cv2.imshow("Window", masked)
cv2.waitKey(0)
cv2.imwrite("result.jpg", masked)


# significant_contour = get_significant_contours(edgeImg)
# print("Found {} contours".format(len(significant_contour)))

# # get smallest contour - the actual paddy field we're interested in
# # cv2.drawContours(image, significant_contour[0], -1, (255, 0, 0), 2)

# # contour smoothing
# # epsilon = 0.1 * cv2.arcLength(significant_contour[0], True)
# # approx = cv2.approxPolyDP(significant_contour[0], epsilon, True)
# # significant_contour[0] = approx

# # for i in significant_contour:
# # 	print(i[0].shape)

# # dilate
# dilated = significant_contour[0][0].copy().astype(np.uint8)
# dilated = cv2.GaussianBlur(dilated, (9, 9), 0)
# # cv2.dilate(dilated, None, 3)
# # dilated = dilated[:,0]
# # print(a.dtype)
# # print(image.dtype)
# print("Dilated :{}".format(dilated.shape))
# print("Image: {}".format(image.shape))
# print(gray.shape)

# ne = []
# for i in dilated:
# 	# print(i[0])
# 	ne.append(i[0])

# # print(len(ne))

# # print(dilated[0][0].shape)
# # print(dilated[0])
# # mask
# mask = edgeImg.copy()
# mask[mask > 0] = 0
# # cv2.rectangle(mask, (2,4), (150, 200), (255, 255,255), -1)
# cv2.fillConvexPoly(mask, np.int32(ne), 255)
# mask = np.logical_not(mask)
# image[mask] = 0


# # show output
# cv2.imshow("All Contours", image)
# cv2.waitKey(0)

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
