# https://stackoverflow.com/a/60066530

import cv2
import numpy as np

# Load image, grayscale, Gaussian blur, adaptive threshold
image = cv2.imread('../img/corner_detect_example.png')
mask = np.zeros(image.shape, dtype=np.uint8)
gray = 255 - cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3, 3), 0)
thresh = cv2.adaptiveThreshold(
    blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 3)

# Morph open
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

# Find distorted rectangle contour and draw onto a mask
cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
rect = cv2.minAreaRect(cnts[0])
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(image, [box], 0, (36, 255, 12), 2)
cv2.fillPoly(mask, [box], (255, 255, 255))

# Find corners on the mask
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
corners = cv2.goodFeaturesToTrack(
    mask, maxCorners=4, qualityLevel=0.5, minDistance=150)

# for corner in corners:
#     x, y = corner.ravel()
#     cv2.circle(image, (int(x), int(y)), 8, (255, 0, 0), -1)
#     print('({}, {})'.format(x,y))

windows = ['image', 'thresh', 'opening', 'mask']

for window in windows:
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, (600, 600))

cv2.imshow('thresh', thresh)
cv2.imshow('opening', opening)
cv2.imshow('mask', mask)
cv2.imshow('image', image)
cv2.waitKey(0)
