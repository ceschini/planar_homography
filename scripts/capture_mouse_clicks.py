# https://www.tutorialspoint.com/opencv-python-how-to-display-the-coordinates-of-points-clicked-on-an-image

import cv2
import numpy as np

coords = []
blank = np.zeros((1600, 1024, 3), np.uint8)


def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        # print(f'({x}, {y})')
        coords.append([x, y])
        print(coords)
        # draw point on the image
        cv2.circle(img, (x, y), 25, (0, 255, 255), -1)


# read the input image
img = cv2.imread('../img/foto1_cap1.jpg')

# create a window
cv2.namedWindow('point coords', cv2.WINDOW_NORMAL)
cv2.resizeWindow('point coords', (1600, 1024))

# bind the callback function to window
cv2.setMouseCallback('point coords', click_event)


# display image
while True:
    cv2.imshow('point coords', img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

if len(coords) == 4:
    print('entrou')
    pts = np.array(coords, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], True, (0, 255, 255), thickness=10)

cv2.namedWindow('result', cv2.WINDOW_NORMAL)
cv2.resizeWindow('result', (1600, 1024))
cv2.imshow('result', img)
cv2.waitKey(0)

cv2.destroyAllWindows()
