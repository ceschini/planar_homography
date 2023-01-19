# https://www.tutorialspoint.com/opencv-python-how-to-display-the-coordinates-of-points-clicked-on-an-image
# https://learnopencv.com/automatic-document-scanner-using-opencv/

import cv2
import numpy as np
from img_quality_eval.eval_lpips import lpips_eval
from img_quality_eval.eval_psnr import psnr_eval

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        # print(f'({x}, {y})')
        coords.append([x, y])
        # print(coords)
        # draw point on the image
        cv2.circle(img, (x, y), 10, (0, 0, 255), -1)


# read the input and ground truth image
img = cv2.imread('../img/foto1_cap1.jpg')
gt = cv2.imread('../img/foto1_gt.jpg')
height, width = img.shape[:2]
gt = cv2.resize(gt, (width, height))

# create blank image
blank = np.zeros((height, width, 3), np.uint8)
blank[:] = (255, 255, 255)

# create a window
cv2.namedWindow('Select corners', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Select corners', (1600, 1024))
cv2.resizeWindow('Select corners', (width, height))

# bind the callback function to window
cv2.setMouseCallback('Select corners', click_event)

# create list of coordinates
coords = []

# display image
while len(coords) < 4:
    cv2.imshow('Select corners', img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

# corresponding coordinates in final image
# output_coords = [[0, 0], [800, 0], [800, 600], [0, 600]]
output_coords = [[0, 0], [width, 0], [width, height], [0, height]]

# generate homography matrix
H, mask = cv2.findHomography(
    np.array(coords, np.float32), np.array(output_coords, np.float32))

# warp input image into the final image
# output_image = cv2.warpPerspective(img, H, (800, 600))
output_image = cv2.warpPerspective(img, H, (width, height))

# draw selected points and homography on blank image
for point in coords:
    cv2.circle(blank, tuple(point), 5, (0, 0, 255), -1)
cv2.polylines(blank, [np.array(coords, np.int32).reshape(
    (-1, 1, 2))], True, (0, 0, 255), thickness=2)
cv2.polylines(blank, [np.array(output_coords, np.int32).reshape(
    (-1, 1, 2))], True, (255, 0, 0), thickness=2)


# print(type(output_image))
# print(type(gt))

# print(f'img shape: {img.shape}')
# print(f'output shape: {output_image.shape}')
# print(f'gt shape: {gt.shape}')

# display the output image and the ground truth
# cv2.imshow('Homography', blank)
cv2.namedWindow('Retified and cropped image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Retified and cropped image', (width, height))
cv2.imshow('Retified and cropped image', output_image)

cv2.namedWindow('ground truth', cv2.WINDOW_NORMAL)
cv2.resizeWindow('ground truth', (width, height))
cv2.imshow('ground truth', gt)

# evaluation metrics
psnr = psnr_eval(gt, output_image)
print(f'PSNR value is {psnr} dB')


lpips_distance = lpips_eval(gt, output_image)
print(f'LPIPS distance is: {lpips_distance}')

cv2.waitKey(0)

cv2.destroyAllWindows()
