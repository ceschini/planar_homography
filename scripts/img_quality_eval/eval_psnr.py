# We should implement PSNR and LPIPS
# PSNR: https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/

from math import log10, sqrt
import cv2
import numpy as np


def psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):
        # MSE is zero means no noise is present in tl
        # therefore psnr have no importance
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def main():
    original = cv2.imread("../../img/eval_original_image.png")
    compressed = cv2.imread("../../img/eval_compressed_image1.png")
    value = psnr(original, compressed)
    print(f'PSNR value is {value} dB')


if __name__ == "__main__":
    main()
