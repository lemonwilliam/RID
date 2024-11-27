import numpy as np
import cv2
import argparse

def demosaic():
    return

def reverse_demosaic():
    return

def crf():
    return

def reverse_crf(img):
    return

def H_Gaussian_noise():
    return

def apply_realistic_noise(img):
    noisy_img = img
    return noisy_img

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', '-i', type=str, required=True)
    parser.add_argument('--output_path', '-o', type=str, required=True)
    args = parser.parse_args()

    # Read original image
    original_img = cv2.imread(args.input_path, cv2.IMREAD_COLOR)

    # Reverse camera response function
    crf_reversed_img = reverse_crf(original_img)

    # Reverse demosaicing
    raw_img = reverse_demosaic(crf_reversed_img)

    # Apply heteroscedastic Gaussian noise
    noisy_raw_img = H_Gaussian_noise(raw_img)

    # Re-apply demosaicing
    demosaic_img = demosaic(noisy_raw_img)

    # Re-apply camera response function
    noisy_img = crf(demosaic_img)

    cv2.imwrite(args.output_path, noisy_img)