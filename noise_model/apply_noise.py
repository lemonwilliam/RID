import numpy as np
import cv2
import argparse
from scipy.interpolate import interp1d


def demosaic(raw_img):
    """
    Apply demosaicing to a RAW image to reconstruct an RGB image.
    Assumes Bayer pattern (BGGR).
    """
    rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_BAYER_BG2BGR)
    return rgb_img


def reverse_demosaic(img):
    """
    Reverse demosaicing: Converts an RGB image to a RAW Bayer pattern.
    Assumes Bayer pattern (BGGR).
    """
    # Convert to grayscale as an approximation for RAW Bayer
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img


def choose_crf(crf_file_lines):

    num_crfs = len(crf_file_lines) // 6
    
    # Randomly select a CRF
    selected_crf_idx = np.random.randint(0, num_crfs - 1)
    start_line = selected_crf_idx * 6
    crf_name = crf_file_lines[start_line].strip()  # First line: CRF name
    crf_type = crf_file_lines[start_line + 1].strip()  # Second line: CRF type
    
    # Extract I= and B= sequences
    i_values = np.array([float(x) for x in crf_file_lines[start_line + 3].strip().split()])
    b_values = np.array([float(x) for x in crf_file_lines[start_line + 5].strip().split()])
    
    print(f"Selected CRF: {crf_name} ({crf_type})")
    
    # Create an interpolation function for the CRF
    crf = interp1d(i_values, b_values, kind='linear', bounds_error=False, fill_value="extrapolate")
    inverse_crf = interp1d(b_values, i_values, kind='linear', bounds_error=False, fill_value="extrapolate")

    return crf, inverse_crf


def apply_crf(img, crf):
    """
    Apply the camera response function (CRF) to an image.
    """
    img_normalized = img / 255.0  # Normalize pixel values to [0, 1]
    img_transformed = np.clip(crf(img_normalized), 0, 1)  # Apply CRF and clip to [0, 1]
    img_output = (img_transformed * 255).astype(np.uint8)  # Convert back to uint8
    return img_output


def reverse_crf(img, inverse_crf):
    """
    Apply the inverse camera response function (CRF) to an image.
    """
    img_normalized = img / 255.0  # Normalize pixel values to [0, 1]
    img_transformed = np.clip(inverse_crf(img_normalized), 0, 1)  # Apply inverse CRF and clip to [0, 1]
    img_output = (img_transformed * 255).astype(np.uint8)  # Convert back to uint8
    return img_output


def H_Gaussian_noise(raw_img, sigma_read=0.01, sigma_shot=0.01):
    """
    Add heteroscedastic Gaussian noise to a RAW image.
    - Sigma_read: Read noise (constant Gaussian noise).
    - Sigma_shot: Shot noise (proportional to signal intensity).
    """
    # Scale the image to [0, 1] for noise application
    raw_normalized = raw_img / 255.0
    read_noise = np.random.normal(0, sigma_read, raw_normalized.shape)
    shot_noise = np.random.normal(0, sigma_shot * raw_normalized, raw_normalized.shape)
    noisy_img = raw_normalized + read_noise + shot_noise
    noisy_img = np.clip(noisy_img, 0, 1)  # Clip to valid range
    return (noisy_img * 255).astype(np.uint8)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', '-i', type=str)
    parser.add_argument('--output_path', '-o', type=str)
    args = parser.parse_args()

    # Open the file and process its lines
    file_path = "./dorfCurves.txt"
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for i in range(0, len(lines), 6):  
        print(lines[i].strip())

    crf, crf_inv = choose_crf(lines)

    # Read original RGB image
    original_img = cv2.imread(args.input_path, cv2.IMREAD_COLOR)

    # Reverse camera response function
    crf_reversed_img = reverse_crf(original_img, crf_inv)

    # Reverse demosaicing
    raw_img = reverse_demosaic(crf_reversed_img)

    # Apply heteroscedastic Gaussian noise
    noisy_raw_img = H_Gaussian_noise(raw_img)

    # Re-apply demosaicing
    demosaic_img = demosaic(noisy_raw_img)

    # Re-apply camera response function
    noisy_img = apply_crf(demosaic_img, crf)

    cv2.imwrite(args.output_path, noisy_img)
    