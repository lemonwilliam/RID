import numpy as np
import cv2
import argparse
from scipy.interpolate import interp1d
from collections import defaultdict


def filter_and_group_crfs(input_file):
    """
    Filters and groups CRFs by their base name and color channel.
    
    Args:
    - input_file (str): Path to the input text file with CRFs.
    
    Returns:
    - grouped_crfs (dict): A dictionary with base names as keys and a dictionary of CRF types as values.
    """
    keywords = ['Red', 'Green', 'Blue']
    grouped_crfs = defaultdict(dict)

    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Process CRFs in groups of 6 lines
    for i in range(0, len(lines), 6):
        crf_name = lines[i].strip()  # CRF name is on the first line of each group
        crf_type = lines[i + 1].strip()  # CRF type (not used for grouping)
        i_values = np.array([float(x) for x in lines[i + 3].strip().split()])
        b_values = np.array([float(x) for x in lines[i + 5].strip().split()])

        # Check which keyword is present
        for keyword in keywords:
            if keyword in crf_name:
                # Extract base name by removing the keyword
                base_name = crf_name.replace(keyword, "")
                grouped_crfs[base_name][keyword + '_I'] = i_values
                grouped_crfs[base_name][keyword + '_B'] = b_values


    return grouped_crfs


def apply_crf(img, crf_group):
    """
    Apply the camera response function (CRF) to an image.
    """
    transformed_img = np.zeros_like(img, dtype=np.float32)

    # Create interpolation functions for the CRFs
    crf_red = interp1d(crf_group['Red_I'], crf_group['Red_B'], kind='linear', bounds_error=False, fill_value="extrapolate")
    crf_green = interp1d(crf_group['Green_I'], crf_group['Green_B'], kind='linear', bounds_error=False, fill_value="extrapolate")
    crf_blue = interp1d(crf_group['Blue_I'], crf_group['Blue_B'], kind='linear', bounds_error=False, fill_value="extrapolate")

    # Apply CRFs to respective channels
    transformed_img[:, :, 2] = crf_red(img[:, :, 2] / 255.0)  # Red channel
    transformed_img[:, :, 1] = crf_green(img[:, :, 1] / 255.0)  # Green channel
    transformed_img[:, :, 0] = crf_blue(img[:, :, 0] / 255.0)  # Blue channel
    
    # Scale back to 0-255 range
    transformed_img = np.clip(transformed_img * 255, 0, 255).astype(np.uint8)
    return transformed_img


def reverse_crf(img, crf_group):
    """
    Apply the inverse camera response function (CRF) to an image.
    """
    transformed_img = np.zeros_like(img, dtype=np.float32)

    # Create interpolation functions for the inverse CRFs
    crf_red_inv = interp1d(crf_group['Red_B'], crf_group['Red_I'], kind='linear', bounds_error=False, fill_value="extrapolate")
    crf_green_inv = interp1d(crf_group['Green_B'], crf_group['Green_I'], kind='linear', bounds_error=False, fill_value="extrapolate")
    crf_blue_inv = interp1d(crf_group['Blue_B'], crf_group['Blue_I'], kind='linear', bounds_error=False, fill_value="extrapolate")

    # Apply CRFs to respective channels
    transformed_img[:, :, 2] = crf_red_inv(img[:, :, 2] / 255.0)  # Red channel
    transformed_img[:, :, 1] = crf_green_inv(img[:, :, 1] / 255.0)  # Green channel
    transformed_img[:, :, 0] = crf_blue_inv(img[:, :, 0] / 255.0)  # Blue channel
    
    # Scale back to normal range
    transformed_img = (transformed_img * 255).astype(np.uint8)
    return transformed_img


def demosaic(raw_img):
    """
    Apply demosaicing to a RAW image to reconstruct an RGB image.
    Assumes Bayer pattern (RGGB).
    """
    rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_BAYER_RG2RGB)
    return rgb_img


def reverse_demosaic(img):
    """
    Reverse demosaicing: Converts an RGB image to a RAW Bayer pattern.
    Assumes Bayer pattern (RGGB).
    """
    # Create an empty Bayer RAW image
    raw_img = np.zeros_like(img[:, :, 0])  # RAW Bayer is a single-channel image

    # Extract color channels
    r_channel = img[0::2, 0::2, 2]  # Red: top-left pixels (even rows, even cols)
    g_channel_1 = img[0::2, 1::2, 1]  # Green (1): top-right pixels (even rows, odd cols)
    g_channel_2 = img[1::2, 0::2, 1]  # Green (2): bottom-left pixels (odd rows, even cols)
    b_channel = img[1::2, 1::2, 0]  # Blue: bottom-right pixels (odd rows, odd cols)

    # Fill the Bayer RAW image
    raw_img[0::2, 0::2] = r_channel  # Top-left
    raw_img[0::2, 1::2] = g_channel_1  # Top-right
    raw_img[1::2, 0::2] = g_channel_2  # Bottom-left
    raw_img[1::2, 1::2] = b_channel  # Bottom-right

    return raw_img


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
    grouped_crfs = filter_and_group_crfs('dorfCurves_filtered.txt')

    # Select a random CRF group
    base_name = np.random.choice(list(grouped_crfs.keys()))
    selected_crf_group = grouped_crfs[base_name]
    print(f"Selected CRF group: {base_name}")

    # Read original RGB image
    original_img = cv2.imread(args.input_path, cv2.IMREAD_COLOR)

    # Reverse camera response function
    crf_reversed_img = reverse_crf(original_img, selected_crf_group)
    cv2.imshow("img", crf_reversed_img)
    cv2.waitKey(0)

    # Reverse demosaicing
    raw_img = reverse_demosaic(crf_reversed_img)
    cv2.imshow("img", raw_img)
    cv2.waitKey(0)

    # Apply heteroscedastic Gaussian noise
    noisy_raw_img = H_Gaussian_noise(raw_img)

    # Re-apply demosaicing
    demosaic_img = demosaic(noisy_raw_img)

    # Re-apply camera response function
    noisy_img = apply_crf(demosaic_img, selected_crf_group)

    cv2.imwrite(args.output_path, noisy_img)
    