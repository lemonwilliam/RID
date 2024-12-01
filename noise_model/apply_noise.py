import numpy as np
import cv2
import os
import argparse
from scipy.interpolate import interp1d
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

    try:
        with open(input_file, 'r') as file:
            lines = file.readlines()

        # Process CRFs in groups of 6 lines
        for i in range(0, len(lines), 6):
            crf_name = lines[i].strip()  # CRF name is on the first line of each group
            i_values = np.array([float(x) for x in lines[i + 3].strip().split()])
            b_values = np.array([float(x) for x in lines[i + 5].strip().split()])

            # Check which keyword is present
            for keyword in keywords:
                if keyword in crf_name:
                    # Extract base name by removing the keyword
                    base_name = crf_name.replace(keyword, "")
                    grouped_crfs[base_name][keyword] = {'I': i_values, 'B': b_values}

    except FileNotFoundError:
        logging.error(f"CRF file not found: {input_file}")
        raise
    except Exception as e:
        logging.error(f"Error processing CRF file: {e}")
        raise

    return grouped_crfs

def apply_crf(img, crf_group):
    """
    Apply the camera response function (CRF) to an RGB image.
    """
    # Ensure input is in the correct format (RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transformed_img = np.zeros_like(img_rgb, dtype=np.float32)

    # Create interpolation functions for the CRFs
    crf_fns = {
        'R': interp1d(crf_group['Red']['I'], crf_group['Red']['B'], 
                      kind='linear', bounds_error=False, fill_value="extrapolate"),
        'G': interp1d(crf_group['Green']['I'], crf_group['Green']['B'], 
                      kind='linear', bounds_error=False, fill_value="extrapolate"),
        'B': interp1d(crf_group['Blue']['I'], crf_group['Blue']['B'], 
                      kind='linear', bounds_error=False, fill_value="extrapolate")
    }

    # Normalize to 0-1 range
    img_norm = img_rgb / 255.0

    # Apply CRFs to respective channels
    for i, color in enumerate(['R', 'G', 'B']):
        transformed_img[:, :, i] = crf_fns[color](img_norm[:, :, i])
    
    # Scale back to 0-255 range and convert to BGR for OpenCV
    transformed_img = np.clip(transformed_img * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(transformed_img, cv2.COLOR_RGB2BGR)

def reverse_crf(img, crf_group):
    """
    Apply the inverse camera response function (CRF) to an RGB image.
    """
    # Ensure input is in the correct format (RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transformed_img = np.zeros_like(img_rgb, dtype=np.float32)

    # Create interpolation functions for the inverse CRFs
    crf_inv_fns = {
        'R': interp1d(crf_group['Red']['B'], crf_group['Red']['I'], 
                      kind='linear', bounds_error=False, fill_value="extrapolate"),
        'G': interp1d(crf_group['Green']['B'], crf_group['Green']['I'], 
                      kind='linear', bounds_error=False, fill_value="extrapolate"),
        'B': interp1d(crf_group['Blue']['B'], crf_group['Blue']['I'], 
                      kind='linear', bounds_error=False, fill_value="extrapolate")
    }

    # Normalize to 0-1 range
    img_norm = img_rgb / 255.0

    # Apply inverse CRFs to respective channels
    for i, color in enumerate(['R', 'G', 'B']):
        transformed_img[:, :, i] = crf_inv_fns[color](img_norm[:, :, i])
    
    # Scale back to 0-255 range and convert to BGR for OpenCV
    transformed_img = np.clip(transformed_img * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(transformed_img, cv2.COLOR_RGB2BGR)

def demosaic(raw_img):
    """
    Apply demosaicing to a RAW image to reconstruct an RGB image.
    Assumes Bayer pattern (RGGB).
    """
    return cv2.cvtColor(raw_img, cv2.COLOR_BAYER_BG2RGB)

def reverse_demosaic(img):
    """
    Reverse demosaicing: Converts an RGB image to a RAW Bayer pattern.
    Assumes Bayer pattern (RGGB).
    """
    # Convert to RGB if input is BGR
    if img.shape[2] == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img

    # Create an empty Bayer RAW image
    raw_img = np.zeros(img_rgb.shape[:2], dtype=np.uint8)

    # Extract color channels
    r_channel = img_rgb[0::2, 0::2, 0]  # Red: top-left pixels (even rows, even cols)
    g_channel_1 = img_rgb[0::2, 1::2, 1]  # Green (1): top-right pixels (even rows, odd cols)
    g_channel_2 = img_rgb[1::2, 0::2, 1]  # Green (2): bottom-left pixels (odd rows, even cols)
    b_channel = img_rgb[1::2, 1::2, 2]  # Blue: bottom-right pixels (odd rows, odd cols)

    # Fill the Bayer RAW image
    raw_img[0::2, 0::2] = r_channel  # Top-left
    raw_img[0::2, 1::2] = g_channel_1  # Top-right
    raw_img[1::2, 0::2] = g_channel_2  # Bottom-left
    raw_img[1::2, 1::2] = b_channel  # Bottom-right

    return raw_img

def add_heteroscedastic_noise(raw_img, sigma_read=0.01, sigma_shot=0.01):
    """
    Add heteroscedastic Gaussian noise to a RAW image.
    - Sigma_read: Read noise (constant Gaussian noise).
    - Sigma_shot: Shot noise (proportional to signal intensity).
    """
    raw_normalized = raw_img / 255.0
    read_noise = np.random.normal(0, sigma_read, raw_normalized.shape)
    shot_noise = np.random.normal(0, sigma_shot * raw_normalized, raw_normalized.shape)
    noisy_img = raw_normalized + read_noise + shot_noise
    noisy_img = np.clip(noisy_img, 0, 1)  # Clip to valid range
    return (noisy_img * 255).astype(np.uint8)

def main(input_path, output_path):
    """
    Main processing pipeline
    """
    # Open the file and process its lines
    grouped_crfs = filter_and_group_crfs('dorfCurves_filtered.txt')

    # Select a random CRF group
    base_name = np.random.choice(list(grouped_crfs.keys()))
    selected_crf_group = grouped_crfs[base_name]
    logging.info(f"Selected CRF group: {base_name}")

    # Read original BGR image
    original_img = cv2.imread(input_path)

    # Reverse camera response function
    crf_reversed_img = reverse_crf(original_img, selected_crf_group)

    # Reverse demosaicing
    raw_img = reverse_demosaic(crf_reversed_img)
    
    # Apply heteroscedastic Gaussian noise
    noisy_raw_img = add_heteroscedastic_noise(raw_img)

    # Re-apply demosaicing (now in RGB)
    demosaic_img = demosaic(noisy_raw_img)

    # Re-apply camera response function
    noisy_img = apply_crf(cv2.cvtColor(demosaic_img, cv2.COLOR_RGB2BGR), selected_crf_group)

    # Save the final image
    cv2.imwrite(output_path, noisy_img)
    logging.info(f"Processed image saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', '-i', type=str, default='./data/input/')
    parser.add_argument('--output_folder', '-o', type=str, default='./data/output/')
    parser.add_argument('--num_image', '-n', type=int, default=1)
    args = parser.parse_args()

    input_path = os.path.join(args.input_folder, f"{args.num_image}.png")
    output_path = os.path.join(args.output_folder, f"{args.num_image}.png")

    main(input_path, output_path)