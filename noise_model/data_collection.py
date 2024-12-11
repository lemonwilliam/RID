import os
import random
import numpy as np
from PIL import Image

# Define constants
BRIGHTNESS_THRESHOLD = 75  # Adjust this value based on your definition of "dark enough"
PATCHES_PER_IMAGE = 5
OUTPUT_FOLDER = "./data/input_patches/"

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def calculate_brightness(patch):
    """Calculate the average brightness of a patch."""
    # Convert to grayscale and calculate mean pixel value
    gray_patch = np.mean(patch, axis=2)  # Average the RGB channels
    return np.mean(gray_patch)

def extract_random_patch(image):
    """Extract a random 1080x720 patch from the given image."""
    width, height = image.size
    patch_width = width // 2
    patch_height = height // 2

    x = random.randint(0, width - patch_width)
    y = random.randint(0, height - patch_height)
    patch = image.crop((x, y, x + patch_width, y + patch_height))
    return patch

def flip_patch_randomly(patch):
    """Randomly flip the patch horizontally or vertically."""
    flip_type = random.choice(["horizontal", "vertical", None])
    if flip_type == "horizontal":
        return patch.transpose(Image.FLIP_LEFT_RIGHT)
    elif flip_type == "vertical":
        return patch.transpose(Image.FLIP_TOP_BOTTOM)
    return patch

def process_images(input_folder):
    """Process all images in the input folder."""
    for i, filename in enumerate(os.listdir(input_folder)):
        if filename.lower().endswith(('.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path)
            
            patches_extracted = 0
            attempts = 0
            
            while patches_extracted < PATCHES_PER_IMAGE and attempts < 100:
                attempts += 1
                patch = extract_random_patch(image)
                brightness = calculate_brightness(np.array(patch))

                if brightness < BRIGHTNESS_THRESHOLD:
                    patch = flip_patch_randomly(patch)
                    output_path = os.path.join(OUTPUT_FOLDER, f"{i*5 + patches_extracted + 1}.jpg")
                    patch.save(output_path)
                    patches_extracted += 1

            if patches_extracted < PATCHES_PER_IMAGE:
                print(f"Warning: Only extracted {patches_extracted} patches from {filename}.")

if __name__ == "__main__":
    input_folder = "./data/input"  # Replace with your folder path
    process_images(input_folder)
    print("Processing completed.")
