import os
import cv2

def split_images_into_patches(input_folder, output_folder):
    """
    Splits each image in the input folder into four equally-sized patches and saves them to the output folder.

    Args:
        input_folder (str): Path to the folder containing the original images.
        output_folder (str): Path to the folder where patches will be saved.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        
        # Read the image
        img = cv2.imread(file_path)
        if img is None:
            print(f"Skipping file (not an image): {filename}")
            continue
        
        # Get image dimensions
        height, width = img.shape[:2]
        
        # Compute the midpoint for width and height
        mid_height = height // 2
        mid_width = width // 2
        
        # Define patches
        patches = {
            "top_left": img[:mid_height, :mid_width],
            "top_right": img[:mid_height, mid_width:],
            "bottom_left": img[mid_height:, :mid_width],
            "bottom_right": img[mid_height:, mid_width:]
        }
        
        # Save each patch
        for patch_name, patch in patches.items():
            patch_filename = f"{os.path.splitext(filename)[0]}_{patch_name}.jpg"
            patch_path = os.path.join(output_folder, patch_filename)
            cv2.imwrite(patch_path, patch)
            print(f"Saved patch: {patch_path}")

# Example usage
input_folder = "./data/input_resized"  # Replace with the path to your folder of images
output_folder = "./data/input_patches"  # Replace with the path to save patches

split_images_into_patches(input_folder, output_folder)
