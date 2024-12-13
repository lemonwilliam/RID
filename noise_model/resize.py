import os
import cv2

def resize_images_in_folder(input_folder, output_folder, target_pixel_count):
    """
    Resizes each image in the folder so that each image has approximately the same number of pixels,
    while maintaining its original aspect ratio. Images with a smaller pixel count than the target
    will be left unchanged.

    Args:
        input_folder (str): Path to the folder containing the images.
        output_folder (str): Path to the folder where resized images will be saved.
        target_pixel_count (int): Desired number of pixels for each image (width * height).
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate over each file in the input folder
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if os.path.isfile(file_path):
            # Read the image
            img = cv2.imread(file_path)
            if img is None:
                print(f"Skipping file (cannot read): {filename}")
                continue
            
            # Get original image dimensions
            original_height, original_width = img.shape[:2]
            original_pixel_count = original_height * original_width
            
            # If original pixel count is smaller than the target, leave the image unchanged
            if original_pixel_count < target_pixel_count:
                # Save the original image without resizing
                output_file_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_file_path, img)
                print(f"Saved unchanged: {filename}")
                continue
            
            # Calculate scaling factor to achieve target_pixel_count while maintaining aspect ratio
            scaling_factor = (target_pixel_count / original_pixel_count) ** 0.5
            new_width = int(original_width * scaling_factor)
            new_height = int(original_height * scaling_factor)
            
            # Resize the image
            resized_img = cv2.resize(img, (new_width, new_height))
            
            # Save the resized image
            output_file_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_file_path, resized_img)
            print(f"Resized and saved: {filename}")
            
# Example usage
input_folder = "./data/hello"
output_folder = "./data/hello2"
target_pixel_count = 3000000  # Adjust this value based on desired pixel count

resize_images_in_folder(input_folder, output_folder, target_pixel_count)
