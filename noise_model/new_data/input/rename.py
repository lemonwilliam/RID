import os

def rename_images(start_number=100):
    # Get the current working directory
    current_dir = os.getcwd()

    # Filter files in the directory for image formats
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    images = [file for file in os.listdir(current_dir) if os.path.splitext(file)[1].lower() in image_extensions]

    # Sort images for consistent renaming order
    images.sort()

    # Rename images
    for i, image in enumerate(images):
        # Define the new filename
        new_name = f"{start_number + i}.jpg"

        # Rename the file
        os.rename(os.path.join(current_dir, image), os.path.join(current_dir, new_name))
        print(f"Renamed: {image} -> {new_name}")

if __name__ == "__main__":
    rename_images()
