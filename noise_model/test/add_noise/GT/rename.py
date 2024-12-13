import os

def rename_images(start_number=1000):
    # Get the current folder where the script is located
    current_folder = os.path.dirname(os.path.abspath(__file__))
    # Define valid image extensions
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    # List all files in the folder
    files = os.listdir(current_folder)
    # Filter files with valid image extensions
    image_files = [f for f in files if os.path.splitext(f)[1].lower() in valid_extensions]
    # Sort the files to ensure consistent renaming order
    image_files.sort()

    # Rename images sequentially
    for i, filename in enumerate(image_files, start=start_number):
        old_path = os.path.join(current_folder, filename)
        new_filename = f"{i}{os.path.splitext(filename)[1].lower()}"
        new_path = os.path.join(current_folder, new_filename)
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_filename}")

if __name__ == "__main__":
    rename_images()
