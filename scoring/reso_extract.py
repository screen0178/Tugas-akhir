import os
from PIL import Image

def extract_image_info(directory):
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
    image_info = []

    for filename in os.listdir(directory):
        if filename.lower().endswith(supported_formats):
            filepath = os.path.join(directory, filename)
            try:
                with Image.open(filepath) as img:
                    width, height = img.size
                    image_info.append((filename, width, height))
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return image_info

# Example usage:
directory_path = "."
info = extract_image_info(directory_path)

# Print the result
for name, width, height in info:
    print(f"{name}: {width}x{height}")
