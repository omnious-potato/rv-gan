import os
import random
from PIL import Image
import shutil

# Define paths
input_dir = "E:\\DL\\test"
output_dir = "E:\\DL\\test\\processed"
os.makedirs(output_dir, exist_ok=True)

# Target size
TARGET_SIZE = 512

def process_image(image_path, output_path):
    try:
        with Image.open(image_path) as img:
            width, height = img.size

            if width == TARGET_SIZE and height == TARGET_SIZE:
                # Direct copy if exactly 512x512
                shutil.copy(image_path, output_path)
            elif width >= TARGET_SIZE and height >= TARGET_SIZE:
                # Random crop
                left = random.randint(0, width - TARGET_SIZE)
                top = random.randint(0, height - TARGET_SIZE)
                right = left + TARGET_SIZE
                bottom = top + TARGET_SIZE

                cropped_img = img.crop((left, top, right, bottom))
                cropped_img.save(output_path)
            else:
                print(f"Skipped (too small): {image_path}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# Process all images in the input directory
for filename in os.listdir(input_dir):
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        process_image(input_path, output_path)
