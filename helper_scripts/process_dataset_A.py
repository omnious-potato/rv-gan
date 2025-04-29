import os
import cv2
import random
import glob
from tqdm import tqdm

# Target size
CROP_SIZE = 512  

def get_next_filename(output_folder):
    """Find the next available incremental filename in the output folder."""
    existing_files = glob.glob(os.path.join(output_folder, "*.jpg"))
    existing_numbers = sorted([int(os.path.basename(f).split(".")[0]) for f in existing_files if os.path.basename(f).split(".")[0].isdigit()])
    return str(existing_numbers[-1] + 1) if existing_numbers else "1"

def resize_to_height(img, target_height):
    """Resize an image while keeping the aspect ratio, ensuring height = target_height."""
    h, w = img.shape[:2]
    scale = target_height / h  # Scaling factor
    new_width = int(w * scale)
    resized_img = cv2.resize(img, (new_width, target_height), interpolation=cv2.INTER_LANCZOS4)
    return resized_img

def random_crop(img, crop_size):
    """Randomly crops a 512x512 section from an image with height=512 and width ≥ 512."""
    h, w = img.shape[:2]
    
    if w < crop_size:
        return None  # Skip if the width is still too small after resizing
    
    x = random.randint(0, w - crop_size)  # Choose random x start point
    return img[:, x:x+crop_size]  # Full height, crop width

def process_images(input_folder):
    """Processes images in the given folder, resizing and cropping them correctly."""
    output_folder = os.path.join(input_folder, "processed")
    os.makedirs(output_folder, exist_ok=True)

    valid_extensions = (".jpg", ".png", ".webp")
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]

    for filename in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        if img is None or img.shape[0] < CROP_SIZE:
            continue  # Skip unreadable images or those with height < 512

        # Resize height to 512 while keeping aspect ratio
        resized_img = resize_to_height(img, CROP_SIZE)

        # Apply random crop
        cropped_img = random_crop(resized_img, CROP_SIZE)
        if cropped_img is None:
            continue  # Skip if width is still too small

        # Determine new filename
        new_filename = get_next_filename(output_folder) + ".jpg"
        save_path = os.path.join(output_folder, new_filename)

        # Save as JPG
        cv2.imwrite(save_path, cropped_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"✅ Processing complete! Cropped images saved in: {output_folder}")

# Example usage
input_directory = "E:\\DL\\"  # Change this to your dataset path
process_images(input_directory)
