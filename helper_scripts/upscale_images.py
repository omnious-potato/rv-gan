from PIL import Image
import os

input_dir = "E:\\Projects\\Robotic Vision PZ GAN integration\\dataset\\processed_images"
output_dir = "E:\\DL\\pure_target_dataset\\classB"


os.makedirs(output_dir, exist_ok=True)


for filename in os.listdir(input_dir):
    if filename.endswith(".png") or filename.endswith(".jpg"):

        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path).convert("RGB")  # Convert to 3 channels (RGB)
        img = img.resize((512, 512), Image.LANCZOS)

        img.save(os.path.join(output_dir, filename))  # Save with the same name