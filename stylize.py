import sys
import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

# --- PATHS ---
U2NET_DIR = 'U-2-Net'
U2NET_PTH = os.path.join(U2NET_DIR, 'saved_models', 'u2net.pth')
CYCLEGAN_REPO = 'pytorch-CycleGAN-and-pix2pix'
CYCLEGAN_PTH = 'generator_A.pth'

IMG_FOLDER_PATH = 'samples'


OUT_PATH = 'output'

OUT_MASK_PATH = 'mask.png'
OUT_CLEANED_PATH = 'input_cleaned.jpg'
OUT_STYLIZED_PATH = 'stylized_output.jpg'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- IMPORT MODULES FROM REPOS ---
sys.path.append(CYCLEGAN_REPO)
sys.path.append(os.path.join(U2NET_DIR, 'model'))

from models.networks import define_G
from u2net import U2NET

# --- LOAD U-2-NET ---
def load_u2net():
    model = U2NET(3, 1)
    model.load_state_dict(torch.load(U2NET_PTH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def remove_background(img_path, model):
    original = Image.open(img_path).convert("RGB")
    orig_size = original.size

    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])
    
    input_tensor = transform(original).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        d1, *_ = model(input_tensor)
        pred = d1[:, 0, :, :]
        pred = (pred - pred.min()) / (pred.max() - pred.min())
        pred = pred.squeeze().cpu().numpy()

    pred_resized = cv2.resize(pred, orig_size)
    mask = (pred_resized * 255).astype(np.uint8)
    mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)[1]
    return np.array(original), mask

def apply_black_bg(image_np, mask):
    black_bg = np.zeros_like(image_np)
    result = np.where(mask[:, :, None] == 255, image_np, black_bg)
    return result

# --- LOAD CYCLEGAN GENERATOR ---
def load_generator(pth):
    netG = define_G(input_nc=3, output_nc=3, ngf=64, netG='resnet_9blocks',
                    norm='instance', use_dropout=False, init_type='normal',
                    init_gain=0.02, gpu_ids=[])
    netG.load_state_dict(torch.load(pth, map_location=DEVICE))
    netG.to(DEVICE)
    netG.eval()
    return netG


# --- Configurable resize behavior ---
resize_mode = 'pad'  # or 'resize'
gan_input_size = 512  # size expected by CycleGAN

def pad_to_square(image):
    """Pads image to make it square with black bars."""
    h, w = image.shape[:2]
    if h == w:
        return image
    size = max(h, w)
    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left
    color = [0, 0, 0]
    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded_image

def transform_for_gan(image_np):
    if resize_mode == 'pad':
        padded = pad_to_square(image_np)
        resized = cv2.resize(padded, (gan_input_size, gan_input_size), interpolation=cv2.INTER_AREA)
    else:
        resized = cv2.resize(image_np, (gan_input_size, gan_input_size), interpolation=cv2.INTER_AREA)

    pil = Image.fromarray(resized)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transform(pil).unsqueeze(0)

def run_cyclegan(image_np, generator):
    input_tensor = transform_for_gan(image_np).to(DEVICE)

    with torch.no_grad():
        output = generator(input_tensor)
    output = output[0].cpu().detach().numpy()
    output = (output + 1) / 2
    output = np.transpose(output, (1, 2, 0)) * 255
    return output.astype(np.uint8)

# --- MAIN PIPELINE ---
def main():
    print("Loading U-2-Net...")
    u2net = load_u2net()
    
    print("Loading CycleGAN generator...")
    generator = load_generator(CYCLEGAN_PTH)
    
    for filename in os.listdir(IMG_FOLDER_PATH):
        full_path = os.path.join(IMG_FOLDER_PATH, filename)

        print(f"Processing file \"{filename}\"")

        no_ext_name = os.path.splitext(filename)[0]
        
        # Load original image
        original_pil = Image.open(full_path).convert("RGB")
        original_np = np.array(original_pil)

        # Stylize original image without background removal
        stylized_raw = run_cyclegan(original_np, generator)
        cv2.imwrite(no_ext_name + "-stylized-raw.jpg", cv2.cvtColor(stylized_raw, cv2.COLOR_RGB2BGR))

        # Background removal
        img_np, mask = remove_background(full_path, u2net)
        cv2.imwrite(no_ext_name + "-mask.jpg", mask)

        cleaned = apply_black_bg(img_np, mask)
        cv2.imwrite(no_ext_name + "-clean.jpg", cv2.cvtColor(cleaned, cv2.COLOR_RGB2BGR))

        # Stylize cleaned image
        stylized = run_cyclegan(cleaned, generator)
        cv2.imwrite(no_ext_name + "-stylized.jpg", cv2.cvtColor(stylized, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    main()
