# Image Stylization Pipeline with U-2-Net and CycleGAN

This project combines background removal using [U-2-Net](https://github.com/xuebinqin/U-2-Net) with image-to-image style transfer using [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). It produces stylized outputs from images, both with and without background removal, for comparison purposes.

---

## 📦 Features

- Removes image background using U-2-Net
- Stylizes images using CycleGAN (with and without background)
- Saves comparison outputs:
  - `*-stylized-raw.jpg`: stylized original
  - `*-clean.jpg`: original image with background removed
  - `*-stylized.jpg`: stylized background-removed image
  - `*-stylized-raw-clean.jpg`: stylized raw image with its background removed post-hoc

---

## ✨ Training

Training isn't anyhow specific - use any preferred implementation of CycleGAN or use [original paper code](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). Small dataset to test functionality and train pixelizing GAN - [link](https://drive.google.com/file/d/19sgfKAg-giz6xd53nVP8M3GUA8feLJ_z/view?usp=sharing)


## 🚀 Inference

### 1. Clone Dependencies

```bash
# Clone U-2-Net
git clone https://github.com/xuebinqin/U-2-Net.git

# Clone CycleGAN repository
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
```

### 2. Install dependencies

```bash
pip install -r U-2-Net/requirements.txt
pip install -r pytorch-CycleGAN-and-pix2pix/requirements.txt
```

### 3. Ensure/modify folder structure if needed

```python
U2NET_DIR = 'U-2-Net'  # path to cloned U-2-Net
CYCLEGAN_REPO = 'pytorch-CycleGAN-and-pix2pix'  # path to cloned CycleGAN repo
CYCLEGAN_PTH = 'generator_A.pth'  # pretrained CycleGAN generator
IMG_FOLDER_PATH = 'samples'  # folder containing images to process
```

### 4. Download U-2-Net model and place in the following path
```python
f"{U2NET_DIR}/saved_models/u2net.pth"
```

### 5. Put images in samples folder

### 6. Execute working script
```bash
python stylize.py
```

