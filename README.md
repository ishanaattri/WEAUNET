# WEAUNET
Wavelet model for image segmentation
# Install dependencies (if needed)
# !pip install pywavelets albumentations opencv-python-headless torch torchvision matplotlib

import os
import cv2
import numpy as np
import pywt
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ===========================================
# 1. Dataset + Wavelet Transform
# ===========================================
class AppleLeafDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def wavelet_decompose(self, image):
        coeffs = pywt.dwt2(image, 'haar')
        cA, (cH, cV, cD) = coeffs
        return np.stack([cA, cH, cV, cD], axis=-1)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = cv2.imread(self.mask_paths[idx], 0)

        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256))

        wavelet_img = self.wavelet_decompose(image)
        wavelet_img = cv2.resize(wavelet_img, (256, 256))

        if self.transform:
            augmented = self.transform(image=wavelet_img, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask.long()

    def __len__(self):
        return len(self.image_paths)

transform = A.Compose([
    A.Normalize(),
    ToTensorV2()
])

# Replace these paths with your actual dataset paths
image_paths = sorted(glob("dataset/images/*.jpg"))
mask_paths = sorted(glob("dataset/masks/*.png"))

dataset = AppleLeafDataset(image_paths, mask_paths, transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# ===========================================
# 2. Attention and Conv Blocks
# ===========================================
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        psi = F.relu(self.W_g(g) + self.W_x(x))
        psi = self.psi(psi)
        return x * psi

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

# ===========================================
# 3. WEA-U-Net Model
# ===========================================
class WEA_UNet(nn.Module):
    def __init__(self, in_ch=4, out_ch=1):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(128, 256)

        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(128, 128, 64)
        self.dec1 = ConvBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(64, 64, 32)
        self.dec2 = ConvBlock(128, 64)

        self.final = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        d1 = self.up1(b)
        e2 = self.att1(d1, e2)
        d1 = self.dec1(torch.cat([d1, e2], dim=1))

        d2 = self.up2(d1)
        e1 = self.att2(d2, e1)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        return torch.sigmoid(self.final(d2))

# ===========================================
# 4. Dice Loss and Training
# ===========================================
def dice_loss(pred, target, smooth=1):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WEA_UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(10):
    model.train()
    epoch_loss = 0
    for images, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = dice_loss(outputs, masks.unsqueeze(1).float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {epoch_loss/len(dataloader):.4f}")

# ===========================================
# 5. Post-Processing and Example Inference
# ===========================================
def post_process(preds, threshold=0.5):
    return (preds > threshold).float()

# Visualize one prediction
model.eval()
with torch.no_grad():
    img, mask = dataset[0]
    img = img.unsqueeze(0).to(device)
    pred = model(img)
    pred_mask = post_process(pred[0][0].cpu())

plt.subplot(1, 3, 1); plt.title("Input"); plt.imshow(img[0][0].cpu(), cmap='gray')
plt.subplot(1, 3, 2); plt.title("Ground Truth"); plt.imshow(mask, cmap='gray')
plt.subplot(1, 3, 3); plt.title("Prediction"); plt.imshow(pred_mask, cmap='gray')
plt.tight_layout(); plt.show()
