import os
import glob
import random
import numpy as np
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# For mixed precision
from torch.cuda import amp

import cv2  # We'll do bridging / line drawing in CPU with OpenCV
from tqdm import tqdm, trange

########################################
# 0. Paths & Hyperparameters
########################################

CLEAN_DATASET_PATH = r"D:\Deep learning\PROJECT\processed_dataset_final\clean_dataset"
NOISY_DATASET_PATH = r"D:\Deep learning\PROJECT\processed_dataset_final\mixed_noisy_dataset"
OUTPUT_DIR         = r"D:\Deep learning\PROJECT\FINAL-PROJECT\PROJECT-NEW-LATEST\FINAL"
os.makedirs(OUTPUT_DIR, exist_ok=True)

num_epochs    = 300
batch_size    = 4  # Reduced from 16 to 4 to fit 6GB VRAM
learning_rate = 2e-4
lambda_L1     = 100

# For 256×512 images
transform = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

########################################
# 1. FingerprintDataset (Noisy–Clean)
########################################

class FingerprintDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, transform=None):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.transform = transform

        self.clean_images = sorted(glob.glob(os.path.join(self.clean_dir, "*")))
        self.noisy_images = sorted(glob.glob(os.path.join(self.noisy_dir, "*")))

        # Map from clean filename => path
        self.clean_map = {}
        for path in self.clean_images:
            fn = os.path.basename(path)
            self.clean_map[fn] = path

        # Build paired list
        self.paired_images = []
        for noisy_path in self.noisy_images:
            noisy_fn = os.path.basename(noisy_path)
            parts    = noisy_fn.split('_')
            if len(parts) < 5:
                print(f"Unexpected filename format: {noisy_fn}")
                continue

            # Example logic from your code
            base_filename = '_'.join(parts[:2] + parts[-2:])
            base_filename = base_filename.replace('.tif','.tif')
            if base_filename in self.clean_map:
                self.paired_images.append((noisy_path, self.clean_map[base_filename]))
            else:
                print(f"No matching clean image for noisy image: {noisy_fn}")

        print(f"Total noisy images: {len(self.noisy_images)}")
        print(f"Total clean images: {len(self.clean_images)}")
        print(f"Total paired images: {len(self.paired_images)}")

    def __len__(self):
        return len(self.paired_images)

    def __getitem__(self, idx):
        noisy_path, clean_path = self.paired_images[idx]
        noisy_img = Image.open(noisy_path).convert("L")
        clean_img = Image.open(clean_path).convert("L")

        if self.transform:
            noisy_img = self.transform(noisy_img)
            clean_img = self.transform(clean_img)
        return noisy_img, clean_img

########################################
# 2. Two-Stage Gated PartialConv + Self-Attn Generator
########################################

class GatedPartialConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True):
        super().__init__()
        self.conv_feat = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.conv_gate = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        self.register_buffer(
            'weight_maskUpdater',
            torch.ones((out_channels, in_channels, kernel_size, kernel_size))
        )
        self.slide_winsize = self.weight_maskUpdater[0,0].numel()

    def forward(self, x, mask):
        with torch.no_grad():
            updated_mask = F.conv2d(
                mask, self.weight_maskUpdater,
                stride=self.conv_feat.stride,
                padding=self.conv_feat.padding
            )
            updated_mask = torch.clamp(updated_mask, 0, 1)
        masked_input = x * mask

        feat = self.conv_feat(masked_input)
        gate = torch.sigmoid(self.conv_gate(masked_input))
        out  = feat * gate

        with torch.no_grad():
            ratio = self.slide_winsize / (updated_mask + 1e-8)
            ratio = torch.clamp(ratio, 0, 10.0)
            ratio = ratio * (updated_mask > 0)

        out = out * ratio
        return out, updated_mask

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query  = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key    = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value  = nn.Conv2d(in_channels, in_channels,      1)
        self.gamma  = nn.Parameter(torch.zeros(1))
        self.softmax= nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.query(x).view(b, -1, h*w)   # B, c//8, N
        k = self.key(x).view(b, -1, h*w)     # B, c//8, N
        v = self.value(x).view(b, -1, h*w)   # B, c,     N

        attn = torch.bmm(q.permute(0,2,1), k)  # B, N, N
        attn = self.softmax(attn)
        out  = torch.bmm(v, attn.permute(0,2,1))  # B, c, N
        out  = out.view(b, c, h, w)
        return self.gamma * out + x

class GatedPartialConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_bn=True):
        super().__init__()
        self.gated = GatedPartialConv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.use_bn= use_bn
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, mask):
        out, upd = self.gated(x, mask)
        if self.use_bn:
            out = self.bn(out)
        out = F.leaky_relu(out, 0.2, inplace=True)
        return out, upd

class GatedPartialConvUNet(nn.Module):
    """
    4-level partialconv UNet with self-attn in the bottleneck
    """
    def __init__(self, in_channels=1, out_channels=1, base=64):
        super().__init__()
        # Encoder
        self.enc1 = GatedPartialConvBlock(in_channels, base,   use_bn=False)
        self.enc2 = GatedPartialConvBlock(base, base*2)
        self.enc3 = GatedPartialConvBlock(base*2, base*4)
        self.enc4 = GatedPartialConvBlock(base*4, base*8)
        self.attn = SelfAttention(base*8)

        # Decoder
        self.dec4 = nn.ConvTranspose2d(base*8, base*4, 4, 2, 1, bias=False)
        self.dec3 = nn.ConvTranspose2d(base*8, base*2, 4, 2, 1, bias=False)
        self.dec2 = nn.ConvTranspose2d(base*4, base,   4, 2, 1, bias=False)
        self.dec1 = nn.ConvTranspose2d(base*2, out_channels, 4, 2, 1, bias=False)

    def forward(self, x, mask):
        e1, m1 = self.enc1(x, mask)
        e2, m2 = self.enc2(e1, m1)
        e3, m3 = self.enc3(e2, m2)
        e4, m4 = self.enc4(e3, m3)
        b      = self.attn(e4)

        d4 = F.relu(self.dec4(b), inplace=True)
        d4 = torch.cat([d4, e3], dim=1)

        d3 = F.relu(self.dec3(d4), inplace=True)
        d3 = torch.cat([d3, e2], dim=1)

        d2 = F.relu(self.dec2(d3), inplace=True)
        d2 = torch.cat([d2, e1], dim=1)

        d1 = self.dec1(d2)
        out = torch.tanh(d1)
        return out

class AdvancedDoubleStageGenerator(nn.Module):
    """
    Stage1 partialconv UNet
    Stage2 partialconv UNet
    """
    def __init__(self, in_channels=1, out_channels=1, base_channels=64):
        super().__init__()
        self.stage1 = GatedPartialConvUNet(in_channels, out_channels, base_channels)
        self.stage2 = GatedPartialConvUNet(in_channels, out_channels, base_channels)

    def forward(self, x, mask):
        coarse  = self.stage1(x, mask)
        refined = self.stage2(coarse, mask)
        return refined

########################################
# 3. Patch Discriminator
########################################

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=2, features=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(features, features*2, 4, 2, 1),
            nn.BatchNorm2d(features*2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(features*2, features*4, 4, 2, 1),
            nn.BatchNorm2d(features*4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(features*4, features*8, 4, 1, 1),
            nn.BatchNorm2d(features*8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(features*8, 1, 4, 1, 1)
        )
    def forward(self, x):
        return self.net(x)

########################################
# 4. Post-processing: minimal bridging + line drawing
########################################

MAX_THIN_ITER = 30    # Reduced iterations to prevent hangs
MAX_ENDPOINTS = 1000  # Reduced cap on endpoints
MAX_CONNECT_DIST = 25 # Adjusted max distance for connections

def morphological_bridging(bin_img):
    """
    bin_img: [H,W], [0..255], bridging lines
    Single pass: morphological close + open
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    closed = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    opened = cv2.morphologyEx(closed,  cv2.MORPH_OPEN,  kernel, iterations=1)
    return opened

def thin_lines(bin_img, max_iter=MAX_THIN_ITER):
    """
    Thinning with iteration limit to avoid infinite loops.
    """
    skel   = np.zeros_like(bin_img)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    temp   = bin_img.copy()

    for _ in range(max_iter):
        erode  = cv2.erode(temp, kernel)
        dilate = cv2.dilate(erode, kernel)
        temp2  = cv2.subtract(temp, dilate)
        skel   = cv2.bitwise_or(skel, temp2)
        temp   = erode.copy()
        if cv2.countNonZero(temp) == 0:
            break
    return skel

def connect_endpoints(bin_img, max_dist=MAX_CONNECT_DIST):
    """
    Find endpoints in bin_img, connect them if close enough.
    Capped if there are too many endpoints.
    """
    skeleton = thin_lines(bin_img)
    endpoints = []
    offsets   = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    h, w      = skeleton.shape

    # Identify endpoints
    for r in range(1, h-1):
        for c in range(1, w-1):
            if skeleton[r,c] == 255:
                count_n = 0
                for (dr,dc) in offsets:
                    if skeleton[r+dr,c+dc] == 255:
                        count_n += 1
                if count_n == 1:
                    endpoints.append((r,c))

    # If no endpoints or the image is blank, return early
    if len(endpoints) < 2:
        return

    # Cap endpoints to avoid O(N^2) on huge sets
    if len(endpoints) > MAX_ENDPOINTS:
        endpoints = endpoints[:MAX_ENDPOINTS]

    used = set()
    endpoints = sorted(endpoints)
    for i, (r1,c1) in enumerate(endpoints):
        if i in used:
            continue
        best_dist = float('inf')
        best_j    = None
        for j, (r2,c2) in enumerate(endpoints):
            if j == i or j in used:
                continue
            dist = (r1 - r2)**2 + (c1 - c2)**2
            if dist < best_dist:
                best_dist = dist
                best_j    = j
        if best_j is not None and best_dist <= max_dist**2:
            r2, c2 = endpoints[best_j]
            cv2.line(bin_img, (c1, r1), (c2, r2), 255, 1)  # Thinner lines for efficiency
            used.add(i)
            used.add(best_j)

def draw_new_lines(final_bin, old_bin):
    """
    final_bin: final [0..255]
    old_bin:   original noisy [0..255]
    We'll do bridging in region where old_bin < threshold => "missing".
    Then connect endpoints, but only 1 pass.
    """
    mask_thresh = 70
    # Define missing region
    missing_mask = (old_bin < mask_thresh)

    # Bridging on final_bin
    bridged = morphological_bridging(final_bin)
    connect_endpoints(bridged, max_dist=MAX_CONNECT_DIST)

    # Re-thicken region
    out = final_bin.copy()
    out[missing_mask] = bridged[missing_mask]
    return out

########################################
# 5. Save Sample Images
########################################

def save_sample_images(
    generator,
    noisy_imgs,
    clean_imgs,
    epoch,
    results_dir,
    max_samples=4
):
    device = noisy_imgs.device
    generator.eval()

    # Trivial mask=1
    mask = torch.ones_like(noisy_imgs)
    with torch.no_grad():
        final_fake = generator(noisy_imgs, mask)

    out_list = []
    num_samples = min(max_samples, final_fake.shape[0])
    for i in range(num_samples):
        final_np = ((final_fake[i].cpu().numpy().squeeze() + 1) / 2 * 255).clip(0,255).astype(np.uint8)
        old_np   = ((noisy_imgs[i].cpu().numpy().squeeze() + 1) / 2 * 255).clip(0,255).astype(np.uint8)

        # Line-draw only where old was "noisy"
        completed = draw_new_lines(final_np, old_np)
        # Convert back to [-1,1]
        out_01 = completed.astype(np.float32) / 255.
        out_2  = out_01 * 2 - 1
        out_list.append(out_2)

    final_completed = torch.tensor(out_list).unsqueeze(1)  # [B,1,H,W], CPU

    # Denorm for plotting
    noisy_cpu = (noisy_imgs[:num_samples].cpu().numpy() * 0.5 + 0.5).clip(0,1)
    final_cpu = (final_completed.numpy()           * 0.5 + 0.5).clip(0,1)
    clean_cpu = (clean_imgs[:num_samples].cpu().numpy() * 0.5 + 0.5).clip(0,1)

    fig, axs = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    for i in range(num_samples):
        axs[i,0].imshow(noisy_cpu[i,0], cmap='gray')
        axs[i,0].set_title("Noisy Input")
        axs[i,0].axis('off')

        axs[i,1].imshow(final_cpu[i,0], cmap='gray')
        axs[i,1].set_title("Lines Drawn On Top")
        axs[i,1].axis('off')

        axs[i,2].imshow(clean_cpu[i,0], cmap='gray')
        axs[i,2].set_title("Ground Truth")
        axs[i,2].axis('off')

    plt.tight_layout()
    samples_folder = os.path.join(results_dir, "samples")
    os.makedirs(samples_folder, exist_ok=True)
    plt.savefig(os.path.join(samples_folder, f"epoch_{epoch}.png"))
    plt.close()
    generator.train()

########################################
# 6. Training Loop
########################################

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load dataset
    dataset = FingerprintDataset(
        clean_dir=CLEAN_DATASET_PATH,
        noisy_dir=NOISY_DATASET_PATH,
        transform=transform
    )
    if len(dataset) == 0:
        raise ValueError("No paired images found. Check dataset paths/filenames.")

    # Adjust num_workers based on CPU cores. Assuming 8 cores for Ryzen 7 7840HS
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    # Models
    generator     = AdvancedDoubleStageGenerator().to(device)
    discriminator = PatchDiscriminator().to(device)

    # Mixed Precision Scaler
    scaler_G = amp.GradScaler()
    scaler_D = amp.GradScaler()

    # Loss
    criterion_GAN = nn.MSELoss()  # LSGAN
    criterion_L1  = nn.L1Loss()

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(),     lr=learning_rate, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # Training
    for epoch in range(1, num_epochs + 1):
        generator.train()
        discriminator.train()

        loop = tqdm(dataloader, desc=f"Epoch [{epoch}/{num_epochs}]", leave=False)
        for i, (noisy, clean) in enumerate(loop):
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

            # Trivial mask=1
            mask = torch.ones_like(noisy).to(device, non_blocking=True)

            #### D step ####
            optimizer_D.zero_grad()
            with torch.no_grad():
                fake = generator(noisy, mask)

            real_input = torch.cat([noisy, clean], dim=1)
            pred_real  = discriminator(real_input)
            loss_d_real= criterion_GAN(pred_real, torch.ones_like(pred_real, device=device))

            fake_input = torch.cat([noisy, fake.detach()], dim=1)
            pred_fake  = discriminator(fake_input)
            loss_d_fake= criterion_GAN(pred_fake, torch.zeros_like(pred_fake, device=device))

            loss_d = 0.5 * (loss_d_real + loss_d_fake)

            # Backpropagation with mixed precision
            scaler_D.scale(loss_d).backward()
            scaler_D.step(optimizer_D)
            scaler_D.update()

            #### G step ####
            optimizer_G.zero_grad()
            with amp.autocast():
                fake = generator(noisy, mask)
                fake_input = torch.cat([noisy, fake], dim=1)
                pred_fake_g= discriminator(fake_input)
                loss_g_gan = criterion_GAN(pred_fake_g, torch.ones_like(pred_fake_g, device=device))
                loss_g_l1  = criterion_L1(fake, clean) * lambda_L1
                loss_g     = loss_g_gan + loss_g_l1

            # Backpropagation with mixed precision
            scaler_G.scale(loss_g).backward()
            scaler_G.step(optimizer_G)
            scaler_G.update()

            loop.set_postfix({
                "loss_d": f"{loss_d.item():.3f}",
                "loss_g": f"{loss_g.item():.3f}"
            })

        # Save sample images every 10 epochs or if it's the first/last epoch
        if epoch == 1 or epoch == num_epochs or epoch % 10 == 0:
            save_sample_images(generator, noisy[:4], clean[:4], epoch, OUTPUT_DIR)
            # Save model states
            torch.save(generator.state_dict(),     os.path.join(OUTPUT_DIR, f'generator_epoch_{epoch}.pth'))
            torch.save(discriminator.state_dict(), os.path.join(OUTPUT_DIR, f'discriminator_epoch_{epoch}.pth'))

        print(f"Epoch [{epoch}/{num_epochs}] completed.")

        # Clear CUDA cache to free up memory
        torch.cuda.empty_cache()

    print("Training Completed!")
    # Final model save
    torch.save(generator.state_dict(),     os.path.join(OUTPUT_DIR, "generator_final.pth"))
    torch.save(discriminator.state_dict(), os.path.join(OUTPUT_DIR, "discriminator_final.pth"))

if __name__ == "__main__":
    main()
