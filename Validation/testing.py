import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch.nn as nn

from new_final import FingerprintDataset, AdvancedDoubleStageGenerator, transform, device

# Paths to saved models and test data
TEST_CLEAN_PATH = r"D:\Deep learning\PROJECT\processed_dataset_final\test_dataset\clean_dataset"
TEST_NOISY_PATH = r"D:\Deep learning\PROJECT\processed_dataset_final\test_dataset\noisy_dataset"
OUTPUT_DIR = r"D:\NUST-EME-FA24\Deep learning\PROJECT\FINAL-PROJECT\PROJECT-NEW-LATEST\FINAL\Validation"

GENERATOR_PATH = os.path.join(OUTPUT_DIR, "generator_epoch_270.pth")
STARTING_EPOCH = 271  # Starting epoch for validation

# Generator and Discriminator losses (simulated)
generator_losses = [5.0, 4.8, 4.6, 4.4, 4.2, 4.0, 3.8, 3.6, 3.5]
discriminator_losses = [0.003, 0.0028, 0.0025, 0.0023, 0.002, 0.0018, 0.0015, 0.0013, 0.001]

# Function to calculate per-batch validation loss
def calculate_batch_losses(generator, test_loader, criterion):
    generator.eval()
    batch_losses = []
    with torch.no_grad():
        for noisy_imgs, clean_imgs in test_loader:
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)
            mask = torch.ones_like(noisy_imgs).to(device)

            fake_imgs = generator(noisy_imgs, mask)
            loss = criterion(fake_imgs, clean_imgs)  # L1 loss
            batch_losses.append(loss.item())

    return batch_losses

# Function to compute SSIM
def compute_ssim(clean_imgs, fake_imgs):
    clean = clean_imgs.cpu().numpy().squeeze()
    fake = fake_imgs.cpu().numpy().squeeze()
    return ssim(clean, fake, data_range=clean.max() - clean.min())

# Function to plot pixel value distribution
def plot_pixel_distributions(noisy_imgs, fake_imgs, clean_imgs, output_dir):
    plt.figure(figsize=(10, 6))

    # Detach tensors and move to CPU before converting to numpy
    noisy_imgs = noisy_imgs.detach().cpu().numpy()
    fake_imgs = fake_imgs.detach().cpu().numpy()
    clean_imgs = clean_imgs.detach().cpu().numpy()

    for data, label in zip([noisy_imgs, fake_imgs, clean_imgs], ["Noisy", "Generated", "Clean"]):
        plt.hist(data.flatten(), bins=50, alpha=0.6, label=label)

    plt.title("Pixel Value Distribution")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pixel_distributions.png"))
    plt.show()


# Function to plot per-batch losses
def plot_batch_losses(batch_losses, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(batch_losses, marker="o", linestyle="-", label="Validation Batch Loss")
    plt.title("Per-Batch Validation Loss")
    plt.xlabel("Batch Index")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "batch_losses.png"))
    plt.show()

# Function to plot loss curves
def plot_loss_curves(generator_losses, discriminator_losses, val_losses, output_dir, starting_epoch):
    epochs = range(starting_epoch, starting_epoch + len(generator_losses))
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, generator_losses, label="Generator Loss", marker='o')
    plt.plot(epochs, discriminator_losses, label="Discriminator Loss", marker='x')
    plt.plot(epochs, val_losses, label="Validation Loss", marker='s')
    plt.title(f"Loss Curves (Starting from Epoch {starting_epoch})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curves.png"))
    plt.show()

# Function to visualize epoch progression
def visualize_epoch_progression(generator, test_loader, output_dir, num_images=4):
    generator.eval()
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    with torch.no_grad():
        for noisy_imgs, clean_imgs in test_loader:
            noisy_imgs = noisy_imgs[:num_images].to(device)
            clean_imgs = clean_imgs[:num_images].to(device)
            mask = torch.ones_like(noisy_imgs).to(device)

            fake_imgs = generator(noisy_imgs, mask)

            # Visualize side-by-side for selected images
            for i in range(num_images):
                noisy_img = ((noisy_imgs[i].cpu().numpy().squeeze() + 1) / 2 * 255).astype("uint8")
                fake_img = ((fake_imgs[i].cpu().numpy().squeeze() + 1) / 2 * 255).astype("uint8")
                clean_img = ((clean_imgs[i].cpu().numpy().squeeze() + 1) / 2 * 255).astype("uint8")

                # Plot the results
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 3, 1)
                plt.title("Noisy")
                plt.imshow(noisy_img, cmap="gray")
                plt.axis("off")
                plt.subplot(1, 3, 2)
                plt.title("Generated")
                plt.imshow(fake_img, cmap="gray")
                plt.axis("off")
                plt.subplot(1, 3, 3)
                plt.title("Clean")
                plt.imshow(clean_img, cmap="gray")
                plt.axis("off")
                plt.tight_layout()

                plt.savefig(os.path.join(output_dir, f"progression_{count}.png"))
                plt.close()
                count += 1
            break  # Only visualize once for simplicity

if __name__ == '__main__':
    # Load test dataset
    test_dataset = FingerprintDataset(clean_dir=TEST_CLEAN_PATH, noisy_dir=TEST_NOISY_PATH, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    # Load pretrained generator model
    generator = AdvancedDoubleStageGenerator().to(device)

    # Load the saved checkpoint
    checkpoint = torch.load(GENERATOR_PATH)
    generator.load_state_dict(checkpoint)

    generator.eval()

    # Define loss function (L1 Loss)
    criterion = nn.L1Loss()

    # Per-batch validation losses
    batch_losses = calculate_batch_losses(generator, test_loader, criterion)
    print(f"Per-batch Validation Losses: {batch_losses}")

    # Plot per-batch losses
    plot_batch_losses(batch_losses, OUTPUT_DIR)

    # Pixel value distributions
    noisy_imgs, clean_imgs = next(iter(test_loader))
    mask = torch.ones_like(noisy_imgs).to(device)
    fake_imgs = generator(noisy_imgs.to(device), mask)
    plot_pixel_distributions(noisy_imgs, fake_imgs, clean_imgs, OUTPUT_DIR)

    # Visualize progression
    visualize_epoch_progression(generator, test_loader, os.path.join(OUTPUT_DIR, "epoch_progression"))

    # Validation loss
    val_losses = [sum(batch_losses) / len(batch_losses)] * len(generator_losses)  # Simplified
    plot_loss_curves(generator_losses, discriminator_losses, val_losses, OUTPUT_DIR, STARTING_EPOCH)

    print("Validation visualization completed!")
