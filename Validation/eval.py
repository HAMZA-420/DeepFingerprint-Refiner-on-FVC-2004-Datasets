import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from new_final import FingerprintDataset, AdvancedDoubleStageGenerator, transform, device

# Paths to saved models and test data
TEST_CLEAN_PATH = r"D:\Deep learning\PROJECT\processed_dataset_final\test_dataset\clean_dataset"
TEST_NOISY_PATH = r"D:\Deep learning\PROJECT\processed_dataset_final\test_dataset\noisy_dataset"
OUTPUT_DIR = r"D:\Deep learning\PROJECT\FINAL-PROJECT\PROJECT-NEW-LATEST\FINAL\Validation"

GENERATOR_PATH = os.path.join(OUTPUT_DIR, "generator_epoch_270.pth")

def visualize_and_save(generator, test_loader, output_dir):
    """
    Generate and save side-by-side visualizations of noisy, generated, and clean images for all test data.

    Args:
        generator (nn.Module): Pretrained generator model.
        test_loader (DataLoader): DataLoader for the test dataset.
        output_dir (str): Directory to save the visualizations.
    """
    generator.eval()
    os.makedirs(output_dir, exist_ok=True)
    image_count = 0  # Counter for naming saved images

    with torch.no_grad():
        for noisy_imgs, clean_imgs in test_loader:
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)
            mask = torch.ones_like(noisy_imgs).to(device)

            fake_imgs = generator(noisy_imgs, mask)

            batch_size = noisy_imgs.size(0)
            for i in range(batch_size):
                noisy_img = noisy_imgs[i].cpu().numpy().squeeze()
                fake_img = fake_imgs[i].cpu().numpy().squeeze()
                clean_img = clean_imgs[i].cpu().numpy().squeeze()

                # Convert images from [-1, 1] to [0, 1] for visualization
                noisy_img = ((noisy_img + 1) / 2).clip(0, 1)
                fake_img = ((fake_img + 1) / 2).clip(0, 1)
                clean_img = ((clean_img + 1) / 2).clip(0, 1)

                # Plot the results side-by-side
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))

                axs[0].imshow(noisy_img, cmap="gray")
                axs[0].set_title("Noisy Input")
                axs[0].axis("off")

                axs[1].imshow(fake_img, cmap="gray")
                axs[1].set_title("Generated Output")
                axs[1].axis("off")

                axs[2].imshow(clean_img, cmap="gray")
                axs[2].set_title("Ground Truth")
                axs[2].axis("off")

                plt.tight_layout()

                # Save the figure
                save_path = os.path.join(output_dir, f"comparison_{image_count}.png")
                plt.savefig(save_path)
                plt.close()

                print(f"Saved {save_path}")
                image_count += 1

    print(f"All {image_count} images have been processed and saved in {output_dir}.")

if __name__ == '__main__':
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load test dataset
    test_dataset = FingerprintDataset(clean_dir=TEST_CLEAN_PATH, noisy_dir=TEST_NOISY_PATH, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    # Load pretrained generator model
    generator = AdvancedDoubleStageGenerator().to(device)

    # Load the saved checkpoint
    if not os.path.exists(GENERATOR_PATH):
        raise FileNotFoundError(f"Generator checkpoint not found at {GENERATOR_PATH}")
    
    checkpoint = torch.load(GENERATOR_PATH, map_location=device)
    generator.load_state_dict(checkpoint)
    print(f"Loaded generator model from {GENERATOR_PATH}")

    generator.eval()

    # Visualize and save all test images
    visualize_and_save(generator, test_loader, os.path.join(OUTPUT_DIR, "test_results"))

    print("Inference completed! Enhanced images are saved in the output directory.")
