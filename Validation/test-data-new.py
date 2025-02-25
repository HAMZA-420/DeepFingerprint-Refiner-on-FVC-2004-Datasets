import os
import cv2
import numpy as np
from tqdm import tqdm

# Define paths
CLEAN_DATASET_PATH = r"D:\Deep learning\PROJECT\processed_dataset_final\clean_dataset"
NOISY_DATASET_PATH = r"D:\Deep learning\PROJECT\processed_dataset_final\mixed_noisy_dataset"

TEST_CLEAN_PATH = r"D:\Deep learning\PROJECT\processed_dataset_final\test_dataset\clean_dataset"
TEST_NOISY_PATH = r"D:\Deep learning\PROJECT\processed_dataset_final\test_dataset\noisy_dataset"

# Ensure test dataset directories exist
os.makedirs(TEST_CLEAN_PATH, exist_ok=True)
os.makedirs(TEST_NOISY_PATH, exist_ok=True)

# Define augmentation functions
def adjust_contrast(image, factor=1.2):
    """Adjust contrast by the given factor."""
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

def adjust_brightness(image, factor=1.2):
    """Adjust brightness by the given factor."""
    return cv2.convertScaleAbs(image, alpha=1, beta=int(255 * (factor - 1)))

def add_mild_gaussian_noise(image, mean=0, sigma=10):
    """Add mild Gaussian noise."""
    gauss = np.random.normal(mean, sigma, image.shape).astype('uint8')
    noisy = cv2.add(image, gauss)
    return noisy

def add_synthetic_occlusion(image):
    """Add subtle occlusions to simulate real-world challenges."""
    h, w = image.shape
    mask = np.zeros_like(image)
    x1, y1 = np.random.randint(0, w // 2), np.random.randint(0, h // 2)
    x2, y2 = x1 + np.random.randint(10, w // 4), y1 + np.random.randint(10, h // 4)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return cv2.addWeighted(image, 0.9, mask, 0.1, 0)

# Define augmentation pipeline for noisy dataset
def create_test_image(image, is_noisy=False):
    """Apply transformations for test dataset."""
    # Maintain orientation and add subtle modifications
    if is_noisy:
        image = adjust_contrast(image, factor=1.1)
        image = adjust_brightness(image, factor=0.9)
        image = add_mild_gaussian_noise(image)
        image = add_synthetic_occlusion(image)
    return image

# Process clean dataset (copy as-is)
print("Copying clean dataset for testing...")
for img_name in tqdm(os.listdir(CLEAN_DATASET_PATH)):
    img_path = os.path.join(CLEAN_DATASET_PATH, img_name)
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
        continue
    # Read the image
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        continue
    # Save to test dataset (copy as-is)
    test_img_path = os.path.join(TEST_CLEAN_PATH, img_name)
    cv2.imwrite(test_img_path, image)

# Process noisy dataset (apply augmentations)
print("Creating augmented noisy dataset for testing...")
for img_name in tqdm(os.listdir(NOISY_DATASET_PATH)):
    img_path = os.path.join(NOISY_DATASET_PATH, img_name)
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
        continue
    # Read the image
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        continue
    # Create test image
    test_image = create_test_image(image, is_noisy=True)
    # Save to test dataset
    test_img_path = os.path.join(TEST_NOISY_PATH, img_name)
    cv2.imwrite(test_img_path, test_image)

print("Test dataset creation completed!")
