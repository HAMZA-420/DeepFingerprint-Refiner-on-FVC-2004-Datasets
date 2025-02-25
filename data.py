import os
import cv2
import shutil
import numpy as np
from multiprocessing import Pool
import logging
from skimage.metrics import structural_similarity as ssim

# Configure logging
logging.basicConfig(filename='dataset_processing.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

# Define individual dataset paths
DATASET_PATHS = [
    r"D:\NUST-EME-FA24\Deep learning\PROJECT\dataset\DB1_B",
    r"D:\NUST-EME-FA24\Deep learning\PROJECT\dataset\DB2_B",
    r"D:\NUST-EME-FA24\Deep learning\PROJECT\dataset\DB3_B",
    r"D:\NUST-EME-FA24\Deep learning\PROJECT\dataset\DB4_B"
]

# Define output directories
OUTPUT_BASE_PATH = r"D:\NUST-EME-FA24\Deep learning\PROJECT\processed_dataset_final"
CLEAN_DATASET_PATH = os.path.join(OUTPUT_BASE_PATH, 'clean_dataset')
NOISY_DATASET_PATH = os.path.join(OUTPUT_BASE_PATH, 'noisy_dataset')
MIXED_NOISY_DATASET_PATH = os.path.join(OUTPUT_BASE_PATH, 'mixed_noisy_dataset')
MIXED_CLEAN_DATASET_PATH = os.path.join(OUTPUT_BASE_PATH, 'mixed_clean_dataset')

# Desired uniform size
UNIFORM_SIZE = (300, 480)  # Width x Height

# Noise types and their respective functions
NOISE_TYPES = [
    'gaussian_blur',
    'motion_blur',
    'missing_fingerprint',
    'cut_fingerprint',
    'partial_fingerprint',
    'knify_cut',
    'low_ink',
    'synthetic_artifacts',
    'gaussian_noise',
    'salt_pepper_noise',
    'double_fingerprint',
    'heavy_ink',
    'merged_ridges'
]

# Mapping of noise types to their functions
NOISE_FUNCTIONS = {}

def create_directories():
    """Create necessary directories for clean and noisy datasets."""
    for path in [CLEAN_DATASET_PATH, NOISY_DATASET_PATH, MIXED_NOISY_DATASET_PATH, MIXED_CLEAN_DATASET_PATH]:
        if not os.path.exists(path):
            os.makedirs(path)
            logging.info(f"Created directory: {path}")

    # Create subdirectories for each noise type
    for noise in NOISE_TYPES:
        noise_dir = os.path.join(NOISY_DATASET_PATH, noise)
        if not os.path.exists(noise_dir):
            os.makedirs(noise_dir)
            logging.info(f"Created noise sub-directory: {noise_dir}")

def is_noisy_image(image):
    """
    Detect if an image is noisy based on specific criteria.
    Uses Variance of Laplacian for blurriness and contrast measurement for heavy ink.
    """
    # Calculate Variance of Laplacian (Blurriness)
    variance = cv2.Laplacian(image, cv2.CV_64F).var()
    if variance < 100:  # Threshold for blurriness
        return True

    # Calculate Contrast (Standard Deviation)
    contrast = image.std()
    if contrast < 50:  # Threshold for low contrast (heavy ink or dark image)
        return True

    return False

# Noise augmentation functions
def apply_gaussian_blur(image):
    blurred = cv2.GaussianBlur(image, (15, 15), 0)
    return blurred

def apply_motion_blur(image):
    # Define the kernel size and create a horizontal motion blur kernel
    kernel_size = 15
    kernel = np.zeros((kernel_size, kernel_size))
    # Create a horizontal motion blur
    kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    motion_blurred = cv2.filter2D(image, -1, kernel)
    return motion_blurred

def apply_missing_fingerprint(image):
    # Remove a random rectangular area to simulate missing fingerprint
    h, w = image.shape
    top_left_x = np.random.randint(0, w//3)
    top_left_y = np.random.randint(0, h//3)
    width = np.random.randint(w//4, w//2)
    height = np.random.randint(h//4, h//2)
    image_missing = image.copy()
    cv2.rectangle(image_missing, (top_left_x, top_left_y),
                  (top_left_x + width, top_left_y + height), (255), -1)  # White rectangle
    return image_missing

def apply_cut_fingerprint(image):
    # Simulate a cut by removing a strip
    h, w = image.shape
    cut_width = np.random.randint(w//10, w//5)
    side = np.random.choice(['left', 'right', 'top', 'bottom'])
    image_cut = image.copy()
    if side == 'left':
        cv2.rectangle(image_cut, (0, 0), (cut_width, h), (255), -1)
    elif side == 'right':
        cv2.rectangle(image_cut, (w - cut_width, 0), (w, h), (255), -1)
    elif side == 'top':
        cv2.rectangle(image_cut, (0, 0), (w, cut_width), (255), -1)
    else:  # bottom
        cv2.rectangle(image_cut, (0, h - cut_width), (w, h), (255), -1)
    return image_cut

def apply_partial_fingerprint(image):
    # Keep only a random partial region of the fingerprint
    h, w = image.shape
    start_x = np.random.randint(0, w//2)
    start_y = np.random.randint(0, h//2)
    end_x = start_x + np.random.randint(w//4, w//2)
    end_y = start_y + np.random.randint(h//4, h//2)
    mask = np.zeros_like(image)
    cv2.rectangle(mask, (start_x, start_y), (end_x, end_y), (255), -1)
    image_partial = cv2.bitwise_and(image, mask)
    return image_partial

def apply_knify_cut(image):
    # Introduce irregular cuts or patterns (e.g., diagonal lines)
    h, w = image.shape
    image_knify = image.copy()
    num_lines = np.random.randint(5, 15)
    for _ in range(num_lines):
        x1 = np.random.randint(0, w)
        y1 = np.random.randint(0, h)
        x2 = np.random.randint(0, w)
        y2 = np.random.randint(0, h)
        thickness = np.random.randint(1, 3)
        cv2.line(image_knify, (x1, y1), (x2, y2), (255), thickness)
    return image_knify

def apply_low_ink(image):
    # Reduce image brightness to simulate low ink
    image_low_ink = cv2.convertScaleAbs(image, alpha=0.5, beta=0)  # Decrease brightness by 50%
    return image_low_ink

def apply_synthetic_artifacts(image):
    # Add synthetic scratches or artifacts
    h, w = image.shape
    image_artifacts = image.copy()
    num_scratches = np.random.randint(3, 7)
    for _ in range(num_scratches):
        x1 = np.random.randint(0, w)
        y1 = np.random.randint(0, h)
        x2 = x1 + np.random.randint(-20, 20)
        y2 = y1 + np.random.randint(-20, 20)
        thickness = np.random.randint(1, 3)
        cv2.line(image_artifacts, (x1, y1), (x2, y2), (255), thickness)
    return image_artifacts

def apply_gaussian_noise(image):
    # Add Gaussian noise
    mean = 0
    sigma = 25
    gauss = np.random.normal(mean, sigma, image.shape).astype('uint8')
    image_gauss = cv2.add(image, gauss)
    return image_gauss

def apply_salt_pepper_noise(image):
    # Add Salt and Pepper noise
    s_vs_p = 0.5
    amount = 0.02
    image_sp = image.copy()
    # Salt
    num_salt = np.ceil(amount * image.size * s_vs_p).astype(int)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    image_sp[coords[0], coords[1]] = 255

    # Pepper
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p)).astype(int)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    image_sp[coords[0], coords[1]] = 0
    return image_sp

def apply_double_fingerprint(image):
    """
    Overlay two fingerprint images to simulate double fingerprints.
    Since we don't have a second fingerprint, we'll create a mirrored version.
    """
    mirrored = cv2.flip(image, 1)  # Horizontal flip
    double_fingerprint = cv2.addWeighted(image, 0.5, mirrored, 0.5, 0)
    return double_fingerprint

def apply_heavy_ink(image):
    """
    Create a dark fingerprint with heavy ink by reducing brightness and increasing contrast.
    """
    image_heavy_ink = cv2.convertScaleAbs(image, alpha=1.5, beta=-50)  # Increase contrast and decrease brightness
    image_heavy_ink = np.clip(image_heavy_ink, 0, 255).astype(np.uint8)
    return image_heavy_ink

def apply_merged_ridges(image):
    """
    Alter fingerprint ridges to appear merged or less distinct.
    This can be simulated by applying morphological operations.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    merged = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=2)
    return merged

# Update the NOISE_FUNCTIONS dictionary with new noise types
NOISE_FUNCTIONS = {
    'gaussian_blur': apply_gaussian_blur,
    'motion_blur': apply_motion_blur,
    'missing_fingerprint': apply_missing_fingerprint,
    'cut_fingerprint': apply_cut_fingerprint,
    'partial_fingerprint': apply_partial_fingerprint,
    'knify_cut': apply_knify_cut,
    'low_ink': apply_low_ink,
    'synthetic_artifacts': apply_synthetic_artifacts,
    'gaussian_noise': apply_gaussian_noise,
    'salt_pepper_noise': apply_salt_pepper_noise,
    'double_fingerprint': apply_double_fingerprint,
    'heavy_ink': apply_heavy_ink,
    'merged_ridges': apply_merged_ridges
}

def process_image(image_info):
    """Process a single image: save clean and apply multiple noise augmentations."""
    image_path, dataset_name = image_info
    try:
        # Read image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            logging.error(f"Failed to read image: {image_path}")
            return

        # Resize image to uniform size
        image_resized = cv2.resize(image, UNIFORM_SIZE)

        # Check if the image is already noisy
        if is_noisy_image(image_resized):
            # Move to existing noisy directory
            noisy_existing_dir = os.path.join(NOISY_DATASET_PATH, 'existing_noisy')
            if not os.path.exists(noisy_existing_dir):
                os.makedirs(noisy_existing_dir)
                logging.info(f"Created directory for existing noisy images: {noisy_existing_dir}")
            shutil.move(image_path, os.path.join(noisy_existing_dir, os.path.basename(image_path)))
            logging.info(f"Moved existing noisy image: {image_path}")
            return

        # Save clean image
        clean_image_name = f"{dataset_name}_{os.path.basename(image_path)}"
        clean_image_path = os.path.join(CLEAN_DATASET_PATH, clean_image_name)
        cv2.imwrite(clean_image_path, image_resized)
        logging.info(f"Saved clean image: {clean_image_path}")

        # Apply noise augmentations
        for noise_type, func in NOISE_FUNCTIONS.items():
            noisy_img = func(image_resized)
            noise_dir = os.path.join(NOISY_DATASET_PATH, noise_type)
            noisy_image_name = f"{dataset_name}_{noise_type}_{os.path.basename(image_path)}"
            noisy_image_path = os.path.join(noise_dir, noisy_image_name)
            cv2.imwrite(noisy_image_path, noisy_img)
            logging.info(f"Saved noisy image: {noisy_image_path}")

            # Also copy to mixed_noisy_dataset
            mixed_noisy_image_path = os.path.join(MIXED_NOISY_DATASET_PATH, noisy_image_name)
            shutil.copy(noisy_image_path, mixed_noisy_image_path)
            logging.info(f"Copied to mixed noisy dataset: {mixed_noisy_image_path}")

    except Exception as e:
        logging.error(f"Error processing {image_path}: {e}")

def create_mixed_clean_dataset():
    """Combine all clean images into mixed_clean_dataset directory."""
    try:
        for img_name in os.listdir(CLEAN_DATASET_PATH):
            if img_name.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
                src_path = os.path.join(CLEAN_DATASET_PATH, img_name)
                dst_path = os.path.join(MIXED_CLEAN_DATASET_PATH, img_name)
                shutil.copy(src_path, dst_path)
                logging.info(f"Copied to mixed clean dataset: {dst_path}")
    except Exception as e:
        logging.error(f"Error creating mixed clean dataset: {e}")

def main():
    """Main function to orchestrate dataset processing."""
    create_directories()

    # Prepare a list of all images to process
    images_to_process = []
    for dataset_path in DATASET_PATHS:
        if not os.path.exists(dataset_path):
            logging.warning(f"Dataset path does not exist: {dataset_path}")
            continue

        dataset_name = os.path.basename(dataset_path)
        logging.info(f"\nProcessing dataset: {dataset_name}")

        for img_name in os.listdir(dataset_path):
            if img_name.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
                image_path = os.path.join(dataset_path, img_name)
                images_to_process.append((image_path, dataset_name))

    # Use multiprocessing Pool to process images in parallel
    with Pool(processes=os.cpu_count()) as pool:
        pool.map(process_image, images_to_process)

    # Create mixed clean dataset
    create_mixed_clean_dataset()

    logging.info("\nDataset processing completed.")

if __name__ == "__main__":
    main()
