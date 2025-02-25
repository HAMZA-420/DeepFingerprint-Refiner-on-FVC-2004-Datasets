# DeepFingerprint-Refiner
**A Gated PartialConv + Self-Attention Approach to Fingerprint Restoration and Denoising**

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg?style=flat-square" />
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg?style=flat-square" />
</p>

This repository contains a **double-stage Generator** (using **Gated Partial Convolutions** and **Self-Attention**) and a **Patch Discriminator**, designed to inpaint and refine fingerprint images. The model **removes noise**, **bridges broken lines**, and **thins the ridges** for a cleaner fingerprint restoration, producing visually and structurally consistent output.

---

## Table of Contents
1. [Features](#features)
2. [Model Architecture](#model-architecture)
3. [Getting Started](#getting-started)
   - [Installation](#installation)
   - [Dataset Structure](#dataset-structure)
   - [Running the Training Script](#running-the-training-script)
4. [Results](#results)
5. [Project Structure](#project-structure)
6. [Usage & Examples](#usage--examples)
7. [Contributing](#contributing)
8. [License](#license)
9. [Acknowledgments](#acknowledgments)

---

## Features
- **Double-Stage Gated PartialConv Generator** to handle missing/noisy fingerprint regions.
- **Self-Attention** to capture global context and improve coherence in fingerprints.
- **Patch Discriminator (LSGAN)** for adversarial training on local patches.
- **Post-processing** with morphological operations and endpoint-bridging to restore continuity in ridges.
- **Mixed Precision Training** for faster runs and lower memory usage on GPUs.

---

## Model Architecture

### Generator
1. **Stage 1**: Gated PartialConv U-Net (4-level depth)  
2. **Stage 2**: Another Gated PartialConv U-Net refining the output of Stage 1  
3. **Self-Attention** inserted in the bottleneck of each U-Net for global coherence.

### Discriminator
- **Patch Discriminator** that operates on concatenated `(noisy_input, generated/real_output)` pairs, following the LSGAN style.

<p align="center">
  <img src="https://user-images.githubusercontent.com/placeholder/model_diagram.png" alt="Model Diagram" width="600">
</p>

---

## Getting Started

### Installation

1. **Clone** this repository:
   ```bash
   git clone https://github.com/YourUsername/DeepFingerprint-Refiner.git
   cd DeepFingerprint-Refiner
   ```
2. **Install Dependencies** (create a virtual environment if you prefer):
   ```bash
   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
   pip install numpy matplotlib opencv-python tqdm Pillow
   ```
   > Adjust the PyTorch version and CUDA version above (`cu118`) to match your system configuration.

3. (Optional) **Check GPU**:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should print True if GPU is accessible
   ```

### Dataset Structure

You need two directories:  
- **Clean dataset** (`CLEAN_DATASET_PATH`): Ground-truth, noiseless fingerprint images.  
- **Noisy dataset** (`NOISY_DATASET_PATH`): Noisy or partially corrupted fingerprint images.

Each directory should contain images in a consistent format (e.g., `.tif`, `.png`, or `.jpg`).  
Ensure that file naming conventions allow the script to pair noisy and clean images properly.

**Example folder structure**:
```
processed_dataset_final/
├── clean_dataset/
│   ├── 0001_clean.tif
│   ├── 0002_clean.tif
│   └── ...
└── mixed_noisy_dataset/
    ├── 0001_noisy_001.tif
    ├── 0002_noisy_002.tif
    └── ...
```
> The pairing logic in `FingerprintDataset` looks for filenames sharing certain segments.

### Running the Training Script

1. **Configure** paths and hyperparameters in `Train.py`:
   ```python
   # Paths & hyperparams in Train.py
   CLEAN_DATASET_PATH = r"D:\Deep learning\PROJECT\processed_dataset_final\clean_dataset"
   NOISY_DATASET_PATH = r"D:\Deep learning\PROJECT\processed_dataset_final\mixed_noisy_dataset"
   OUTPUT_DIR         = r"D:\Deep learning\PROJECT\FINAL-PROJECT\PROJECT-NEW-LATEST\FINAL"
   
   num_epochs         = 300
   batch_size         = 4
   learning_rate      = 2e-4
   lambda_L1          = 100
   ```
2. **Run Training**:
   ```bash
   python Train.py
   ```
3. **Check Logs & Samples**:
   - **Training progression** is displayed in the console (loss metrics).
   - **Sample images** are saved every 10 epochs in `OUTPUT_DIR/samples/`.

**When training completes**, the final model weights (`generator_final.pth` and `discriminator_final.pth`) will be saved in `OUTPUT_DIR`.

---

## Results

**Qualitative Examples** (from the saved `samples/epoch_X.png`):
1. **Noisy Input**: Original corrupted fingerprint.
2. **Lines Drawn On Top**: Generator output (Stage 2) with morphological bridging and endpoint connections.
3. **Ground Truth**: Clean dataset reference.

<p align="center">
  <img src="[https://user-images.githubusercontent.com/placeholder/results_example.png](https://github.com/HAMZA-420/DeepFingerprint-Refiner-on-FVC-2004-Datasets/blob/master/samples/epoch_190.png)" alt="Results Example" width="700">
</p>

---

## Project Structure

A typical layout of the repository might look like this:

```
DeepFingerprint-Refiner/
├── samples/                # Generated sample images during training
├── training_results/       # Additional logs/results
├── Validation/
│   ├── epoch_progression/  # Checkpoints / partial results over epochs
│   ├── Test_Results/
│   ├── eval.py             # Evaluation script
│   ├── new_final.py        # Possibly a refined version of test or evaluation
│   ├── test-data-new.py    # Data loader or test script
│   ├── testing.py          # Official testing script
│   ├── batch_losses.png    # Example of recorded losses
│   ├── loss_curves.png     # Visualization of generator/discriminator losses
│   └── ...
├── discriminator_epoch_270.pth  # Example of saved discriminator checkpoint
├── generator_epoch_270.pth      # Example of saved generator checkpoint
├── Train.py                # Main training script
├── Data.py                # Data script
├── README.md               # Project documentation (this file)
└── ...
```

Feel free to adjust or reorganize to fit your workflow.

---

## Usage & Examples

1. **Inference / Testing**:
   - Modify `testing.py` (or your chosen script) to load the saved weights:
     ```python
     generator = AdvancedDoubleStageGenerator()
     generator.load_state_dict(torch.load("path/to/generator_final.pth"))
     generator.eval()
     ...
     # Then run inference on new images
     ```

2. **Extend Post-Processing**:
   - You can further tweak morphological operations (e.g., kernel sizes, thinning parameters) in the code around:
     ```python
     morphological_bridging()
     thin_lines()
     connect_endpoints()
     ```

3. **Visualizing**:
   - Use `matplotlib`, `OpenCV`, or any library of your choice to monitor the generated fingerprint images.

---

## Contributing
Contributions, suggestions, and fixes are welcome!  
1. **Fork** the repository  
2. **Create** a new feature branch  
3. **Commit** your changes  
4. **Open** a Pull Request  

We appreciate all improvements, ranging from bug fixes to new functionalities.

---

## License
This project is licensed under the terms of the [MIT License](LICENSE). You are free to use, modify, and distribute this software, with attribution to this repository.

---

## Acknowledgments
- **PyTorch** for an elegant deep learning framework.
- **NVIDIA PartialConv Paper** for the original PartialConvolution approach:  
  *[“Image Inpainting for Irregular Holes Using Partial Convolutions” – Liu et al., ECCV 2018]*  
- **Self-Attention** references:  
  *[“Self-Attention Generative Adversarial Networks” – Zhang et al., ICML 2019]*

Special thanks to all open-source contributors and the research community working on image inpainting, GANs, and fingerprint enhancement.

---

> **Disclaimer**: This repository is intended for **research** and **educational** purposes. Performance on different datasets may vary. Always ensure compliance with applicable laws and regulations when working with fingerprint or biometric data.
