# Conditional DDPM for Image Denoising

This project implements a Conditional Denoising Diffusion Probabilistic Model (DDPM) for image denoising tasks using PyTorch. It is designed to learn the mapping from noisy or low-quality images (condition) to clean ground truth images.

## Project Structure

```
diffusion/
├── config/             # Configuration files
├── core/               # Core utilities (logger, metrics)
├── model/              # Model definitions
│   ├── conditional_ddpm_modules/
│   │   ├── diffusion.py    # Main DDPM logic
│   │   └── unet.py         # U-Net architecture
│   ├── base_model.py
│   ├── model.py        # Model wrapper
│   └── networks.py     # Network initialization
├── dataset.py          # Dataset loading and augmentation
├── train.py            # Training and evaluation script
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Requirements

Install the required dependencies:

```bash
pip install -r requirement.txt
```

Common dependencies include:
- torch
- torchvision
- numpy
- tqdm
- tensorboardX
- wandb (optional, for logging)
- Pillow

## Dataset Preparation

The dataset is managed via text files listing the image filenames.

1.  **Data Organization**:
    Place your Ground Truth (GT) and Condition (Noisy) images in corresponding directories.
    
    Example:
    - `dataset/gt_images/`
    - `dataset/cond_images/`

2.  **File Lists**:
    Create `train.txt`, `val.txt`, and `test.txt` containing the filenames (e.g., `image_001.png`) to be used for each phase. One filename per line.

3.  **Configuration**:
    Update `config/config.json` with the paths to your data:
    ```json
    "datasets": {
        "train": {
            "gt_dataset_path": "/path/to/gt_images",
            "cond_dataset_path": "/path/to/cond_images",
            "dataroot": "train.txt",
            ...
        },
        "val": { ... },
        "test": { ... }
    }
    ```

## Configuration

The model and training parameters are defined in `config/config.json`. Key parameters include:

-   `model`: Defines U-Net structure (channels, attention, etc.) and diffusion beta schedule.
-   `train`: Training iterations, learning rate, validation frequency.
-   `wandb`: WandB project name (optional).

## Training

To start training the model:

```bash
python train.py --config config/config.json --phase train
```

**Arguments:**
-   `--config`: Path to the configuration file.
-   `--phase`: `train` or `val`.
-   `--gpu_ids`: Specify GPU IDs (e.g., `0` or `0,1`).
-   `--debug`: Enable debug mode (runs fewer iterations for testing).
-   `-enable_wandb`: Enable Weights & Biases logging.

## Evaluation

To evaluate the model using the `test` dataset defined in config:

```bash
python train.py --config config/config.json --phase val
```

This will load the resume state defined in `config.json` (under `path.resume_state`) or the latest checkpoint if configured, and generate results in the `results/` directory.

## Key Implementation Details

-   **Network**: A U-Net with attention mechanisms and time embedding injection.
-   **Diffusion**: Gaussian Diffusion with linear beta schedule.
-   **Conditioning**: The condition image is concatenated with the noisy input at the channel dimension (Input channels = 6 -> Output channels = 3).

## Fixes & Updates

Recent updates addressed the following issues in `diffusion.py`:
1.  **Corrected `predict_start_from_noise`**: Fixed the mathematical formula for reconstructing $x_0$ from $x_t$ and noise.
2.  **Fixed Conditional Input Handling**: Resolved an issue where the concatenated input (6 channels) was incorrectly passed to functions expecting only the latent state (3 channels), causing shape mismatches.
3.  **Variable Naming**: improved clarity in sampling loops.

