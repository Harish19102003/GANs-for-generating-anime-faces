# GANs-for-generating-anime-faces 

This project implements **Generative Adversarial Networks (GANs)** using **PyTorch Lightning** to generate **anime face images**.

It includes:
- DCGAN (default) and WGAN-GP options
- Optional self-attention (SAGAN-style)
- Optional spectral normalization
- Optional data augmentation
- Optional FID evaluation

---

##  Introduction

We train a GAN on the **Anime Face Dataset** (Kaggle) to generate realistic anime-style faces.

---

##  Objective

- Implement and train a GAN capable of generating 64×64 anime face images.
- Explore different training tricks: WGAN-GP, spectral norm, self-attention.
- Evaluate results visually and (optionally) with FID metric.

---

##  Dataset

- **Name:** Anime Face Dataset  
- **Source:** [Kaggle Dataset Link](https://www.kaggle.com/splcher/animefacedataset)  
- **Description:** Contains thousands of cropped anime face images suitable for generative tasks.

---

##  Project Structure

```
GANs-for-generating-anime-faces/
├── gan_anime_faces.py     # Main GAN training script
├── README.md              # Documentation + Report 
├── requirements.txt       # Dependencies
├── .gitignore             # Ignore cache/checkpoint/log files
├── samples/               # Saved real + generated images per epoch
├── checkpoints/           # Model checkpoints
└── tb_logs/               # TensorBoard logs
```

---

##  Usage

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Download Dataset
```bash
kaggle datasets download -d splcher/animefacedataset
unzip animefacedataset.zip -d data/images
```

# Train GAN

## Default (DCGAN):
```bash
python gan_anime_faces.py --data_dir data/images --batch_size 128 --max_epochs 50 --gpus 1
```

## With WGAN-GP:
```bash
python gan_anime_faces.py --data_dir data/images --use_wgangp --max_epochs 50 --gpus 1
```

## With self-attention + spectral norm:
```bash
python gan_anime_faces.py --data_dir data/images --use_self_attention --use_spectral_norm
```

## With FID:
```bash
python gan_anime_faces.py --data_dir data/images --compute_fid --gpus 1
```
## View Results

Generated images → samples/

Real samples → samples/real_images.png

TensorBoard logs → tb_logs/

## Run:
```bash
tensorboard --logdir tb_logs
```

# Evaluation
Using FID

## Generate fake images:
```bash
python generate_for_fid.py --ckpt checkpoints/epoch-049.ckpt --real_dir data/images --out generated
```

## Run FID:
```bash
pytorch-fid data/images_resized generated
```
# Report Section
## Training Setup

Dataset: [Anime Face Dataset Kaggle](https://www.kaggle.com/splcher/animefacedataset)

Image size: 64×64

Latent dimension (nz): 100

Batch size: 128

Optimizer: Adam (lr=0.0002, β1=0.5)

Epochs: 50

## Model variants tested:

DCGAN baseline

WGAN-GP

Self-Attention

Spectral Norm

Results at Epoch 49

***FID score: 51.94***

## Samples:

samples/epoch_000.png → noisy blobs (untrained generator)

samples/epoch_010.png → blurry but facial structure forming

samples/epoch_030.png → recognizable anime faces

samples/epoch_049.png → clearer faces, some artifacts remain

## Observations

Training was stable with DCGAN setup.

Generator outputs improved steadily with epochs.

FID ~52 indicates moderate quality (faces are recognizable but not sharp).

Adding WGAN-GP + self-attention may further improve quality.

## Challenges

Mode collapse risk (generator repeating similar faces).

Dataset has small resolution (64×64), limiting detail.

FID calculation failed initially due to inconsistent image sizes → fixed by resizing all real images.

## Future Work

Train longer (100–200 epochs).

Enable WGAN-GP + spectral norm + self-attention for stability and global coherence.

Experiment with higher resolution (128×128).

Evaluate with more samples (10k+) for better FID estimation.