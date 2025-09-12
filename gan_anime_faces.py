"""
gan_anime_faces.py
Anime Face Generation using GANs with PyTorch Lightning

Features:
- DCGAN (BCE) training (default)
- Optional WGAN-GP training (`--use_wgangp`)
- Optional spectral normalization (`--use_spectral_norm`)
- Optional self-attention (`--use_self_attention`)
- Optional simple augmentations (`--augment`)
- Optional FID computation (`--compute_fid`) using torchmetrics
- Saves baseline real images, epoch_000 (initial fake), and per-epoch sample grids
"""

# ----------------------------
# Imports
# ----------------------------
import os
from pathlib import Path
import argparse
from typing import Optional, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Try importing FID (optional, requires torchmetrics)
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    _HAS_TORCHMETRICS = True
except Exception:
    _HAS_TORCHMETRICS = False


# ----------------------------
# Dataset
# ----------------------------
class ImageFolderSingle(Dataset):
    """Custom Dataset: loads images from a folder (no subfolders)."""
    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        exts = ["png", "jpg", "jpeg", "bmp"]  # valid extensions
        self.paths: List[Path] = []
        for e in exts:
            self.paths.extend(sorted(self.root.glob(f"*.{e}")))
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in {root}. Put image files directly inside this folder.")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # Open image and convert to RGB
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


class AnimeDataModule(pl.LightningDataModule):
    """Lightning DataModule: handles dataset + dataloaders."""
    def __init__(self, data_dir: str, img_size: int = 64, batch_size: int = 128, num_workers: int = 4, augment: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment = augment

        # Transformation pipeline (resize → crop → to tensor → normalize [-1,1])
        if self.augment:
            # With augmentations (flip + color jitter)
            self.transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.CenterCrop(self.img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.02, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3),
            ])
        else:
            # Without augmentations
            self.transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3),
            ])

    def setup(self, stage: Optional[str] = None):
        # Create dataset instance
        self.dataset = ImageFolderSingle(self.data_dir, transform=self.transform)

    def train_dataloader(self):
        # Return training DataLoader
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)


# ----------------------------
# Self-Attention (SAGAN-style)
# ----------------------------
class SelfAttention(nn.Module):
    """Self-Attention block: lets model focus on important regions globally."""
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # learnable scale
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, W, H = x.size()
        # Compute query, key, value matrices
        proj_query = self.query_conv(x).view(B, -1, W*H).permute(0,2,1)  # B x N x C'
        proj_key = self.key_conv(x).view(B, -1, W*H)                     # B x C' x N
        energy = torch.bmm(proj_query, proj_key)                        # B x N x N
        attention = self.softmax(energy)                                # Attention map
        proj_value = self.value_conv(x).view(B, -1, W*H)                 # B x C x N
        out = torch.bmm(proj_value, attention.permute(0,2,1))           # Apply attention
        out = out.view(B, C, W, H)
        out = self.gamma * out + x  # residual connection
        return out


# ----------------------------
# Generator (DCGAN style)
# ----------------------------
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3, use_self_attention=False):
        """
        nz: latent vector size
        ngf: feature maps in generator
        nc: output channels (3 for RGB)
        """
        super().__init__()
        self.use_self_attention = use_self_attention

        # Transposed convolutions gradually upsample noise → 64x64 image
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),   # 1x1 → 4x4
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False), # 4x4 → 8x8
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False), # 8x8 → 16x16
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),   # 16x16 → 32x32
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),      # 32x32 → 64x64
            nn.Tanh()  # output in [-1,1]
        )

        # Optional self-attention block at 16x16
        if self.use_self_attention:
            self.attn = SelfAttention(ngf*2)
        else:
            self.attn = None

    def forward(self, z):
        # Forward pass with optional attention
        x = self.main[0:3](z)   # 1x1 → 4x4
        x = self.main[3:6](x)   # 4x4 → 8x8
        x = self.main[6:9](x)   # 8x8 → 16x16
        if self.attn is not None:
            x = self.attn(x)    # apply attention at 16x16
        x = self.main[9:12](x)  # 16x16 → 32x32
        x = self.main[12:](x)   # 32x32 → 64x64
        return x


# ----------------------------
# Discriminator (DCGAN style)
# ----------------------------
class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64, use_spectral_norm=False, use_self_attention=False):
        """
        nc: input channels
        ndf: feature maps in discriminator
        """
        super().__init__()
        self.use_self_attention = use_self_attention

        # Helper for conv layers with optional spectral norm
        def conv(in_c, out_c, k, s, p, bias=False):
            layer = nn.Conv2d(in_c, out_c, k, s, p, bias=bias)
            if use_spectral_norm:
                return nn.utils.spectral_norm(layer)
            return layer

        # Downsampling conv layers
        self.conv1 = nn.Sequential(conv(nc, ndf, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True))
        self.conv2 = nn.Sequential(conv(ndf, ndf*2, 4, 2, 1), nn.BatchNorm2d(ndf*2), nn.LeakyReLU(0.2, inplace=True))
        self.conv3 = nn.Sequential(conv(ndf*2, ndf*4, 4, 2, 1), nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2, inplace=True))
        self.conv4 = nn.Sequential(conv(ndf*4, ndf*8, 4, 2, 1), nn.BatchNorm2d(ndf*8), nn.LeakyReLU(0.2, inplace=True))
        self.final_conv = nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False)  # output logits

        # Optional self-attention block at 16x16
        if self.use_self_attention:
            self.attn = SelfAttention(ndf*4)
        else:
            self.attn = None

    def forward(self, x):
        # Forward through layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.attn is not None:
            x = self.attn(x)
        x = self.conv4(x)
        x = self.final_conv(x)  # 1x1 output
        return x.view(-1)       # flatten (batch,)


# ----------------------------
# GAN LightningModule
# ----------------------------
class GAN(pl.LightningModule):
    def __init__(self, nz=100, ngf=64, ndf=64, nc=3,lr=2e-4, beta1=0.5,use_spectral_norm=False,use_self_attention=False, use_wgangp=False, gp_lambda=10.0, n_critic=5, compute_fid=False):
        """
        Wraps Generator + Discriminator + Loss + Optimizers.
        Supports DCGAN (BCE) and WGAN-GP training.
        """
        super().__init__()
        self.save_hyperparameters()

        # Networks
        self.netG = Generator(nz=nz,ngf=ngf,nc=nc,use_self_attention=use_self_attention)
        self.netD = Discriminator(nc=nc,ndf=ndf,use_spectral_norm=use_spectral_norm,use_self_attention=use_self_attention)

        # Loss for DCGAN (more stable than BCE+Sigmoid)
        if not self.hparams.use_wgangp:
            self.criterion = nn.BCEWithLogitsLoss()

        # Fixed noise for visualizing progress
        self.fixed_noise = torch.randn(64, self.hparams.nz, 1, 1)

        # Manual optimization (we control when to update G vs D)
        self.automatic_optimization = False

        # FID metric (optional)
        self.compute_fid = compute_fid and _HAS_TORCHMETRICS
        if self.compute_fid:
            self.fid = FrechetInceptionDistance(feature=2048)

    def configure_optimizers(self):
        # Separate Adam optimizers for D and G
        optD = torch.optim.Adam(self.netD.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))
        optG = torch.optim.Adam(self.netG.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))
        return [optD, optG], []

    def gradient_penalty(self, real, fake):
        """WGAN-GP gradient penalty."""
        alpha = torch.rand(real.size(0), 1, 1, 1, device=real.device)
        interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        d_interpolates = self.netD(interpolates)
        grads = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones_like(d_interpolates),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
        grads = grads.view(grads.size(0), -1)
        gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
        return gp

    def training_step(self, batch, batch_idx):
        # Get optimizers
        optD, optG = self.optimizers()
        real_imgs = batch
        batch_size = real_imgs.size(0)
        device = real_imgs.device

        # -----------------
        # 1) Discriminator update
        # -----------------
        optD.zero_grad()
        z = torch.randn(batch_size, self.hparams.nz, 1, 1, device=device)
        fake_imgs = self.netG(z)

        d_real = self.netD(real_imgs)
        d_fake = self.netD(fake_imgs.detach())

        if self.hparams.use_wgangp:
            # WGAN-GP loss
            d_loss = -(d_real.mean() - d_fake.mean())
            gp = self.gradient_penalty(real_imgs, fake_imgs.detach())
            d_loss += self.hparams.gp_lambda * gp
        else:
            # DCGAN BCE loss with label smoothing
            real_labels = torch.full((batch_size,), 0.9, device=device)
            fake_labels = torch.zeros(batch_size, device=device)
            d_loss_real = self.criterion(d_real, real_labels)
            d_loss_fake = self.criterion(d_fake, fake_labels)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

        self.manual_backward(d_loss)
        optD.step()

        # -----------------
        # 2) Generator update
        # -----------------
        do_g_update = True
        if self.hparams.use_wgangp:
            # For WGAN-GP, update G every n_critic steps
            do_g_update = (self.global_step % self.hparams.n_critic) == 0

        g_loss = torch.tensor(0.0, device=device)
        if do_g_update:
            optG.zero_grad()
            z2 = torch.randn(batch_size, self.hparams.nz, 1, 1, device=device)
            gen_imgs = self.netG(z2)
            d_gen = self.netD(gen_imgs)
            if self.hparams.use_wgangp:
                g_loss = -d_gen.mean()
            else:
                # G wants D(fake) ≈ real (1.0)
                target_labels = torch.full((batch_size,), 0.9, device=device)
                g_loss = self.criterion(d_gen, target_labels)
            self.manual_backward(g_loss)
            optG.step()

        # Logging
        self.log("loss/d", d_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("loss/g", g_loss, prog_bar=True, on_step=True, on_epoch=True)

        # FID accumulation (optional)
        if self.compute_fid:
            with torch.no_grad():
                real_for_fid = (real_imgs + 1) / 2
                fake_for_fid = (fake_imgs + 1) / 2
                self.fid.update(real_for_fid, real=True)
                self.fid.update(fake_for_fid, real=False)

    def on_train_start(self):
        # Save initial fake images before training
        os.makedirs("samples", exist_ok=True)
        z = self.fixed_noise.to(self.device)
        with torch.no_grad():
            fake_imgs = self.netG(z).cpu()
        grid = make_grid(fake_imgs, nrow=8, normalize=True)
        save_image(grid, "samples/epoch_000.png")

    def on_train_epoch_end(self):
        # Save fixed-noise samples every epoch
        z = self.fixed_noise.to(self.device)
        with torch.no_grad():
            samples = self.netG(z).cpu()
        grid = make_grid(samples, nrow=8, normalize=True)
        save_image(grid, f"samples/epoch_{self.current_epoch+1:03d}.png")

        # Compute FID if enabled
        if self.compute_fid:
            fid_val = self.fid.compute()
            self.log("metrics/fid", fid_val, prog_bar=True)
            self.fid.reset()


# ----------------------------
# Main function
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Anime Face GAN")
    # Dataset args
    parser.add_argument('--data_dir', type=str, required=True, help='path to images folder')
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)

    # Model hyperparams
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.5)

    # Training config
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--gpus', type=int, default=0, help='0 = CPU, >0 = number of GPUs')
    parser.add_argument('--precision', type=int, default=32, choices=[16, 32])

    # Extra flags
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--use_spectral_norm', action='store_true')
    parser.add_argument('--use_self_attention', action='store_true')
    parser.add_argument('--use_wgangp', action='store_true')
    parser.add_argument('--gp_lambda', type=float, default=10.0)
    parser.add_argument('--n_critic', type=int, default=5)
    parser.add_argument('--compute_fid', action='store_true')

    args = parser.parse_args()

    # Data
    dm = AnimeDataModule(data_dir=args.data_dir, img_size=args.img_size,
                         batch_size=args.batch_size, num_workers=args.num_workers,
                         augment=args.augment)
    dm.setup()

    # Save grid of real images (baseline)
    os.makedirs("samples", exist_ok=True)
    real_batch = next(iter(dm.train_dataloader()))
    save_image(make_grid(real_batch[:64], nrow=8, normalize=True), "samples/real_images.png")

    # Model
    model = GAN(nz=args.nz, ngf=args.ngf, ndf=args.ndf, nc=3,
                lr=args.lr, beta1=args.beta1,
                use_spectral_norm=args.use_spectral_norm,
                use_self_attention=args.use_self_attention,
                use_wgangp=args.use_wgangp, gp_lambda=args.gp_lambda,
                n_critic=args.n_critic, compute_fid=args.compute_fid)

    # Logging + Checkpointing
    logger = TensorBoardLogger("tb_logs", name="anime_gan")
    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints", filename="epoch-{epoch:03d}", save_top_k=-1,auto_insert_metric_name=False )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Trainer
    trainer = pl.Trainer(max_epochs=args.max_epochs,
                         accelerator="gpu" if args.gpus > 0 and torch.cuda.is_available() else "cpu",
                         devices=args.gpus if args.gpus > 0 else None,
                         precision=args.precision,
                         callbacks=[checkpoint_callback, lr_monitor],
                         logger=logger,
                         log_every_n_steps=50)

    # Train
    trainer.fit(model, dm)


if __name__ == "__main__": main()
