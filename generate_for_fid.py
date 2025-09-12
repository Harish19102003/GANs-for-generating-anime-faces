import os
import torch
from torchvision.utils import save_image
from torchvision import transforms
from pathlib import Path
from PIL import Image
from gan_anime_faces import Generator  # assumes Generator is in same repo

def load_gen(ckpt_path, nz=100, ngf=64, nc=3, device='cpu', use_self_attention=False):
    """Load generator from Lightning checkpoint or state dict"""
    g = Generator(nz=nz, ngf=ngf, nc=nc, use_self_attention=use_self_attention)
    state = torch.load(ckpt_path, map_location=device)

    # Lightning checkpoints have "state_dict"
    if "state_dict" in state:
        sd = {k.replace("netG.", ""): v for k, v in state["state_dict"].items() if k.startswith("netG.")}
        g.load_state_dict(sd, strict=False)
    else:
        g.load_state_dict(state)

    g.to(device).eval()
    return g


def generate_images(generator, out_dir, num_images=1000, nz=100, device='cpu', batch=64):
    """Generate fake images and save them"""
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    with torch.no_grad():
        while count < num_images:
            bs = min(batch, num_images - count)
            z = torch.randn(bs, nz, 1, 1, device=device)
            imgs = generator(z).cpu()  # in [-1,1]
            imgs = (imgs + 1) / 2      # to [0,1]
            for i in range(imgs.size(0)):
                save_image(imgs[i], os.path.join(out_dir, f"{count+i:06d}.png"))
            count += bs
    print(f"[INFO] Generated {num_images} images in {out_dir}")


def resize_real_images(real_dir, out_dir, size=(64, 64)):
    """Resize all real images to fixed size"""
    os.makedirs(out_dir, exist_ok=True)
    img_paths = list(Path(real_dir).glob("*"))
    for p in img_paths:
        try:
            img = Image.open(p).convert("RGB")
            img = img.resize(size, Image.BILINEAR)
            img.save(Path(out_dir) / p.name)
        except Exception as e:
            print(f"[WARN] Skipped {p}: {e}")
    print(f"[INFO] Resized {len(img_paths)} images into {out_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to generator checkpoint")
    parser.add_argument("--real_dir", required=True, help="Folder with real images")
    parser.add_argument("--real_resized", default="data/images_resized", help="Output folder for resized real images")
    parser.add_argument("--out", default="generated", help="Output folder for fake images")
    parser.add_argument("--num", type=int, default=1000, help="How many fake images to generate")
    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--nc", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Step 1: resize real images
    resize_real_images(args.real_dir, args.real_resized, size=(64, 64))

    # Step 2: load generator + generate fake images
    gen = load_gen(args.ckpt, nz=args.nz, ngf=args.ngf, nc=args.nc, device=args.device)
    generate_images(gen, args.out, num_images=args.num, nz=args.nz, device=args.device)
