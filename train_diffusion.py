from genericpath import isfile
from math import isfinite
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset

from src.image_gen.diffusion import LatentDiffusionUNet
from src.image_gen.noise_scheduler import DDPMScheduler

ROOT = Path(__file__).resolve().parent
LATENT_DIR = ROOT / "data" / "latents"
CHECKPOINT_DIR = ROOT / "checkpoints" / "diffusion"

TRAIN_LATENTS = LATENT_DIR / "train_latents.pt"
DEV_LATENTS = LATENT_DIR / "dev_latents.pt"
LATENTS_STATS = LATENT_DIR / "latents_stats.pt"

BATCH_SIZE = 4
BASE_CHANNELS = 32
NUM_TIMESTEPS = 1000
EPOCHS = 10
LEARNING_RATE = 1e-4
MAX_GRAD_NORM = 1.0
SEED = 42

# Set to true to continue from checkpoints/latest.pt
RESUME = False

def get_device():

    if torch.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def save_checkpoint_atomic(checkpoint,path):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp_path = path.with_suffix(path.suffix+".tmp")

    try:
        with open(temp_path,"wb") as f:
            torch.save(checkpoint,f)
            f.flush()
            os.fsync(f.fileno())

        # Verify the temporary checkpoint before replacing the old one.
        loaded = torch.load(
            temp_path,
            map_location="cpu",
            weights_only=True,
        )

        if "model" not in loaded or "optimizer" not in loaded:
            raise RuntimeError("Invalid diffusion checkpoint")

        os.replace(temp_path,path)
        print(f"Saved checkpoint: {path}")
    finally:
        if temp_path.exists():
            temp_path.unlink()

def load_latents(path):
    latents = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(latents,torch.Tensor):
        raise TypeError(f"Expected tensor in {path}")
    if latents.ndim != 4:
        raise ValueError(
            f"Expected [N,C,H,W] latents in {path}, "
            f"got {tuple(latents.shape)}"
        )

    if latents.shape[0] == 0:
        raise ValueError(f"No latents found in {path}")

    if not torch.isfinite(latents).all().item():
        raise RuntimeError(f"Non-finite latents in {path}")

    return latents.float().contiguous()

def normalize_latents(latents,mean,std):
    return(
        latents - mean[None,:,None,None]
    ) / std[None,:,None,None]

def make_checkpoint(
        model,
        optimizer,
        epoch,
        best_val_loss,
        mean,
        std,
):
    return {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "model_config": {
            "latent_channels": 4,
            "base_channels": BASE_CHANNELS,
        },
        "scheduler_config": {
            "num_train_timesteps": NUM_TIMESTEPS,
            "beta_start": 1e-4,
            "beta_end": 2e-2,
        },
        "latent_mean": mean.cpu(),
        "latent_std": std.cpu(),
        "training_config": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "seed": SEED,
        },
    }

@torch.no_grad()
def evaluate(model,scheduler,loader,device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for (clean,) in loader:
        clean = clean.to(device=device,dtype=torch.float32)
        batch_size = clean.shape[0]

        noise = torch.randn_like(clean)
        timesteps = torch.randint(
            0,
            scheduler.num_train_timesteps,
            (batch_size,),
            device=device,
            dtype=torch.long,
        )

        noisy = scheduler.add_noise(clean,noise,timesteps)
        predicted_noise = model(noisy,timesteps)

        loss = F.mse_loss(predicted_noise,noise)

        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite validation loss")

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    model.train()
    return total_loss / max(total_samples,1)

def main():

    random.seed(SEED)
    torch.manual_seed(SEED)

    device = get_device()
    print(f"Training device: {device}")

    for path in (TRAIN_LATENTS,DEV_LATENTS,LATENTS_STATS):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path}"
            )

    train_latents = load_latents(TRAIN_LATENTS)
    dev_latents = load_latents(DEV_LATENTS)

    if train_latents.shape[1:] != dev_latents.shape[1:]:
        raise ValueError("Train and dev latent shapes do not match")

    if train_latents.shape[1] != 4:
        raise ValueError("Expected 4-channel autoencoder latents")

    if train_latents.shape[-2:] != (64,64):
        raise ValueError("Expected 64x64 latent spatial dimension")

    stats = torch.load(
        LATENTS_STATS,
        map_location="cpu",
        weights_only=True,
    )

    mean = stats["mean"].float().reshape(-1)
    std = stats["std"].float().reshape(-1).clamp_min(1e-6)

    if mean.numel() != train_latents.shape[1]:
        raise ValueError("Latent mean has a invalid channel count")

    if std.numel() != train_latents.shape[1]:
        raise ValueError("Latent std has an invalid channel count")

    if not torch.isfinite(mean).all().item():
        raise RuntimeError("Invalid latent means")
    if(
        not torch.isfinite(std).all().item()
        or (std<=0).any().item()
    ):
        raise RuntimeError("Invalid latent standard deviations")

    train_latents = normalize_latents(train_latents,mean,std)
    dev_latents = normalize_latents(dev_latents,mean,std)

    print(f"Training latents: {tuple(train_latents.shape)}")
    print(f"Development latents: {tuple(dev_latents.shape)}")
    print(f"Batch size: {BATCH_SIZE}")

    train_loader = DataLoader(
        TensorDataset(train_latents),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    dev_loader = DataLoader(
        TensorDataset(dev_latents),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    model = LatentDiffusionUNet(
        latent_channels=4,
        base_channels=BASE_CHANNELS,
    ).to(device)

    scheduler = DDPMScheduler(
        num_train_timesteps=NUM_TIMESTEPS,
        beta_start=1e-4,
        beta_end=2e-2,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    start_epoch = 1
    best_val_loss = float("inf")

    latest_path = CHECKPOINT_DIR / "latest.pt"

    if RESUME and latest_path.exists():
        checkpoint = torch.load(
            latest_path,
            map_location="cpu",
            weights_only=True,
        )

        expected_config = {
            "latent_channels":4,
            "base_channels":BASE_CHANNELS,
        }

        if checkpoint["model_config"] != expected_config:
            raise ValueError("Checkpoint model configuration differs")

        expected_schedule = {
            "num_train_timesteps": NUM_TIMESTEPS,
            "beta_start": 1e-4,
            "beta_end": 2e-2,
        }

        if checkpoint["scheduler_config"] != expected_schedule:
            raise ValueError("Checkpoint noise schedule differs")

        if not torch.equal(checkpoint["latent_mean"], mean):
            raise ValueError("Checkpoint latent means differ")

        if not torch.equal(checkpoint["latent_std"], std):
            raise ValueError("Checkpoint latent stds differ")

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]

        print(f"Resuming from epoch {start_epoch}")

    CHECKPOINT_DIR.mkdir(parents=True,exist_ok=True)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Validation batches: {len(dev_loader)}")

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for step, (clean,) in enumerate(train_loader, start=1):
            clean = clean.to(device=device, dtype=torch.float32)
            batch_size = clean.shape[0]

            noise = torch.randn_like(clean)

            timesteps = torch.randint(
                0,
                scheduler.num_train_timesteps,
                (batch_size,),
                device=device,
                dtype=torch.long,
            )

            noisy = scheduler.add_noise(clean, noise, timesteps)
            predicted_noise = model(noisy, timesteps)

            loss = F.mse_loss(predicted_noise, noise)

            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch}, step {step}"
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                MAX_GRAD_NORM,
            )

            if not torch.isfinite(grad_norm).item():
                raise RuntimeError(
                    f"Non-finite gradient at epoch {epoch}, step {step}"
                )

            optimizer.step()

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            if step == 1 or step % 50 == 0:
                print(
                    f"epoch={epoch:03d} "
                    f"batch={step:04d}/{len(train_loader)} "
                    f"loss={loss.item():.5f}"
                )

        train_loss = total_loss / max(total_samples, 1)
        val_loss = evaluate(model, scheduler, dev_loader, device)

        improved = val_loss < best_val_loss

        if improved:
            best_val_loss = val_loss

        print(
            f"\nepoch={epoch:03d} "
            f"train_noise_mse={train_loss:.6f} "
            f"val_noise_mse={val_loss:.6f} "
            f"best_val_mse={best_val_loss:.6f}\n"
        )

        checkpoint = make_checkpoint(
            model,
            optimizer,
            epoch,
            best_val_loss,
            mean,
            std,
        )

        save_checkpoint_atomic(
            checkpoint,
            latest_path,
        )

        if improved:
            save_checkpoint_atomic(
                checkpoint,
                CHECKPOINT_DIR / "best.pt",
            )

    print(f"Best validation noise MSE: {best_val_loss:.6f}")
    print("Diffusion training complete.")


if __name__ == "__main__":
    main()