from pathlib import Path

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from src.image_gen.data import CaptionImageDataset
from src.image_gen.autoencoder import ImageAutoEncoder

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "checkpoints" / "image_autoencoder" / "best.pt"
OUTPUT_DIR = ROOT / "outputs" / "autoencoder"
OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

IMAGE_SIZE = 512
MAX_IMAGES = 1000
BATCH_SIZE = 2

def get_device():

    if torch.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

@torch.no_grad()
def main():

    device = get_device()
    print(f"Validation device: {device}")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    config = checkpoint["config"]
    model = ImageAutoEncoder(
        latent_channels=config["latent_channels"],
        base_channels=config["base_channels"],
    )

    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    dataset = load_dataset("intro/flickr8k")
    dev_split = dataset["dev"]

    count = min(MAX_IMAGES,len(dev_split))
    dev_split = dev_split.select(range(count))

    dev_data = CaptionImageDataset(
        dev_split,
        image_size=IMAGE_SIZE,
    )

    loader = DataLoader(
        dev_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    total_abs_zero = 0.0
    total_squared_error = 0.0
    total_pixels = 0

    latent_sum = 0.0
    latent_squared_sum = 0.0
    latent_count = 0
    latent_min = float("inf")
    latent_max = float("-inf")
    batches = 0

    for batch in loader:
        images = batch["pixel_values"].to(device)

        # Call the encoder directly.
        z = model.encode(images)

        # Run the full autoencoder forward pass.
        result = model(images)
        reconstruction = result["reconstruction"]

        print(f"Input:          {tuple(images.shape)}")
        print(f"Encoded latent: {tuple(z.shape)}")
        print(f"Forward latent: {tuple(result['latents'].shape)}")
        print(f"Reconstruction: {tuple(reconstruction.shape)}")

        expected_shape = (
            images.shape[0],
            config["latent_channels"],
            IMAGE_SIZE // 8,
            IMAGE_SIZE // 8,
        )

        if tuple(z.shape) != expected_shape:
            raise RuntimeError(
                f"Expected latent {expected_shape}, got {tuple(z.shape)}"
            )

        if not torch.isfinite(z).all().item():
            raise RuntimeError("Non-finite latent values")

        if not torch.isfinite(reconstruction).all().item():
            raise RuntimeError("Non-finite reconstruction values")

        # Accumalate statistics in FP64 for reliable reporting
        z64 = z.float()
        latent_sum += z64.sum().item()
        latent_squared_sum += z64.square().sum().item()
        latent_count += z.numel()
        latent_min = min(latent_min,z.min().item())
        latent_max = max(latent_max,z.max().item())

        error = reconstruction.float() - images.float()
        total_abs_zero += error.abs().sum().item()
        total_squared_error += error.square().sum().item()
        total_pixels += error.numel()
        batches+=1

        if batches == 1:
            print(f"Input shape:           {tuple(images.shape)}")
            print(f"Latent shape:          {tuple(images.shape)}")
            print(f"Reconstruction shape:  {tuple(images.shape)}")

    if total_pixels == 0 or latent_count == 0:
        raise RuntimeError("No images were evaluated")

    latent_mean = latent_sum / latent_count
    latent_variance = max(
        latent_squared_sum / latent_count - latent_mean**0.5,0.0
    )

    latent_std = latent_variance**0.5

    #images are expected to be normalized to [-1,1]
    mae = total_abs_zero / total_pixels
    rmse = (total_squared_error / total_pixels) ** 0.5

    print("\n--- Reconstruction metrics ---")
    print(f"Images evaluated: {count}")
    print(f"MAE ([-1, 1] scale): {mae:.6f}")
    print(f"RMSE ([-1, 1] scale): {rmse:.6f}")

    print("\n--- Latent statistics ---")
    print(f"Mean: {latent_mean:.6f}")
    print(f"Std:  {latent_std:.6f}")
    print(f"Min:  {latent_min:.6f}")
    print(f"Max:  {latent_max:.6f}")

    print("\nValidation complete.")

if __name__ == "__main__":
    main()