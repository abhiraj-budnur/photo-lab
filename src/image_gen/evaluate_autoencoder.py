
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from src.image_gen.autoencoder import ImageAutoEncoder
from src.image_gen.data import CaptionImageDataset


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "checkpoints" / "image_autoencoder" / "best.pt"
OUTPUT = ROOT / "outputs" / "autoencoder" / "validation_grid.png"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def main():
    device = get_device()

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
    split = dataset["dev"].select(range(min(16, len(dataset["dev"]))))
    loader = DataLoader(
        CaptionImageDataset(split, image_size=512),
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    grid_items = []
    total_l1 = 0.0
    total_images = 0

    for batch in loader:
        images = batch["pixel_values"].to(device)
        result = model(images)
        reconstructions = result["reconstruction"]

        total_l1 += (
            (reconstructions - images).abs()
            .mean(dim=(1, 2, 3))
            .sum()
            .item()
        )
        total_images += images.shape[0]

        for original, reconstructed in zip(images, reconstructions):
            grid_items.extend([
                original.detach().cpu(),
                reconstructed.detach().cpu(),
            ])

    grid = torch.stack(grid_items)
    grid = (grid.clamp(-1, 1) + 1) / 2

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, OUTPUT, nrow=4, padding=4)

    print(f"Mean validation L1: {total_l1 / total_images:.4f}")
    print(f"Saved comparison grid: {OUTPUT}")


if __name__ == "__main__":
    main()
