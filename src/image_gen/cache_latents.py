import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from src.image_gen.autoencoder import ImageAutoEncoder
from src.image_gen.data import CaptionImageDataset

ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT = ROOT / "checkpoints" / "image_autoencoder" / "best.pt"
OUTPUT_DIR = ROOT / "data" / "latents"

IMAGE_SIZE = 512
MAX_TRAIN_IMAGES = 1000
MAX_DEV_IMAGES = 100
BATCH_SIZE = 2

def get_device():

    if torch.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def save_atomic(obj,path):

    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")

    try:
        with open(temp_path,"wb") as f:
            torch.save(obj,f)
            f.flush()
            os.fsync(f.fileno())

            torch.load(
                temp_path,
                map_location="cpu",
                weights_only=True
            )
            os.replace(temp_path,path)

    finally:
        if temp_path.exists():
            temp_path.unlink()

@torch.inference_mode()
def encode_split(model,split,count,device,name):
    count = min(count,len(split))
    print(f"[DEBUG] Requested count: {count}")
    print(f"[DEBUG] Split length before select: {len(split)}")
    split = split.select(range(count))
    print(f"[DEBUG] Split length after select: {len(split)}")

    image_data = CaptionImageDataset(
        split,
        image_size=IMAGE_SIZE,
    )

    loader = DataLoader(
        image_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(f"[DEBUG] Dataset wrapper length: {len(image_data)}")
    print(f"[DEBUG] DataLoader batch size: {loader.batch_size}")
    print(f"[DEBUG] Expected batches: {len(loader)}")

    latent_batches = []

    for step,batch in enumerate(loader,start=1):
        images = batch["pixel_values"].to(device)
        latents = model.encoder(images)

        if not torch.isfinite(latents).all().item():
            raise RuntimeError(
                f"Non-finite latents in {name}, batch {step}"
            )

        # Store compact CPU tensors; convert back to FP32 for training.
        latent_batches.append(
            latents.detach().to(
                device="cpu",
                dtype=torch.float16,
            )
        )

        if step <= 3 or step % 25 == 0 or step == len(loader):
            print(
                f"{name}: batch {step}/{len(loader)}, "
                f"images in batch={images.shape[0]}, "
                f"latents accumulated={sum(t.shape[0] for t in latent_batches)}"
            )

    if not latent_batches:
        raise RuntimeError(
            f"No latents generated for {name}"
        )

    latents = torch.cat(latent_batches,dim=0)
    if latents.shape[0] != count:
        raise RuntimeError(
            f"{name}: expected {count} latents, "
            f"but encoded {latents.shape[0]}"
        )
    print(f"{name} latent shape: {tuple(latents.shape)}")
    return latents

def main():

    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

    device = get_device()
    print(f"Encoding device: {device}")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    config = checkpoint["config"]

    model = ImageAutoEncoder(
        latent_channels = config["latent_channels"],
        base_channels = config["base_channels"],
    )

    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    dataset = load_dataset("intro/flickr8k")
    print("Dataset splits:", list(dataset.keys()))

    for split_name in ("train", "dev"):
        split = dataset[split_name]
        print(f"\n{split_name}: {len(split)} rows")
        print("Columns:", split.column_names)
        print("First image type:", type(split[0]["image"]))

        wrapped = CaptionImageDataset(
            split.select(range(min(5, len(split)))),
            image_size=IMAGE_SIZE,
        )
        print("Wrapper length for 5 rows:", len(wrapped))

    if "train" not in dataset or "dev" not in dataset:
        raise RuntimeError(
            f"Expected train and dev splits, got {list(dataset.keys())}"
        )

    train_latents = encode_split(
        model,
        dataset["train"],
        MAX_TRAIN_IMAGES,
        device,
        "train",
    )

    dev_latents = encode_split(
        model,
        dataset["dev"],
        MAX_DEV_IMAGES,
        device,
        "dev",
    )

    train_fp32 = train_latents.float()

    channel_mean = train_fp32.mean(dim=(0,2,3))
    channel_std = train_fp32.std(
        dim=(0,2,3),
        unbiased=False,
    ).clamp_min(1e-6)

    if not torch.isfinite(channel_mean).all():
        raise RuntimeError(
            "Invalid latent channel means"
        )

    if not torch.isfinite(channel_std).all():
        raise RuntimeError(
            "Invalid latent channel standard deviations"
        )

    if train_latents.shape[1] != config["latent_channels"]:
        raise RuntimeError(
            "Latent channel count differs from checkpoint"
        )

    save_atomic(
        train_latents,
        OUTPUT_DIR / "train_latents.pt",
    )

    save_atomic(
        dev_latents,
        OUTPUT_DIR / "dev_latents.pt",
    )
    save_atomic(
        {
            "mean":channel_mean,
            "std":channel_std,
            "latent_channels":config["latent_channels"],
            "latent_size":IMAGE_SIZE // 8,
            "image_size":IMAGE_SIZE,
            "autoencoder_checkpoint":str(CHECKPOINT),
        },
        OUTPUT_DIR / "latents_stats.pt",
    )

    print("\nLatent cache complete.")
    print(f"Training latents: {tuple(train_latents.shape)}")
    print(f"Development latents: {tuple(dev_latents.shape)}")
    print(f"Channel means: {channel_mean.tolist()}")
    print(f"Channel stds: {channel_std.tolist()}")
    print(f"Saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()