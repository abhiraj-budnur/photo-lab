import os
import random
from pathlib import Path
from unittest import result

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from PIL import Image

from src.image_gen.autoencoder import ImageAutoEncoder
from src.image_gen.data import CaptionImageDataset

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = ROOT / "checkpoins"/"image_autoencoder"
PREVIEW_DIR = ROOT / "outputs"/ "autoencoder"
CHECKPOINT_DIR.mkdir(parents=True,exist_ok=True)
PREVIEW_DIR.mkdir(parents=True,exist_ok=True)

SEED = 42
IMAGE_SIZE = 512
BATCH_SIZE = 2
EPOCHS = 10
LEARNING_RATE = 1e-4
MAX_TRAIN_IMAGES = 1000
MAX_VAL_IMAGES = 100

def get_device():
    if torch.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def save_checkpoint_atomic(state,path):
    path = Path(path)
    temporary_path = path.with_suffix(".pt.tmp")

    try:
        with open(temporary_path,"wb") as file:
            torch.save(state,file)
            file.flush()
            os.fsync(file.fileno())

            torch.load(
                temporary_path,
                map_location="cpu",
                weights_only=True,
            )
            os.replace(temporary_path,path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

def save_preview(original, reconstruction,epoch):
    # Use the first image in the batch
    pair = torch.cat(
        [original[0],reconstruction[0]],dim=2
    ).detach().float().cpu()

    pair = ((pair.clamp(-1,1)+1)*127.5)
    pair = pair.round().byte().permute(1,2,0).numpy()

    Image.fromarray(pair).save(
        PREVIEW_DIR / f"epoch_ {epoch:03d}.png"
    )

@torch.no_grad()
def evaluate(model,loader,device):
    model.eval()
    total_loss = 0.0
    total_images = 0
    preview = None

    for batch in loader:
        images = batch["pixel_values"].to(device)
        result = model(images)

        count = images.shape[0]
        total_loss += result["loss"].item() * count
        total_images += count

        if preview is None:
            preview = (
                images[:1].detach().cpu(),
                result["reconstruction"][:1].detach().cpu(),
            )

    if total_images == 0:
        raise RuntimeError("Validation dataset is empty")

    return total_loss / total_images,preview

def main():

    random.seed(SEED)
    torch.manual_seed(SEED)

    device = get_device()
    print(f"Training device: {device}")

    print("Loading Flickr8k from the hugging face cache...")
    dataset = load_dataset("intro/flickr8k")

    if "train" not in dataset:
        raise RuntimeError(f"Expected a train split; found {list(dataset.keys())}")

    # prefer the validation split; use test only if validation is absent
    available_splits = list(dataset.keys())

    print(f"Available dataset splits: {available_splits}")

    if "dev" in available_splits:
        val_name = "dev"
    elif "validation" in available_splits:
        val_name = "validation"
    else:
        raise RuntimeError(
            f"No development/validation split found. "
            f"Available splits: {available_splits}"
        )

    train_split = dataset["train"]
    val_split = dataset[val_name]

    print(f"Training split: train ({len(train_split):,} images)")
    print(f"Validation split: {val_name} ({len(val_split):,} images)")

    train_split = dataset["train"]
    val_split = dataset[val_name]

    # Keep the first run small and reproducible
    train_count = min(MAX_TRAIN_IMAGES,len(train_split))
    val_count = min(MAX_VAL_IMAGES,len(val_split))

    train_split = train_split.select(range(train_count))
    val_split = val_split.select(range(val_count))

    train_data = CaptionImageDataset(
        train_split,image_size=IMAGE_SIZE
    )
    val_data = CaptionImageDataset(
        val_split,image_size=IMAGE_SIZE
    )

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = ImageAutoEncoder(
        latent_channels=4,
        base_channels=16,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = LEARNING_RATE,
        weight_decay=1e-4,
    )

    best_val_loss = float("inf")

    print(f"Training images: {len(train_data)}")
    print(f"Validation images: {len(val_data)}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1,EPOCHS+1):
        model.train()
        total_loss = 0.0
        total_images = 0

        for step,batch in enumerate(train_loader,start=1):
            images = batch["pixel_values"].to(device)

            optimizer.zero_grad(set_to_none=True)
            result = model(images)
            loss = result["loss"]

            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch}, step {step}"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),max_norm=1.0
            )
            optimizer.step()

            count = images.shape[0]
            total_loss += loss.item() * count
            total_images += count

            if step % 25 == 0 or step == len(train_loader):
                print(
                    f"epoch={epoch:02d}/{EPOCHS} "
                    f"step={step:04d}/{len(train_loader)} "
                    f"batch_l1={loss.item():.4f}"
                )

        train_loss = total_loss / total_images
        val_loss,preview = evaluate(model,val_loader,device)

        save_preview(preview[0],preview[1],epoch)

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": {
                "latent_channels": 4,
                "base_channels": 16,
                "image_size": IMAGE_SIZE,
            },
            "train_l1": train_loss,
            "val_l1": val_loss,
        }

        save_checkpoint_atomic(
            checkpoint,CHECKPOINT_DIR/"latest.pt"
        )

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            save_checkpoint_atomic(
            checkpoint,CHECKPOINT_DIR/"best.pt"
            )

        print(
            f"epoch={epoch:02d} "
            f"train_l1={train_loss:.4f} "
            f"val_l1={val_loss:.4f} "
            f"best_val_l1={best_val_loss:.4f}"
        )

    print("Training complete")
    print(f"Checkpoints: {CHECKPOINT_DIR}")
    print(f"Previews: {PREVIEW_DIR}")

if __name__ == "__main__":
    main()
