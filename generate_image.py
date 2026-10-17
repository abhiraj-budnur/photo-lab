from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.image_gen.autoencoder import ImageAutoEncoder
from src.image_gen.diffusion import LatentDiffusionUNet
from src.image_gen.noise_scheduler import DDPMScheduler

ROOT = Path(__file__).resolve().parent

DIFFUSION_CHECKPOINT = (
    ROOT / "checkpoints" / "diffusion" / "best.pt"
)

AUTOENCODER_CHECKPOINT = (
    ROOT / "checkpoints" / "image_autoencoder" / "best.pt"
)

OUTPUT_DIR = ROOT / "outputs" / "diffusion_samples"

NUM_IMAGES = 1
SEED = 42

def get_device():
    if torch.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.inference_mode()
def sample_latents(model,scheduler,num_images,device):
    """Generate normalized latent samples using reverse DDPM"""

    model.eval()

    sample = torch.randn(
        num_images,
        4,
        64,
        64,
        device=device,
        dtype = torch.float32,
    )

    for timestep in reversed(
        range(scheduler.num_train_timesteps)
    ):
        t = torch.full(
            (num_images,),
            timestep,
            device=device,
            dtype=torch.long,
        )

        predicted_noise = model(sample,t)

        beta_t = scheduler.betas[timestep]
        alpha_t = scheduler.alphas[timestep]
        alpha_bar_t = scheduler.alpha_cumprod[timestep]

        # DDPM reverse-process mean.
        sample = (
            sample
            - beta_t
            / torch.sqrt(1.0-alpha_bar_t)
            * predicted_noise
        ) / torch.sqrt(alpha_t)

        # Add posterior noise except at the final step.
        if timestep > 0:
            variance = scheduler.posterior_variance[timestep]
            sample = sample + (
                torch.sqrt(variance)
                * torch.randn_like(sample)
            )

        if timestep % 100 == 0:
            print(
                f"Reverse diffusion timestep: {timestep:04d}"
            )

    return sample

def save_image(image_tensor,path):
    """Save an RGB tensor in [-1,1] as a PNG"""

    image_tensor = image_tensor.detach().float().cpu()
    image_tensor = image_tensor.clamp(-1.0,1.0)

    image_tensor = (image_tensor + 1.0) / 2.0
    image_tensor = (
        image_tensor.permute(1,2,0) * 255.0
    ).round().byte()

    image_array = image_tensor.numpy()

    Image.fromarray(
        np.asarray(image_array,dtype=np.uint8)
    ).save(path)

def main():
    torch.manual_seed(SEED)

    device = get_device()
    print(f"Generation device: {device}")

    for path in (
        DIFFUSION_CHECKPOINT,
        AUTOENCODER_CHECKPOINT,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing Checkpoint: {path}"
            )

    diffusion_ckpt = torch.load(
        DIFFUSION_CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    autoencoder_ckpt = torch.load(
        AUTOENCODER_CHECKPOINT,
        map_location="cpu",
        weights_only=True
    )
    # Reconstruct the diffusion model using saved configuration.
    model_config = diffusion_ckpt["model_config"]

    model = LatentDiffusionUNet(
        latent_channels=model_config["latent_channels"],
        base_channels=model_config["base_channels"],
    )

    model.load_state_dict(diffusion_ckpt["model"])
    model = model.to(device).eval()

    # Reconstruct the trained autoencoder.
    ae_config = autoencoder_ckpt["config"]

    autoencoder = ImageAutoEncoder(
        latent_channels=ae_config["latent_channels"],
        base_channels=ae_config["base_channels"],
    )

    autoencoder.load_state_dict(autoencoder_ckpt["model"])
    autoencoder = autoencoder.to(device).eval()

    # Use the exact noise schedule from diffusion training.
    schedule_config = diffusion_ckpt["scheduler_config"]

    scheduler = DDPMScheduler(
        num_train_timesteps=schedule_config[
            "num_train_timesteps"
        ],
        beta_start=schedule_config["beta_start"],
        beta_end=schedule_config["beta_end"],
    ).to(device)

    # These statistics were saved with the diffusion checkpoint.
    latent_mean = diffusion_ckpt["latent_mean"].to(
        device=device,
        dtype=torch.float32,
    ).reshape(1, 4, 1, 1)

    latent_std = diffusion_ckpt["latent_std"].to(
        device=device,
        dtype=torch.float32,
    ).reshape(1, 4, 1, 1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"Using diffusion checkpoint from epoch "
        f"{diffusion_ckpt['epoch']}"
    )
    print("Starting reverse diffusion...")

    normalized_latents = sample_latents(
        model,
        scheduler,
        NUM_IMAGES,
        device,
    )

    # Undo normalization performed by train_diffusion.py.
    latents = normalized_latents * latent_std + latent_mean

    print(f"Generated latent shape: {tuple(latents.shape)}")

    reconstructed = autoencoder.decode(latents)

    if not torch.isfinite(reconstructed).all().item():
        raise RuntimeError(
            "Generated image contains non-finite values"
        )

    print(
        f"Generated image tensor shape: "
        f"{tuple(reconstructed.shape)}"
    )

    for index, image_tensor in enumerate(reconstructed):
        path = OUTPUT_DIR / f"generated_{index + 1:03d}.png"
        save_image(image_tensor, path)
        print(f"Saved image: {path}")

    print("Image generation complete.")


if __name__ == "__main__":
    main()