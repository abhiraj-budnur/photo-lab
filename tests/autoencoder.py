import torch
from src.image_gen.autoencoder import ImageAutoEncoder

def test_autoencoder_shapes_and_backward():
    model = ImageAutoEncoder(
        latent_channels=4,
        base_channels=16,
    )

    images = torch.randn(2,3,64,64).clamp(-1,1)
    result = model(images)

    assert result["latents"].shape == (2,4,8,8)
    assert result["reconstruction"].shape == images.shape
    assert torch.isfinite(result["loss"])

    result["loss"].backward()

    grads = [
        p.grad for p in model.parameters()
        if p.requires_grad
    ]

    assert all(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)

def test_autoencoder_supports_512_resolution():
    model = ImageAutoEncoder(
        latent_channels=4,
        base_channels=16,
    ).eval()

    images = torch.randn(1,3,512,512)

    with torch.no_grad():
        result = model(images)

    assert result["latents"].shape == (1,4,64,64)
    assert result["reconstruction"].shape == (1,3,512,512)
    assert torch.isfinite(result["reconstruction"]).all()