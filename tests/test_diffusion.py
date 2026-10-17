import torch

from src.image_gen.diffusion import LatentDiffusionUNet

def test_diffusion_unet_shape_and_finite_output():

    model = LatentDiffusionUNet(
        latent_channels=4,
        base_channels=32,
    )

    x = torch.randn(2,4,64,64)
    timesteps = torch.tensor([0,500])

    output = model(x,timesteps)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()

def test_diffusion_unet_backward():

    model = LatentDiffusionUNet(
        latent_channels=4,
        base_channels=32,
    )

    x = torch.randn(2,4,64,64)
    timesteps = torch.randint(0,100,(2,))

    prediction = model(x,timesteps)
    loss = prediction.square().mean()

    loss.backward()

    assert model.input_conv.weight.grad is not None
    assert torch.isfinite(
        model.input_conv.weight.grad
    ).all()

def test_diffusion_unet_rejects_invalid_timestep_shape():
    model = LatentDiffusionUNet(
        latent_channels=4,
        base_channels=32,
    )

    x = torch.randn(2,4,64,64)

    timesteps = torch.tensor([10])

    try:
        model(x,timesteps)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid timestep shape to fail")
