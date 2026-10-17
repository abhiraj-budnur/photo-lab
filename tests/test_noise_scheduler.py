import pytest
import torch

from src.image_gen.noise_scheduler import DDPMScheduler


def test_scheduler_add_noise_shape_and_finite():
    scheduler = DDPMScheduler()
    clean = torch.randn(2, 4, 64, 64)
    noise = torch.randn_like(clean)
    timesteps = torch.tensor([0, 500])

    noisy = scheduler.add_noise(clean, noise, timesteps)

    assert noisy.shape == clean.shape
    assert torch.isfinite(noisy).all()


def test_scheduler_recovers_clean_sample():
    scheduler = DDPMScheduler()

    clean = torch.randn(2, 4, 64, 64)
    noise = torch.randn_like(clean)
    timesteps = torch.tensor([10, 700])

    noisy = scheduler.add_noise(clean, noise, timesteps)

    recovered = scheduler.predict_clean_sample(
        noisy,
        noise,
        timesteps,
    )

    torch.testing.assert_close(
        recovered,
        clean,
        rtol=1e-4,
        atol=1e-4,
    )


def test_scheduler_rejects_invalid_timestep_shape():
    scheduler = DDPMScheduler()
    clean = torch.randn(2, 4, 64, 64)
    noise = torch.randn_like(clean)

    with pytest.raises(ValueError):
        scheduler.add_noise(
            clean,
            noise,
            torch.tensor([10]),
        )


def test_scheduler_rejects_out_of_range_timestep():
    scheduler = DDPMScheduler()
    clean = torch.randn(2, 4, 64, 64)
    noise = torch.randn_like(clean)

    with pytest.raises(ValueError):
        scheduler.add_noise(
            clean,
            noise,
            torch.tensor([10, 1000]),
        )


def test_scheduler_buffers_are_finite():
    scheduler = DDPMScheduler()

    assert torch.isfinite(scheduler.betas).all()
    assert torch.isfinite(scheduler.alpha_cumprod).all()
    assert torch.isfinite(scheduler.posterior_variance).all()

    assert (scheduler.betas > 0).all()
    assert (scheduler.betas < 1).all()
    assert (
        scheduler.alpha_cumprod[1:]
        < scheduler.alpha_cumprod[:-1]
    ).all()