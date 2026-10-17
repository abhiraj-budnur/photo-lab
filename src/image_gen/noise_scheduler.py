import torch
import torch.nn as nn


class DDPMScheduler(nn.Module):
    """Linear DDPM noise scheduler for latent diffusion."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ):
        super().__init__()

        if num_train_timesteps < 2:
            raise ValueError("num_train_timesteps must be >= 2")

        if not 0.0 < beta_start < beta_end < 1.0:
            raise ValueError(
                "Require 0 < beta_start < beta_end < 1"
            )

        self.num_train_timesteps = num_train_timesteps

        # Construct the schedule in FP32 for numerical stability.
        betas = torch.linspace(
            beta_start,
            beta_end,
            num_train_timesteps,
            dtype=torch.float32,
        )

        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)

        # alpha_bar at the previous timestep; alpha_bar[-1] is not
        # used as the previous value for timestep zero.
        alpha_cumprod_prev = torch.cat(
            [
                torch.ones(1, dtype=torch.float32),
                alpha_cumprod[:-1],
            ]
        )

        posterior_variance = (
            betas
            * (1.0 - alpha_cumprod_prev)
            / (1.0 - alpha_cumprod)
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)

        self.register_buffer(
            "sqrt_alpha_cumprod",
            torch.sqrt(alpha_cumprod),
        )
        self.register_buffer(
            "sqrt_one_minus_alpha_cumprod",
            torch.sqrt(1.0 - alpha_cumprod),
        )

        self.register_buffer(
            "posterior_variance",
            posterior_variance.clamp_min(0.0),
        )

        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alpha_cumprod_prev)
            / (1.0 - alpha_cumprod),
        )

        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alpha_cumprod_prev)
            * torch.sqrt(alphas)
            / (1.0 - alpha_cumprod),
        )

    @staticmethod
    def _extract(
        values: torch.Tensor,
        timesteps: torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """Gather one coefficient per batch item and broadcast it."""

        selected = values.gather(0, timesteps)

        return selected.reshape(
            sample.shape[0],
            *([1] * (sample.ndim - 1)),
        )

    def _validate(
        self,
        sample: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> None:
        if sample.ndim != 4:
            raise ValueError(
                "sample must have shape [B, C, H, W]"
            )

        if not sample.is_floating_point():
            raise ValueError("sample must be floating point")

        if timesteps.ndim != 1:
            raise ValueError("timesteps must have shape [B]")

        if timesteps.shape[0] != sample.shape[0]:
            raise ValueError(
                "timesteps batch size must match sample batch size"
            )

        if timesteps.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("timesteps must contain integer indices")

        if timesteps.device != sample.device:
            raise ValueError(
                "timesteps and sample must be on the same device"
            )

        if timesteps.numel() == 0:
            raise ValueError("Batch must not be empty")

        if (
            (timesteps < 0).any().item()
            or (
                timesteps >= self.num_train_timesteps
            ).any().item()
        ):
            raise ValueError("Timestep is outside the schedule")

        if self.betas.device != sample.device:
            raise ValueError(
                "Move the scheduler to the sample device first"
            )

    def add_noise(
        self,
        clean_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Sample x_t from clean x_0 and Gaussian noise."""

        self._validate(clean_samples, timesteps)

        if noise.shape != clean_samples.shape:
            raise ValueError(
                "noise and clean_samples must have identical shapes"
            )

        if noise.device != clean_samples.device:
            raise ValueError(
                "noise and clean_samples must be on the same device"
            )

        signal = self._extract(
            self.sqrt_alpha_cumprod,
            timesteps,
            clean_samples,
        )

        noise_scale = self._extract(
            self.sqrt_one_minus_alpha_cumprod,
            timesteps,
            clean_samples,
        )

        return (
            signal * clean_samples
            + noise_scale * noise
        )

    def predict_clean_sample(
        self,
        noisy_samples: torch.Tensor,
        predicted_noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate x_0 from x_t and predicted epsilon."""

        self._validate(noisy_samples, timesteps)

        if predicted_noise.shape != noisy_samples.shape:
            raise ValueError(
                "predicted_noise must match noisy_samples shape"
            )

        signal = self._extract(
            self.sqrt_alpha_cumprod,
            timesteps,
            noisy_samples,
        )

        noise_scale = self._extract(
            self.sqrt_one_minus_alpha_cumprod,
            timesteps,
            noisy_samples,
        )

        return (
            noisy_samples - noise_scale * predicted_noise
        ) / signal.clamp_min(1e-8)