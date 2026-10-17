from pathlib import Path

import torch

from src.models.transformer import TransformerConfig, SmallTransformer


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoints" / "moe" / "best.pt"
TOKENS = ROOT / "data" / "validation_tokens.pt"

BATCH_SIZE = 4
BLOCK_SIZE = 128


def main():
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    config = TransformerConfig(**checkpoint["config"])
    assert config.use_moe, "Checkpoint is not an MoE model"

    model = SmallTransformer(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.train()

    tokens = torch.load(
        TOKENS,
        map_location="cpu",
        weights_only=True,
    )

    starts = torch.randint(
        len(tokens) - BLOCK_SIZE - 1,
        (BATCH_SIZE,),
    )

    x = torch.stack([
        tokens[i : i + BLOCK_SIZE]
        for i in starts.tolist()
    ]).to(device)

    model.zero_grad(set_to_none=True)

    result = model(x, labels=x)
    loss = result["loss"]

    if not torch.isfinite(loss).item():
        raise RuntimeError("Non-finite total loss")

    loss.backward()

    print(f"Total loss: {loss.item():.4f}")
    print(f"LM loss:    {result['lm_loss'].item():.4f}")
    print(f"Aux loss:   {result['aux_loss'].item():.4f}")

    for layer_idx, layer in enumerate(model.layers):
        if not layer.use_moe:
            continue

        print(f"\nLayer {layer_idx}")

        router_grad = layer.mlp.router.weight.grad

        if router_grad is None:
            print("  Router gradient: MISSING")
        else:
            norm = router_grad.norm().item()
            print(f"  Router gradient norm: {norm:.6e}")

            if not torch.isfinite(router_grad).all().item():
                raise RuntimeError("Non-finite router gradient")

        for expert_idx, expert in enumerate(layer.mlp.experts):
            grads = [
                p.grad
                for p in expert.parameters()
                if p.requires_grad
            ]

            missing = sum(g is None for g in grads)
            finite = all(
                torch.isfinite(g).all().item()
                for g in grads if g is not None
            )

            total_norm = sum(
                g.float().norm().item() ** 2
                for g in grads if g is not None
            ) ** 0.5

            print(
                f"  Expert {expert_idx}: "
                f"grad_norm={total_norm:.6e}, "
                f"missing={missing}, finite={finite}"
            )

            if not finite:
                raise RuntimeError(
                    f"Non-finite gradient in layer {layer_idx}, "
                    f"expert {expert_idx}"
                )


if __name__ == "__main__":
    main()