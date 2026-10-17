import torch
from torch import optim

from src.models.transformer import TransformerConfig, SmallTransformer

def get_device():
    if torch.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def main():
    torch.manual_seed(42)
    device = get_device()
    print(f"Device: {device}")

    config = TransformerConfig(
        vocab_size=512,
        hidden_size=192,
        num_layers=4,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=512,
        max_seq_len=128,
    )

    model = SmallTransformer(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = 3e-4,
        weight_decay=0.1,
    )

    model.train()

    for step in range(1,101):
        # Random tokens are for pipeline validation only.
        # They cannot teach the model language or image generation.
        tokens = torch.randint(
            0,
            config.vocab_size,
            (4,config.max_seq_len),
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)
        output = model(tokens,labels=tokens)
        loss = output["loss"]

        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"Invalid loss at step {step}: {loss}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()

        if step % 10 == 0:
            print(f"step = {step:03d} loss={loss.item():.4f}")

    torch.save(
        {
            "model": model.state_dict(),
            "config": config.__dict__,
        },
        "checkpoints/smoke_test.pt",
    )
    print("Smoke test completed.")


if __name__ == "__main__":
    main()