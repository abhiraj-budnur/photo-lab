from pathlib import Path
import torch
from src.models.transformer import SmallTransformer,TransformerConfig

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoints"/"moe"/"best.pt"
TOKENS = ROOT / "data" / "validation_tokens.pt"

BATCH_SIZE = 8
BLOCK_SIZE = 128

def main():

    device = (
        torch.device("mps")
        if torch.mps.is_available()
        else torch.device("cpu")
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    config = TransformerConfig(**checkpoint["config"])

    if not config.use_moe:
        raise RuntimeError("Checkpoint does not use MoE")

    model = SmallTransformer(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    tokens = torch.load(
        TOKENS,
        map_location="cpu",
        weights_only=True
    )

    if len(tokens) < BLOCK_SIZE + 1:
        raise ValueError("Validation token stream is too short")

    starts = torch.randint(
        len(tokens) - BLOCK_SIZE - 1,
        (BATCH_SIZE,),
    )

    x = torch.stack([
        tokens[i : i+ BLOCK_SIZE] for i in starts.tolist()
    ]).to(device)

    with torch.no_grad():
        result = model(x,labels=x)

    print(f"Validation LM loss: {result['lm_loss'].item():.4f}")
    print(f"Auxiliary loss:     {result['aux_loss'].item():.4f}")

    for index,layer in enumerate(model.layers):
        if not layer.use_moe:
            continue

        counts = layer.mlp.last_expert_counts

        if counts is None:
            print(f"Layer {index}: routing statistics unavailable")
            continue

        proportions = counts / counts.sum().clamp_min(1)
        values = [round(v,3) for v in proportions.cpu().tolist()]

        print(f"Layer {index} expert assignment fractions: {values}")

if __name__ == "__main__":
    main()