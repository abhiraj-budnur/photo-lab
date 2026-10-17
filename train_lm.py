import os
from pathlib import Path
import random
import torch

from src.models.transformer import TransformerConfig, SmallTransformer
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints" / "moe"

SEED = 42
BATCH_SIZE = 8
BLOCK_SIZE = 128
MAX_STEPS = 1000
EVAL_INTERVAL = 100
EVAL_BATCHES = 20
LEARNING_RATE = 1e-4

def save_checkpoint_atomic(checkpoint, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")

    try:
        with open(temp_path, "wb") as f:
            torch.save(checkpoint, f)
            f.flush()
            os.fsync(f.fileno())

        # Verify the saved file before replacing the destination.
        loaded = torch.load(
            temp_path,
            map_location="cpu",
            weights_only=True,
        )

        if "model" not in loaded or "config" not in loaded:
            raise RuntimeError("Checkpoint is missing required fields")

        if temp_path.stat().st_size == 0:
            raise RuntimeError("Checkpoint file is empty")

        os.replace(temp_path, path)

        print(f"Saved {path} ({path.stat().st_size:,} bytes)")

    finally:
        if temp_path.exists():
            temp_path.unlink()

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_batch(tokens, batch_size, block_size, device):
    # Sample random contiguous windows from a flat token stream.
    starts = torch.randint(
        len(tokens) - block_size - 1,
        (batch_size,),
    )

    x = torch.stack([
        tokens[i : i + block_size] for i in starts.tolist()
    ])
    y = x.clone()

    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_tokens, val_tokens, device):
    model.eval()
    results = {}

    for name, tokens in [
        ("train", train_tokens),
        ("val", val_tokens),
    ]:
        lm_losses = []
        aux_losses = []

        for _ in range(EVAL_BATCHES):
            x, y = get_batch(
                tokens, BATCH_SIZE, BLOCK_SIZE, device
            )

            result = model(x, labels=y)

            lm_losses.append(result["lm_loss"].item())
            aux_losses.append(result["aux_loss"].item())

        results[name] = {
            "lm_loss": sum(lm_losses) / len(lm_losses),
            "aux_loss": sum(aux_losses) / len(aux_losses),
        }

    model.train()
    return results

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    device = get_device()
    print(f"Training device: {device}")

    train_tokens = torch.load(
        DATA_DIR / "train_tokens.pt",
        map_location="cpu",
        weights_only=True,
    )
    val_tokens = torch.load(
        DATA_DIR / "validation_tokens.pt",
        map_location="cpu",
        weights_only=True,
    )

    tokenizer = Tokenizer.from_file(
        str(DATA_DIR / "tokenizer.json")
    )
    vocab_size = tokenizer.get_vocab_size()

    assert train_tokens.numel() > BLOCK_SIZE + 1
    assert val_tokens.numel() > BLOCK_SIZE + 1
    assert int(train_tokens.max()) < vocab_size
    assert int(val_tokens.max()) < vocab_size

    config = TransformerConfig(
        vocab_size=vocab_size,
        hidden_size=192,
        num_layers=4,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=512,
        max_seq_len=BLOCK_SIZE,

        use_moe=True,
        num_experts=4,
        moe_top_k=2,
        moe_intermediate_size=256,
        moe_aux_loss_coef=0.01,
    )

    model = SmallTransformer(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.1,
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training tokens: {len(train_tokens):,}")
    print(f"Validation tokens: {len(val_tokens):,}")

    for step in range(1, MAX_STEPS + 1):
        model.train()
        x, y = get_batch(
            train_tokens, BATCH_SIZE, BLOCK_SIZE, device
        )

        optimizer.zero_grad(set_to_none=True)
        loss = model(x, labels=y)["loss"]

        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % EVAL_INTERVAL == 0 or step == 1:
            metrics = estimate_loss(
                model, train_tokens, val_tokens, device
            )
            print(
                f"step={step:04d} "
                f"train_lm={metrics['train']['lm_loss']:.4f} "
                f"val_lm={metrics['val']['lm_loss']:.4f}"
                f"train_aux={metrics['train']['aux_loss']:.4f} "
                f"val_aux={metrics['val']['aux_loss']:.4f}"
            )

            checkpoint = {
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config.__dict__,
                "best_val_loss": best_val_loss,
            }

            torch.save(
                checkpoint,
                CHECKPOINT_DIR / "latest.pt",
            )

            val_loss = metrics["val"]["lm_loss"]
            improved = val_loss < best_val_loss

            if improved:
                best_val_loss = val_loss

            checkpoint = {
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config.__dict__,
                "best_val_loss": best_val_loss,
            }

            save_checkpoint_atomic(
                checkpoint,
                CHECKPOINT_DIR / "latest.pt",
            )

            if improved:
                save_checkpoint_atomic(
                    checkpoint,
                    CHECKPOINT_DIR / "best.pt",
                )

    print(f"Best validation loss: {best_val_loss:.4f}")
    print("Training complete.")


if __name__ == "__main__":
    main()
