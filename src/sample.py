from pathlib import Path

import torch
from tokenizers import Tokenizer

from src.models.transformer import TransformerConfig, SmallTransformer


ROOT = Path(__file__).resolve().parents[2]


def main():
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    checkpoint = torch.load(
        ROOT / "checkpoints" / "best.pt",
        map_location="cpu",
        weights_only=True,
    )

    config = TransformerConfig(**checkpoint["config"])
    model = SmallTransformer(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    tokenizer = Tokenizer.from_file(
        str(ROOT / "data" / "tokenizer.json")
    )

    prompt = "The history of science"
    token_ids = tokenizer.encode(prompt).ids

    input_ids = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device,
    )

    output = model.generate(
        input_ids,
        max_new_tokens=60,
        temperature=0.8,
        top_k=40,
    )

    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()