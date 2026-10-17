from pathlib import Path

import torch
from datasets import load_dataset
from tokenizers import Tokenizer,decoders,models,pre_tokenizers,trainers

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True,exist_ok=True)

VOCAB_SIZE = 2048
SPECIAL_TOKENS = ["<unk>","<pad>","<bos>","<eos>"]

def main():
    print("Downloading /loading WikiText-2...")
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
    )

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False
    )
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # Learn tokenizer merges from training data only
    training_texts = (
        text for text in dataset["train"]["text"] if text.strip()
    ) 
    tokenizer.train_from_iterator(
        training_texts,
        trainer=trainer
    )
    tokenizer_path = DATA_DIR / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    eos_id = tokenizer.token_to_id("<eos>")
    assert eos_id is not None

    print(f"Actual vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"EOS token ID: {eos_id}")

    for split_name, output_name in [
        ("train", "train_tokens.pt"),
        ("validation", "validation_tokens.pt"),
    ]:
        ids = []

        for text in dataset[split_name]["text"]:
            if not text.strip():
                continue

            ids.extend(tokenizer.encode(text).ids)
            ids.append(eos_id)

        tensor = torch.tensor(ids, dtype=torch.long)
        output_path = DATA_DIR / output_name
        torch.save(tensor, output_path)

        print(
            f"{split_name}: {len(tensor):,} tokens -> {output_path}"
        )

    print("Data preparation complete.")


if __name__ == "__main__":
    main()