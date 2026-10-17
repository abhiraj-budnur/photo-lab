from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT/"data"/"flickr8k"
DATA_DIR.mkdir(parents=True,exist_ok=True)

def main():
    print("Downloading Dataset")
    dataset = load_dataset("intro/flickr8k")

    print(dataset)

    for split in dataset:
        print(f"{split}: {len(dataset[split]):,} images")

    # Save the dataset in the hugging face cache.
    # Keep the project data directory for our own processed files
    print(f"Project Data Directory: {DATA_DIR}")
    print("Dataset download and loading complete")

    example = dataset["train"][0]
    print("Available fields: ",list(example.keys()))
    print("Example Caption: ",example["caption_0"])

if __name__ == "__main__":
    main()