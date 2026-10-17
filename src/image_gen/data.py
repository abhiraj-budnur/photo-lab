from operator import index

import torch
import numpy as np
from typing import Any
from PIL import Image,ImageOps
from torch.utils.data import Dataset


def extract_caption(row:dict[str,Any])->str:
    """Support common single-caption and multi-caption dataset layouts."""
    for key in ("caption","text","description"):
        value = row.get(key)
        if isinstance(value,str) and value.strip():
            return value.strip()

    captions = row.get("captions")
    if isinstance(captions,list) and captions:
        first = captions[0]
        if isinstance(first,str) and first.strip():
            return first.strip()
        if isinstance(first,dict):
            value = first.get("caption") or first.get("text")
            if isinstance(value,str) and value.strip():
                return value.strip()

    for key in sorted(row):
        if key.startswith("caption_"):
            value = row[key]
            if isinstance (value,str) and value.strip():
                return value.strip()

    raise ValueError("No non-empty caption field found in dataset row")


class CaptionImageDataset(Dataset):

    def __init__(
            self,
            hf_split,
            image_size:int=512,
    ):
        self.data = hf_split
        self.image_size = image_size

        if len(self.data) == 0:
            raise ValueError("Dataset split is empty")

        sample = self.data[0]

        if "image" not in sample:
            raise ValueError(
                f"Expected an image field; found {list(sample.keys())}"
            )

        # Fail early if the dataset's caption schema is incompatible
        extract_caption(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        row = self.data[index]
        image = row["image"]

        if not isinstance(image,Image.Image):
            raise TypeError(
                f"Expected PIL image, got {type(image).__name__}"
            )

        image = image.convert("RGB")
        image = ImageOps.fit(
            image,
            (self.image_size,self.image_size),
            method=Image.Resampling.LANCZOS,
        )

        array = np.asarray(image,dtype=np.float32).copy() / 127.5-1.0
        pixels = torch.from_numpy(array).permute(2,0,1).contiguous()

        return {
            "pixel_values":pixels,
            "caption":extract_caption(row),
        }