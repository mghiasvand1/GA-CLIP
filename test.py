from datasets import load_dataset
from inference import inference
from tqdm import tqdm
import torch

examples = load_dataset("facebook/winoground", split="test")

cor, _all = 0, 0
a = "0.5"
for example in tqdm(examples):
    _all += 1
    x = inference(
        [example["image_0"], example["image_1"]], example["caption_0"], "text", [a]
    )
    if x != None:
        if isinstance(x, dict):
            x = x[a]
        w = torch.argmax(x).item()
        y = inference(
            [example["image_0"], example["image_1"]], example["caption_1"], "text", [a]
        )
        if y != None:
            if isinstance(y, dict):
                y = y[a]
            z = torch.argmax(y).item()
            if w == 0 and z == 1:
                cor += 1
print(f"{100*cor/_all:.2f}")

cor, _all = 0, 0
a = "0.5"
for example in tqdm(examples):
    _all += 1
    x = inference(
        example["image_0"], [example["caption_0"], example["caption_1"]], "image", [a]
    )
    if x != None:
        if isinstance(x, dict):
            x = x[a]
        w = torch.argmax(x).item()
        y = inference(
            example["image_1"],
            [example["caption_0"], example["caption_1"]],
            "image",
            [a],
        )
        if y != None:
            if isinstance(y, dict):
                y = y[a]
            z = torch.argmax(y).item()
            if w == 0 and z == 1:
                cor += 1
print(f"{100*cor/_all:.2f}")
