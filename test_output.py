from datasets import load_dataset
from inference import inference
from tqdm import tqdm
import torch


def eval_text_pivot(ds, a="0.5"):
    cor = tot = 0
    for ex in tqdm(ds):
        tot += 1
        x = inference([ex["image_0"], ex["image_1"]], ex["caption_0"], "text", [a])
        y = inference([ex["image_0"], ex["image_1"]], ex["caption_1"], "text", [a])
        if x and y:
            x, y = (
                x[a] if isinstance(x, dict) else x,
                y[a] if isinstance(y, dict) else y,
            )
            if torch.argmax(x) == 0 and torch.argmax(y) == 1:
                cor += 1
    print(f"Text-pivot acc: {100*cor/tot:.2f}%")


def eval_image_pivot(ds, a="0.5"):
    cor = tot = 0
    for ex in tqdm(ds):
        tot += 1
        x = inference(ex["image_0"], [ex["caption_0"], ex["caption_1"]], "image", [a])
        y = inference(ex["image_1"], [ex["caption_0"], ex["caption_1"]], "image", [a])
        if x and y:
            x, y = (
                x[a] if isinstance(x, dict) else x,
                y[a] if isinstance(y, dict) else y,
            )
            if torch.argmax(x) == 0 and torch.argmax(y) == 1:
                cor += 1
    print(f"Image-pivot acc: {100*cor/tot:.2f}%")


if __name__ == "__main__":
    ds = load_dataset("facebook/winoground", split="test")
    eval_text_pivot(ds)
    eval_image_pivot(ds)
