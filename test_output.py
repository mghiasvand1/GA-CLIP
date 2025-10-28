from datasets import load_dataset
from inference import inference
from tqdm import tqdm
import torch


def eval_image_pivot(ds):
    results = {str(a): 0 for a in ALPHAS}
    tot = len(ds)
    for ex in tqdm(ds):
        x = inference(
            ex["image_0"], [ex["caption_0"], ex["caption_1"]], "image", ALPHAS
        )
        y = inference(
            ex["image_1"], [ex["caption_0"], ex["caption_1"]], "image", ALPHAS
        )
        for a in ALPHAS:
            a_str = str(a)
            x_result = x[a_str] if isinstance(x, dict) else x
            y_result = y[a_str] if isinstance(y, dict) else y
            if torch.argmax(x_result) == 0 and torch.argmax(y_result) == 1:
                results[a_str] += 1
    for a in ALPHAS:
        a_str = str(a)
        results[a_str] = 100 * results[a_str] / tot
    top_10 = sorted(results.items(), key=lambda x: x[1], reverse=True)[:10]
    for alpha_val, accuracy in top_10:
        print(f"{alpha_val}: {accuracy:.2f}%")


def eval_text_pivot(ds):
    results = {str(a): 0 for a in ALPHAS}
    tot = len(ds)
    for ex in tqdm(ds):
        x = inference([ex["image_0"], ex["image_1"]], ex["caption_0"], "text", ALPHAS)
        y = inference([ex["image_0"], ex["image_1"]], ex["caption_1"], "text", ALPHAS)
        for a in ALPHAS:
            a_str = str(a)
            x_result = x[a_str] if isinstance(x, dict) else x
            y_result = y[a_str] if isinstance(y, dict) else y
            if torch.argmax(x_result) == 0 and torch.argmax(y_result) == 1:
                results[a_str] += 1
    for a in ALPHAS:
        a_str = str(a)
        results[a_str] = 100 * results[a_str] / tot
    top_10 = sorted(results.items(), key=lambda x: x[1], reverse=True)[:10]
    for alpha_val, accuracy in top_10:
        print(f"{alpha_val}: {accuracy:.2f}%")


ALPHAS = [
    0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    4.5,
    5.0,
]
ds = load_dataset("facebook/winoground", split="test")
eval_image_pivot(ds)
eval_text_pivot(ds)
