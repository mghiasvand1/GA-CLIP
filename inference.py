from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, CLIPProcessor
from huggingface_hub import login, hf_hub_download
from train_clip import CLIP
import torch.nn as nn
import torch

API_KEY = ""
login(token=API_KEY)
TRAINED_PARAMS_PATH = hf_hub_download(
    repo_id="mghiasvand1/GA-CLIP_clip",
    filename="trained_params.pth",
    repo_type="model",
)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = torch.device("cuda")
lm = AutoModelForSeq2SeqLM.from_pretrained("mghiasvand1/GA-CLIP_lm").to(device)
tokenizer = AutoTokenizer.from_pretrained("mghiasvand1/GA-CLIP_lm")
model = CLIP().to(device).load_params(TRAINED_PARAMS_PATH)
model.eval()


@torch.no_grad()
def inference(img, caption, pivot, alpha):
    image = [i.convert("RGB") for i in img] if pivot == "text" else img.convert("RGB")
    try:
        first_prompts, second_prompts, prefixes, clip_prompt, seps = (
            [] for _ in range(5)
        )
        if pivot == "image":
            for y in range(len(caption)):
                first_prompts.append(f"{caption[y]} => key objects")
                first_prompts.append(f"{caption[y]} => relevant objects")
        if pivot == "text":
            first_prompts.append(f"{caption} => key objects")
            first_prompts.append(f"{caption} => relevant objects")
        inputs = tokenizer(
            first_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
        outputs = lm.generate(**inputs, max_new_tokens=128)
        first_outputs = [
            tokenizer.decode(output, skip_special_tokens=True) for output in outputs
        ]
        inputs, replacers = [], []
        for z in range(len(first_outputs)):
            if z % 2 == 0:
                for obj in first_outputs[z].split(", "):
                    inputs.append(f"{caption[z // 2]} => {obj}")
                    prefixes.append(f"properties of {obj}: ")
                    replacers.append("")
            else:
                first_outputs[z] = first_outputs[z].replace("),", ");")
                for pair in first_outputs[z].split("; "):
                    a, b = pair.replace("(", "").replace(")", "").split(", ")
                    a, b = a.strip(), b.strip()
                    replacers.append(f"{a}, {b}")
                    inputs.append(f"{caption[z // 2]} => ({a}, {b})")
                    prefixes.append("relation between __TEMP__: ")
                prefixes.append("general caption: ")
        inputs = tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True, max_length=128
        )
        inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
        outputs = lm.generate(**inputs, max_new_tokens=128)
        outputs = [
            tokenizer.decode(output, skip_special_tokens=True) for output in outputs
        ]
        c, _, flag = 0, 0, True
        while flag:
            if _ < len(prefixes):
                if "properties" in prefixes[_]:
                    clip_prompt.append(prefixes[_] + outputs[_])
                    _ += 1
                elif "relation" in prefixes[_]:
                    objs = tuple(
                        sorted(
                            replacers[_].split(", "), key=lambda x: outputs[_].find(x)
                        )
                    )
                    clip_prompt.append(
                        prefixes[_].replace("__TEMP__", " and ".join(objs)) + outputs[_]
                    )
                    _ += 1
                elif "general" in prefixes[_]:
                    clip_prompt.append(prefixes[_] + caption[c])
                    seps.append(_ + c)
                    c += 1
                    prefixes.pop(_)
            else:
                flag = False
        img_inputs = processor(
            images=image, return_tensors="pt", padding=True, truncation=True
        )
        img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
        clip_prompt = [prompt.lower() for prompt in clip_prompt]
        text_inputs = processor(
            text=clip_prompt, return_tensors="pt", padding=True, truncation=True
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        logits = model(img_inputs, text_inputs).tolist()
        if pivot == "image":
            logits = logits[0]
        sub_parts, general_parts = [], []
        if pivot == "image":
            start_idx = 0
            for end_idx in seps:
                sub_slice = logits[start_idx : end_idx + 1]
                sub_parts.append(sub_slice)
                general_parts.append(logits[end_idx])
                start_idx = end_idx + 1
        elif pivot == "text":
            for logits in logits:
                general_parts.append(logits.pop())
                sub_parts.append(logits)
        sub_avg = [sum(sp) / len(sp) for sp in sub_parts]
        sub_tensor = torch.tensor(sub_avg, dtype=torch.float32)
        sub_softmax = nn.functional.softmax(sub_tensor, dim=0)
        general_tensor = torch.tensor(general_parts, dtype=torch.float32)
        general_softmax = nn.functional.softmax(general_tensor, dim=0)
        final_percentages = {}
        for a in alpha:
            general_scaled = general_softmax * float(a)
            combined = sub_softmax + general_scaled
            final_percentages[f"{a}"] = combined / combined.sum() * 100
    except Exception:
        img_inputs = processor(
            images=image, return_tensors="pt", padding=True, truncation=True
        )
        img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
        clip_prompt = (
            [f"general caption: {c.lower()}" for c in caption]
            if pivot == "image"
            else f"general caption: {caption.lower()}"
        )
        text_inputs = processor(
            text=clip_prompt, return_tensors="pt", padding=True, truncation=True
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        logits = model(img_inputs, text_inputs).tolist()
        if pivot == "image":
            logits = logits[0]
        elif pivot == "text":
            logits = [_logits[0] for _logits in logits]
        final_percentages = nn.functional.softmax(
            torch.tensor(logits, dtype=torch.float32), dim=0
        )
    return final_percentages
