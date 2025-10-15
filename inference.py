from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, CLIPProcessor
from huggingface_hub import login
from train_clip import GA_CLIP
import torch

API_KEY = ""
MODEL_NAME_LM = "mghiasvand1/GA-CLIP_lm"
PARAMS_PATH = "/kaggle/input/ga-clip-params/linear_params.pth"

login(token=API_KEY)


@torch.no_grad()
def inference(img, caption, pivot, alpha):
    try:
        lm = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME_LM).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_LM)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = GA_CLIP().to("cuda")
        if pivot == "text":
            image = [i.convert("RGB") for i in img]
        elif pivot == "image":
            image = img.convert("RGB")
        model.load_params(PARAMS_PATH)
        model.eval()
        try:
            first_prompts, second_prompts, prefixes, clip_prompt, seps = (
                [],
                [],
                [],
                [],
                [],
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
            inputs = {key: tensor.to("cuda") for key, tensor in inputs.items()}
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
                inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            inputs = {key: tensor.to("cuda") for key, tensor in inputs.items()}
            outputs = lm.generate(**inputs, max_new_tokens=128)
            outputs = [
                tokenizer.decode(output, skip_special_tokens=True) for output in outputs
            ]
            c = 0
            _ = 0
            flag = True
            while flag:
                if _ < len(prefixes):
                    if "properties" in prefixes[_]:
                        clip_prompt.append(prefixes[_] + outputs[_])
                        _ += 1
                    elif "relation" in prefixes[_]:
                        objs = tuple(
                            sorted(
                                replacers[_].split(", "),
                                key=lambda x: outputs[_].find(x),
                            )
                        )
                        clip_prompt.append(
                            prefixes[_].replace("__TEMP__", " and ".join(objs))
                            + outputs[_]
                        )
                        _ += 1
                    elif "general" in prefixes[_]:
                        clip_prompt.append(prefixes[_] + caption[c])
                        seps.append(_ + c)
                        c += 1
                        prefixes.pop(_)
                else:
                    flag = False
            granularities, captions = [], []
            for prompt in clip_prompt:
                g, c = prompt.lower().split(": ", 1)
                granularities.append(g)
                captions.append(c)
            img_inputs = processor(
                images=image, return_tensors="pt", padding=True, truncation=True
            )
            img_inputs = {k: v.to("cuda") for k, v in img_inputs.items()}
            image_embeds = model.encode_image(img_inputs["pixel_values"])
            gran_len = len(granularities)
            if pivot == "image":
                image_embeds = image_embeds.repeat(gran_len, 1)
            elif pivot == "text":
                repeated_image_embeds = []
                for i in range(image_embeds.size(0)):
                    repeated_embed = image_embeds[i].unsqueeze(0).repeat(gran_len, 1)
                    repeated_image_embeds.append(repeated_embed)
                all_granularities, all_captions = [], []
                for i in range(len(image)):
                    all_granularities.extend(granularities)
                    all_captions.extend(captions)
                image_embeds = torch.cat(repeated_image_embeds, dim=0)
                granularities = all_granularities
                captions = all_captions
            gran_inputs = processor(
                text=granularities, return_tensors="pt", padding=True, truncation=True
            )
            gran_inputs = {k: v.to("cuda") for k, v in gran_inputs.items()}
            gran_embeds = model.encode_text(
                gran_inputs["input_ids"], gran_inputs["attention_mask"]
            )
            cap_inputs = processor(
                text=captions, return_tensors="pt", padding=True, truncation=True
            )
            cap_inputs = {k: v.to("cuda") for k, v in cap_inputs.items()}
            cap_embeds = model.encode_text(
                cap_inputs["input_ids"], cap_inputs["attention_mask"]
            )
            logits = model(image_embeds, gran_embeds, cap_embeds)
            logits_list = torch.diag(logits).tolist()
            if pivot == "text":
                image_logits = []
                logits_list = [
                    logits_list[i : i + gran_len]
                    for i in range(0, len(logits_list), gran_len)
                ]
                for logit in logits_list:
                    image_logits.append(logit)
            sub_parts = []
            general_parts = []
            if pivot == "image":
                start_idx = 0
                for end_idx in seps:
                    sub_slice = logits_list[start_idx : end_idx + 1]
                    sub_parts.append(sub_slice)
                    general_parts.append(logits_list[end_idx])
                    start_idx = end_idx + 1
            elif pivot == "text":
                for logits in image_logits:
                    general_parts.append(logits.pop())
                    sub_parts.append(logits)
            sub_avg = [sum(sp) / len(sp) for sp in sub_parts]
            sub_tensor = torch.tensor(sub_avg, dtype=torch.float32)
            sub_softmax = torch.nn.functional.softmax(sub_tensor, dim=0)
            general_tensor = torch.tensor(general_parts, dtype=torch.float32)
            general_softmax = torch.nn.functional.softmax(general_tensor, dim=0)
            final_percentages = {}
            for a in alpha:
                general_scaled = general_softmax * float(a)
                combined = sub_softmax + general_scaled
                final_percentages[f"{a}"] = combined / combined.sum() * 100
        except Exception:
            img_inputs = processor(
                images=image, return_tensors="pt", padding=True, truncation=True
            )
            img_inputs = {k: v.to("cuda") for k, v in img_inputs.items()}
            image_embeds = model.encode_image(img_inputs["pixel_values"])
            granularities = ["general caption" for i in range(len(caption))]
            captions = [c.lower() for c in caption]
            gran_len = len(granularities)
            if pivot == "image":
                image_embeds = image_embeds.repeat(gran_len, 1)
            elif pivot == "text":
                repeated_image_embeds = []
                for i in range(image_embeds.size(0)):
                    repeated_embed = image_embeds[i].unsqueeze(0).repeat(gran_len, 1)
                    repeated_image_embeds.append(repeated_embed)
                all_granularities, all_captions = [], []
                for i in range(len(image)):
                    all_granularities.extend(granularities)
                    all_captions.extend(captions)
                image_embeds = torch.cat(repeated_image_embeds, dim=0)
                granularities = all_granularities
                captions = all_captions
            gran_inputs = processor(
                text=granularities, return_tensors="pt", padding=True, truncation=True
            )
            gran_inputs = {k: v.to("cuda") for k, v in gran_inputs.items()}
            gran_embeds = model.encode_text(
                gran_inputs["input_ids"], gran_inputs["attention_mask"]
            )
            cap_inputs = processor(
                text=captions, return_tensors="pt", padding=True, truncation=True
            )
            cap_inputs = {k: v.to("cuda") for k, v in cap_inputs.items()}
            cap_embeds = model.encode_text(
                cap_inputs["input_ids"], cap_inputs["attention_mask"]
            )
            logits = model(image_embeds, gran_embeds, cap_embeds)
            logits_list = torch.diag(logits).tolist()
            final_percentages = torch.nn.functional.softmax(
                torch.tensor(logits_list, dtype=torch.float32), dim=0
            )
        return final_percentages
    except Exception:
        return None
