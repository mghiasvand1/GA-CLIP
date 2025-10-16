from transformers import CLIPProcessor, CLIPModel, set_seed
from torch.utils.data import Dataset, DataLoader, Sampler
from IPython.display import display, HTML
from datasets import load_dataset
from huggingface_hub import login
import random, torch, json, os
from torch.optim import AdamW
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from math import ceil
import numpy as np

PARAMS_PATH = "/kaggle/working/trained_params.pth"
MODEL_NAME = "openai/clip-vit-base-patch32"
IMG_DATA = "/kaggle/input/coco-image-caption/train2014/train2014"
API_KEY = ""
SEED = 1
BATCH_SIZE = 80
EPOCHS = 8
LR = 1e-4
WD = 0.1
LOSS_WEIGHTS = {"L1": 1.0, "L2": 1.0, "L3": 1.0}


def fix_seed(seed):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class GA_CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        clip = CLIPModel.from_pretrained(MODEL_NAME)
        self.vision_model = clip.vision_model
        self.text_model = clip.text_model
        self.linear_caption = nn.Linear(512, 512, bias=False)
        self.linear_fuse = nn.Linear(1280, 512, bias=False)
        with torch.no_grad():
            self.linear_caption.weight.copy_(clip.text_projection.weight)
            self.linear_fuse.weight.copy_(
                torch.cat(
                    [
                        clip.visual_projection.weight,
                        clip.text_projection.weight,
                    ],
                    dim=1,
                )
            )
        for p in self.vision_model.parameters():
            p.requires_grad = False

        for name, param in self.text_model.named_parameters():
            if "bias" not in name:
                param.requires_grad = False

    def load_params(self, path):
        params = torch.load(path)
        text_model_state = self.text_model.state_dict()
        text_model_state.update(params.get("text_model_biases"))
        self.text_model.load_state_dict(text_model_state)
        linear_caption_state = self.linear_caption.state_dict()
        linear_caption_state.update(params.get("linear_caption_weights"))
        self.linear_caption.load_state_dict(linear_caption_state)
        linear_fuse_state = self.linear_fuse.state_dict()
        linear_fuse_state.update(params.get("linear_fuse_weights"))
        self.linear_fuse.load_state_dict(linear_fuse_state)

    def save_params(self):
        text_model_biases = {
            k: v for k, v in self.text_model.state_dict().items() if "bias" in k
        }
        linear_caption_weights = self.linear_caption.state_dict()
        linear_fuse_weights = self.linear_fuse.state_dict()
        data = {
            "text_model_biases": text_model_biases,
            "linear_caption_weights": linear_caption_weights,
            "linear_fuse_weights": linear_fuse_weights,
        }
        torch.save(data, PARAMS_PATH)

    @torch.no_grad()
    def encode_image(self, pixel_values):
        image_outputs = self.vision_model(pixel_values)
        return image_outputs.pooler_output

    def encode_text(self, input_ids, attention_mask):
        text_outputs = self.text_model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        return text_outputs.pooler_output

    def forward(self, img_emb, gran_emb, cap_emb):
        cap_proj = nn.functional.normalize(self.linear_caption(cap_emb), p=2, dim=-1)
        fuse_proj = nn.functional.normalize(
            self.linear_fuse(torch.cat([img_emb, gran_emb], dim=-1)), p=2, dim=-1
        )
        logits = fuse_proj @ cap_proj.t()
        return logits


display(
    HTML(
        "<script>Jupyter.notebook.kernel.execute('config NotebookApp.iopub_msg_rate_limit=10000000000')</script>"
    )
)
login(token=API_KEY)
fix_seed(SEED)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
device = torch.device("cuda")
model = GA_CLIP().to(device)
optimizer = AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WD
)


class ClipDataset(Dataset):
    def __init__(self):
        self.items_pos = []
        self.items_neg = {}
        dataset = load_dataset(
            "mghiasvand1/GA-CLIP_data", data_files="clip_train.jsonl", split="train"
        )
        for line in dataset:
            iid = int(line["image_id"])
            data = {
                "id": int(line["id"]),
                "image_id": iid,
                "text": line["text"],
                "status": line["status"],
            }
            if line["status"] == "Pos":
                self.items_pos.append(data)
            else:
                if iid not in self.items_neg:
                    self.items_neg[iid] = []
                self.items_neg[iid].append(data)
        self.pos_indices_by_image = {}
        for idx, item in enumerate(self.items_pos):
            iid = item["image_id"]
            self.pos_indices_by_image.setdefault(iid, []).append(idx)
        self.unique_image_ids = list(self.pos_indices_by_image.keys())

    def __len__(self):
        return len(self.items_pos)

    def __getitem__(self, idx):
        entry = self.items_pos[idx]
        img_path = f"{IMG_DATA}/COCO_train2014_{str(entry['image_id']).zfill(12)}.jpg"
        image = Image.open(img_path).convert("RGB")
        granularity, caption = entry["text"].split(": ", 1)
        negs = self.items_neg.get(entry["image_id"])
        for n in negs:
            n["granularity"], n["caption"] = n["text"].split(": ", 1)
        return {
            **entry,
            "image": image,
            "granularity": granularity,
            "caption": caption,
            "negatives": negs,
        }


class UniqueImageBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, image_fraction=1.0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.image_ids = list(dataset.unique_image_ids)
        self.image_fraction = image_fraction

    def __iter__(self):
        k = int(ceil(len(self.image_ids) * self.image_fraction))
        chosen = list(self.image_ids)[:k]
        random.shuffle(chosen)
        for start in range(0, len(chosen), self.batch_size):
            batch_img_ids = chosen[start : start + self.batch_size]
            batch_indices = []
            for iid in batch_img_ids:
                batch_indices.extend(self.dataset.pos_indices_by_image[iid])
            yield batch_indices

    def __len__(self):
        k = int(ceil(len(self.image_ids) * self.image_fraction))
        return ceil(k / self.batch_size)


def collate_fn(batch):
    images, granularities, captions, meta = [], [], [], []
    for item in batch:
        images.append(item["image"])
        granularities.append(item["granularity"])
        captions.append(item["caption"])
        meta.append({k: item[k] for k in ("id", "image_id", "status")})
        negs = item.get("negatives")
        for n in negs:
            granularities.append(n["granularity"])
            captions.append(n["caption"])
            meta.append({k: n[k] for k in ("id", "image_id", "status")})
    img_inputs = processor(
        images=images, return_tensors="pt", padding=True, truncation=True
    )
    gran_inputs = processor(
        text=granularities, return_tensors="pt", padding=True, truncation=True
    )
    cap_inputs = processor(
        text=captions, return_tensors="pt", padding=True, truncation=True
    )
    return img_inputs, gran_inputs, cap_inputs, meta


def CL(logits, meta):
    image_ids = torch.tensor([m["image_id"] for m in meta])
    mask = (
        (image_ids.unsqueeze(1) != image_ids.unsqueeze(0)).float().fill_diagonal_(1.0)
    )
    logits = logits.masked_fill(mask == 0, float("-inf"))
    labels = torch.arange(logits.shape[0])
    loss_i2t = nn.functional.cross_entropy(logits, labels)
    loss_t2i = nn.functional.cross_entropy(logits.t(), labels)
    return (loss_i2t + loss_t2i) / 2


def NL(logits, meta):
    ids = [m["id"] for m in meta]
    image_ids = [m["image_id"] for m in meta]
    statuses = [m["status"] for m in meta]
    batch_loss = []
    image_to_indices = {}
    for idx, img_id in enumerate(image_ids):
        if img_id not in image_to_indices:
            image_to_indices[img_id] = []
        if statuses[idx] == "Pos":
            image_to_indices[img_id].append(idx)
    for img_id, indices in image_to_indices.items():
        img_loss = []
        for pos_idx in indices:
            pos_id = ids[pos_idx]
            neg_indices = [
                i
                for i in range(len(meta))
                if statuses[i] != "Pos" and str(pos_id) in statuses[i]
            ]
            if not neg_indices:
                continue
            pos_similarity = logits[pos_idx, pos_idx]
            neg_similarities = logits[pos_idx, neg_indices]
            numerator = torch.exp(pos_similarity)
            denominator = numerator + torch.sum(torch.exp(neg_similarities))
            loss = -torch.log(numerator / denominator)
            img_loss.append(loss)
        batch_loss.append(torch.stack(img_loss).mean())
    return torch.stack(batch_loss).mean()


def fine_tune():
    dataset = ClipDataset()
    sampler = UniqueImageBatchSampler(dataset, BATCH_SIZE)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    total_steps = len(loader) * EPOCHS
    pbar = tqdm(total=total_steps, unit="batch")
    epoch_losses = []
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        num_batches = 0
        for img_inputs, gran_inputs, cap_inputs, meta in loader:
            img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
            gran_inputs = {k: v.to(device) for k, v in gran_inputs.items()}
            cap_inputs = {k: v.to(device) for k, v in cap_inputs.items()}
            image_embeds = model.encode_image(img_inputs["pixel_values"])
            gran_embeds = model.encode_text(
                gran_inputs["input_ids"], gran_inputs["attention_mask"]
            )
            cap_embeds = model.encode_text(
                cap_inputs["input_ids"], cap_inputs["attention_mask"]
            )
            keep_idx_L1 = [i for i, m in enumerate(meta) if m["status"] == "Pos"]
            keep_idx_L2 = [
                i
                for i, m in enumerate(meta)
                if m["status"] == "Pos" or "InterNeg" in m["status"]
            ]
            keep_idx_L3 = [
                i
                for i, m in enumerate(meta)
                if m["status"] == "Pos" or "IntraNeg" in m["status"]
            ]
            meta_L1 = [meta[i] for i in keep_idx_L1]
            meta_L2 = [meta[i] for i in keep_idx_L2]
            meta_L3 = [meta[i] for i in keep_idx_L3]
            logits = model(
                image_embeds, gran_embeds[keep_idx_L1], cap_embeds[keep_idx_L1]
            )
            L1 = CL(logits, meta_L1)
            logits = model(
                image_embeds,
                gran_embeds[keep_idx_L2],
                cap_embeds[keep_idx_L2],
            )
            L2 = NL(logits, meta_L2)
            logits = model(
                image_embeds,
                gran_embeds[keep_idx_L3],
                cap_embeds[keep_idx_L3],
            )
            L3 = NL(logits, meta_L3)
            loss = (
                LOSS_WEIGHTS["L1"] * L1
                + LOSS_WEIGHTS["L2"] * L2
                + LOSS_WEIGHTS["L3"] * L3
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
            pbar.update(1)
        avg_loss = epoch_loss / num_batches if num_batches > 0 else float("inf")
        epoch_losses.append(avg_loss)
        loss_str = "; ".join([f"E{i+1}: {l:.3f}" for i, l in enumerate(epoch_losses)])
        pbar.set_description(
            f"Epoch {epoch + 1}/{EPOCHS} | Training losses [{loss_str}]"
        )
    model.save_params()
