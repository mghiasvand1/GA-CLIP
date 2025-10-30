from transformers import CLIPProcessor, CLIPModel, set_seed
from torch.utils.data import Dataset, DataLoader, Sampler
import random, torch, json, os, requests, tempfile
from huggingface_hub import login, upload_file
from datasets import load_dataset
from torch.optim import AdamW
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from math import ceil

API_KEY = ""
SEED = 1
BATCH_SIZE = 128
EPOCHS = 3
LR = 5e-4
LW = {"L1": 0.25, "L2": 0.3, "L3": 0.45}


def fix_seed(seed):
    set_seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.vision_model = clip.vision_model
        self.visual_projection = nn.Linear(768, 512, bias=False)
        self.text_model = clip.text_model
        self.text_projection = nn.Linear(512, 512, bias=False)
        with torch.no_grad():
            self.visual_projection.weight.copy_(clip.visual_projection.weight)
            self.text_projection.weight.copy_(clip.text_projection.weight)
        for p in (
            list(self.vision_model.parameters())
            + list(self.visual_projection.parameters())
            + list(self.text_model.parameters())
        ):
            p.requires_grad = False

    def load_params(self, path):
        params = torch.load(path)
        text_projection_state = self.text_projection.state_dict()
        text_projection_state.update(params.get("text_projection_weights"))
        self.text_projection.load_state_dict(text_projection_state)
        return self

    def save_params(self):
        text_projection_weights = self.text_projection.state_dict()
        data = {
            "text_projection_weights": text_projection_weights,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "trained_params.pth")
            torch.save(data, save_path)
            upload_file(
                path_or_fileobj=save_path,
                repo_id="mghiasvand1/GA-CLIP_clip",
                path_in_repo="trained_params.pth",
                repo_type="model",
            )

    def forward(self, img_inputs, text_inputs, keep_indices_list=[]):
        with torch.no_grad():
            image_outputs = self.vision_model(img_inputs["pixel_values"]).pooler_output
            image_proj = nn.functional.normalize(
                self.visual_projection(image_outputs), dim=-1
            )
            text_outputs = self.text_model(
                text_inputs["input_ids"], text_inputs["attention_mask"]
            ).pooler_output
        text_proj = nn.functional.normalize(self.text_projection(text_outputs), dim=-1)
        if keep_indices_list:
            logits = []
            for indices in keep_indices_list:
                _logits = image_proj @ text_proj[indices].t()
                logits.append(_logits)
        else:
            logits = image_proj @ text_proj.t()
        return logits


fix_seed(SEED)
login(token=API_KEY)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = torch.device("cuda")
model = CLIP().to(device)
optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)


class ClipDataset(Dataset):
    def __init__(self):
        self.items_pos = []
        self.items_neg = {}
        self.img_dir = Path("images")
        self.dir_existed = self.img_dir.exists()
        self.img_dir.mkdir(exist_ok=True)
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
            if not self.dir_existed:
                img_path = self.img_dir / f"{str(iid).zfill(12)}.jpg"
                if not img_path.exists():
                    img_url = f"http://images.cocodataset.org/train2017/{str(iid).zfill(12)}.jpg"
                    img_data = requests.get(img_url).content
                    with open(img_path, "wb") as f:
                        f.write(img_data)
        self.pos_indices_by_image = {}
        for idx, item in enumerate(self.items_pos):
            iid = item["image_id"]
            self.pos_indices_by_image.setdefault(iid, []).append(idx)
        self.unique_image_ids = list(self.pos_indices_by_image.keys())

    def __len__(self):
        return len(self.items_pos)

    def __getitem__(self, idx):
        entry = self.items_pos[idx]
        img_path = self.img_dir / f"{str(entry['image_id']).zfill(12)}.jpg"
        image = Image.open(img_path).convert("RGB")
        text = entry["text"]
        negs = self.items_neg.get(entry["image_id"])
        return {
            **entry,
            "image": image,
            "text": text,
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
    images, texts, meta = [], [], []
    for item in batch:
        images.append(item["image"])
        texts.append(item["text"])
        meta.append({k: item[k] for k in ("id", "image_id", "status")})
        negs = item.get("negatives")
        for n in negs:
            texts.append(n["text"])
            meta.append({k: n[k] for k in ("id", "image_id", "status")})
    img_inputs = processor(
        images=images, return_tensors="pt", padding=True, truncation=True
    )
    text_inputs = processor(
        text=texts, return_tensors="pt", padding=True, truncation=True
    )
    return img_inputs, text_inputs, meta


def MPCL(logits, i2pt, t2pi):
    i2t_losses = []
    for img_idx, pos_text_indices in i2pt.items():
        pos_logits = logits[img_idx, pos_text_indices]
        denom_logits = logits[img_idx, :]
        loss = -torch.log(
            torch.sum(torch.exp(pos_logits)) / torch.sum(torch.exp(denom_logits))
        )
        i2t_losses.append(loss)
    i2t_loss = torch.stack(i2t_losses).mean()
    t2i_losses = []
    for text_idx, img_idx in t2pi.items():
        numerator = logits[img_idx, text_idx]
        denominator = logits[:, text_idx]
        loss = -torch.log(torch.exp(numerator) / torch.sum(torch.exp(denominator)))
        t2i_losses.append(loss)
    t2i_loss = torch.stack(t2i_losses).mean()
    return 0.5 * (i2t_loss + t2i_loss)


def NL(logits, i2tp, t2nt):
    batch_losses = []
    for img_idx, pos_text_indices in i2tp.items():
        img_losses = []
        for pos_idx in pos_text_indices:
            neg_indices = t2nt.get(pos_idx)
            if not neg_indices:
                continue
            pos_similarity = logits[img_idx, pos_idx]
            neg_similarities = logits[img_idx, neg_indices]
            numerator = torch.exp(pos_similarity)
            denominator = numerator + torch.sum(torch.exp(neg_similarities))
            loss = -torch.log(numerator / denominator)
            img_losses.append(loss)
        if img_losses:
            batch_losses.append(torch.stack(img_losses).mean())
    return torch.stack(batch_losses).mean()


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
    scaler = torch.GradScaler("cuda")
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        num_batches = 0
        for img_inputs, text_inputs, meta in loader:
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
            meta_pos = [meta[i] for i in keep_idx_L1]
            seen = set()
            unique_indices = [
                i
                for i, m in enumerate(meta_pos)
                if m["image_id"] not in seen and not seen.add(m["image_id"])
            ]
            img_inputs = {
                k: v[unique_indices].to(device) for k, v in img_inputs.items()
            }
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            map_index = {
                "pos": {"i2pt": {}, "t2pi": {}},
                "pos_interneg": {"i2pt": {}, "t2nt": {}},
                "pos_intraneg": {"i2pt": {}, "t2nt": {}},
            }
            meta_pos_interneg = [meta[i] for i in keep_idx_L2]
            meta_pos_intraneg = [meta[i] for i in keep_idx_L3]
            imageid_to_unique = {
                meta_pos[m]["image_id"]: i for i, m in enumerate(unique_indices)
            }
            for local_idx, u_idx in enumerate(unique_indices):
                img_id = meta_pos[u_idx]["image_id"]
                map_index["pos"]["i2pt"][local_idx] = [
                    i for i, m in enumerate(meta_pos) if m["image_id"] == img_id
                ]
                map_index["pos_interneg"]["i2pt"][local_idx] = [
                    i
                    for i, m in enumerate(meta_pos_interneg)
                    if m["image_id"] == img_id and m["status"] == "Pos"
                ]
                map_index["pos_intraneg"]["i2pt"][local_idx] = [
                    i
                    for i, m in enumerate(meta_pos_intraneg)
                    if m["image_id"] == img_id and m["status"] == "Pos"
                ]
            for i, m in enumerate(meta_pos):
                map_index["pos"]["t2pi"][i] = imageid_to_unique[m["image_id"]]
            for i, m in enumerate(meta_pos_interneg):
                if m["status"] == "Pos":
                    map_index["pos_interneg"]["t2nt"][i] = [
                        j
                        for j, x in enumerate(meta_pos_interneg)
                        if str(m["id"]) in x["status"]
                    ]
            for i, m in enumerate(meta_pos_intraneg):
                if m["status"] == "Pos":
                    map_index["pos_intraneg"]["t2nt"][i] = [
                        j
                        for j, x in enumerate(meta_pos_intraneg)
                        if str(m["id"]) in x["status"]
                    ]
            with torch.autocast("cuda"):
                logits_list = model(
                    img_inputs, text_inputs, [keep_idx_L1, keep_idx_L2, keep_idx_L3]
                )
                L1 = MPCL(
                    logits_list[0], map_index["pos"]["i2pt"], map_index["pos"]["t2pi"]
                )
                L2 = NL(
                    logits_list[1],
                    map_index["pos_interneg"]["i2pt"],
                    map_index["pos_interneg"]["t2nt"],
                )
                L3 = NL(
                    logits_list[2],
                    map_index["pos_intraneg"]["i2pt"],
                    map_index["pos_intraneg"]["t2nt"],
                )
                loss = LW["L1"] * L1 + LW["L2"] * L2 + LW["L3"] * L3
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
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
