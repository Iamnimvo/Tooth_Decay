# processing/train_model.py
# -*- coding: utf-8 -*-

import os
import re
import json
import cv2
import numpy as np
from collections import defaultdict
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from ToothClassifier import ToothClassifier

# ============================
# 📂 مسیرها
# ============================
PROC_DIR   = os.path.dirname(__file__)                                  # processing/
ROOT_DIR   = os.path.dirname(PROC_DIR)                                   # ریشه پروژه
DATA_DIR   = os.path.join(ROOT_DIR, "preprocess", "preprocessed_images")
MODEL_PATH = os.path.join(PROC_DIR, "tooth_classifier.pth")
LABELS_JS  = os.path.join(PROC_DIR, "labels.json")

# ============================
# ⚙️ تنظیمات آموزش
# ============================
RANDOM_SEED   = 42
BATCH_SIZE    = 32
NUM_EPOCHS    = 12
LR            = 1e-3
VAL_FRACTION  = 0.1        # 10% از "گروه‌ها" برای ولیدیشن
USE_IMAGENET  = True       # اگر نمی‌خواهی وزن‌های ImageNet دانلود شوند: False
USE_AMP       = True       # mixed precision روی GPU
APPLY_ROI_MASK = True      # ماسک‌کردن ROI فک/دندان روی ورودی‌ها (کمک به کاهش اهمیت پس‌زمینه)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# ============================
# 🧹 کمکی‌ها
# ============================
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_SEED)

def list_images(d: str) -> List[str]:
    if not os.path.isdir(d): return []
    return sorted([os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(IMG_EXTS)])

def infer_label_from_name(path: str) -> int:
    name = os.path.basename(path).lower()
    return 1 if any(k in name for k in ("disease", "decay", "caries", "cavity")) else 0

def base_stem(path: str) -> str:
    """foo_aug2.jpg -> foo"""
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"_aug\d+$", "", stem, flags=re.IGNORECASE)

def build_jaw_roi(gray: np.ndarray) -> np.ndarray:
    """ماسک ROI برای ناحیه‌ی فک/دندان (0/255)"""
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return mask
    c = max(cnts, key=cv2.contourArea)
    jaw = np.zeros_like(mask)
    cv2.drawContours(jaw, [c], -1, 255, -1)
    return jaw

# ============================
# 📦 دیتاست
# ============================
class XrayDataset(Dataset):
    """
    انتظار: همه‌ی تصاویر در DATA_DIR (خروجی preprocess) هستند.
    برچسب از نام فایل استخراج می‌شود.
    """
    def __init__(self, img_paths: List[str], labels: List[int], transform):
        self.img_paths = img_paths
        self.labels    = labels
        self.transform = transform
        self.class_to_idx = {"healthy": 0, "decayed": 1}

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self.img_paths[idx]
        y = self.labels[idx]

        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"خواندن تصویر ناموفق: {p}")

        if APPLY_ROI_MASK:
            roi = build_jaw_roi(img)
            img = cv2.bitwise_and(img, img, mask=roi)

        # به 3 کاناله (برای ResNet)
        img3 = np.stack([img, img, img], axis=2)
        x = self.transform(img3)
        return x, torch.tensor(y, dtype=torch.long)

# ============================
# 🔁 ترنسفورم‌ها
# ============================
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    # Augmentation سبکِ آنلاین (می‌تونی بیشترش کنی)
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485]*3, std=[0.229]*3),
])

transform_val = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485]*3, std=[0.229]*3),
])

# ============================
# 🧩 ساخت split گروه‌محور (بدون نشت)
# ============================
def make_grouped_split(paths: List[str], labels: List[int], val_fraction: float, seed: int):
    groups = defaultdict(list)
    for i, p in enumerate(paths):
        groups[base_stem(p)].append(i)

    stems = list(groups.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(stems)

    n_val = max(1, int(len(stems) * val_fraction))
    val_stems = set(stems[:n_val])
    train_stems = set(stems[n_val:])

    # ولیدیشن: فقط فایل‌های پایه (بدون _aug)
    val_idx = [i for s in val_stems for i in groups[s]
               if "_aug" not in os.path.basename(paths[i]).lower()]
    # آموزش: همه‌ی فایل‌های گروه‌های train (پایه + آگو)
    train_idx = [i for s in train_stems for i in groups[s]]

    return train_idx, val_idx

# ============================
# 📊 لودرها + سمپلر متوازن
# ============================
def make_loaders():
    all_paths = list_images(DATA_DIR)
    if not all_paths:
        raise RuntimeError(f"هیچ تصویری در '{DATA_DIR}' پیدا نشد.")
    all_labels = [infer_label_from_name(p) for p in all_paths]

    train_idx, val_idx = make_grouped_split(all_paths, all_labels, VAL_FRACTION, RANDOM_SEED)

    train_paths = [all_paths[i] for i in train_idx]
    train_labels= [all_labels[i] for i in train_idx]
    val_paths   = [all_paths[i] for i in val_idx]
    val_labels  = [all_labels[i] for i in val_idx]

    # وزن کلاس‌ها برای sampler
    counts = np.array([train_labels.count(0), train_labels.count(1)], dtype=np.float32)
    counts[counts == 0] = 1.0
    class_w = 1.0 / counts
    sample_w = np.array([class_w[y] for y in train_labels], dtype=np.float32)
    sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

    train_ds = XrayDataset(train_paths, train_labels, transform_train)
    val_ds   = XrayDataset(val_paths,   val_labels,   transform_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    class_to_idx = {"healthy": 0, "decayed": 1}
    return train_loader, val_loader, class_to_idx

# ============================
# 🧠 آموزش
# ============================
def accuracy(logits, labels):
    preds = logits.argmax(1)
    return (preds == labels).float().mean().item()

def train():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    train_loader, val_loader, class_to_idx = make_loaders()
    print(f"📊 train={len(train_loader.dataset)}  |  val={len(val_loader.dataset)}")

    model = ToothClassifier(use_imagenet_weights=USE_IMAGENET).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and DEVICE.type == "cuda")

    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        # ----- Train -----
        model.train()
        t_loss, t_acc, n_batches = 0.0, 0.0, 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [train]"):
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=USE_AMP and DEVICE.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            t_loss += loss.item()
            t_acc  += accuracy(logits, y)
            n_batches += 1
        t_loss /= max(1, n_batches)
        t_acc   = 100.0 * t_acc / max(1, n_batches)

        # ----- Val -----
        model.eval()
        v_loss, v_acc, n_batches = 0.0, 0.0, 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [val]"):
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                logits = model(x)
                loss = criterion(logits, y)
                v_loss += loss.item()
                v_acc  += accuracy(logits, y)
                n_batches += 1
        v_loss /= max(1, n_batches)
        v_acc   = 100.0 * v_acc / max(1, n_batches)

        print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | Train: loss={t_loss:.4f}, acc={t_acc:.2f}%"
              f" | Val: loss={v_loss:.4f}, acc={v_acc:.2f}%")

        # ذخیره بهترین
        if v_acc > best_val_acc:
            torch.save(model.state_dict(), MODEL_PATH)
            with open(LABELS_JS, "w", encoding="utf-8") as f:
                json.dump({"class_to_idx": class_to_idx}, f, ensure_ascii=False, indent=2)
            best_val_acc = v_acc
            print(f"💾 Best saved → {MODEL_PATH}  (val_acc={best_val_acc:.2f}%)")

    print("✅ آموزش تمام شد.")

if __name__ == "__main__":
    train()
