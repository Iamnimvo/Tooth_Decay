# -*- coding: utf-8 -*-
"""
Preprocess X-ray images:
 - دانلود از Arvan
 - تبدیل به grayscale + حذف زمینه سفید
 - resize/pad به 224x224
 - بهبود کنتراست + کاهش نویز
 - ذخیره نسخه اصلی و augmented
"""

import os
import cv2
import boto3
import numpy as np
import albumentations as A
from tqdm import tqdm

# ============================
# 🔐 تنظیمات Arvan
# ============================
ACCESS_KEY  = "a8761df0-960e-4dd1-b5f2-ef8ef60823a9"
SECRET_KEY  = "f502aad1cec94636d4381cadf302a6114df05bf825864e554b82010d6d2441ab"
REGION_HOST = "s3.ir-thr-at1.arvanstorage.ir"
BUCKET_NAME = "ehsannima"
FOLDER_PREF = "xray/"   # مسیر داخل باکت

# ============================
# 📂 مسیرهای محلی (نسبی به پروژه)
# ============================
BASE_DIR         = os.path.dirname(__file__)       # پوشه preprocess/
DOWNLOAD_DIR     = os.path.join(BASE_DIR, "xray_images")
FINAL_OUTPUT_DIR = os.path.join(BASE_DIR, "preprocessed_images")

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)

# ============================
# ⚙️ تنظیمات پردازش
# ============================
TARGET_SIZE = (224, 224)
NUM_AUGS_PER_IMAGE = 3
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# ============================
# 📥 دانلود از Arvan
# ============================
def download_from_arvan():
    print(f"🔗 اتصال به Arvan bucket={BUCKET_NAME}, prefix={FOLDER_PREF}")
    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{REGION_HOST}",
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
    )

    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=FOLDER_PREF)

    total, downloaded, skipped = 0, 0, 0
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            total += 1
            filename = os.path.basename(key)
            out_path = os.path.join(DOWNLOAD_DIR, filename)
            if os.path.exists(out_path):
                skipped += 1
                continue
            s3.download_file(BUCKET_NAME, key, out_path)
            downloaded += 1

    print(f"📊 کل: {total} | دانلود جدید: {downloaded} | پرش: {skipped}")

# ============================
# 🖼 توابع پردازش تصویر
# ============================
def contour_based_crop(img_gray):
    _, thresh = cv2.threshold(img_gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_gray
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return img_gray[y:y+h, x:x+w]

def resize_or_pad(img, size=(224, 224)):
    h, w = img.shape
    target_h, target_w = size
    if h >= target_h and w >= target_w:
        return cv2.resize(img, size)
    delta_h = max(0, target_h - h)
    delta_w = max(0, target_w - w)
    top, bottom = delta_h // 2, delta_h - delta_h // 2
    left, right = delta_w // 2, delta_w - delta_w // 2
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

def preprocess_one(img_gray):
    img = contour_based_crop(img_gray)
    img = resize_or_pad(img, TARGET_SIZE)
    img = CLAHE.apply(img)
    img = cv2.medianBlur(img, 5)
    img = cv2.bilateralFilter(img, 9, 75, 75)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img

augment = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.5),
    A.GaussianBlur(blur_limit=3, p=0.3),
    A.Affine(scale=(0.9, 1.1), translate_percent=0.05, p=0.5),
    A.ElasticTransform(alpha=1.0, sigma=50, alpha_affine=50, p=0.5),
    A.GridDistortion(distort_limit=0.3, p=0.3),
])

# ============================
# 🚀 اجرای اصلی
# ============================
def main():
    # دانلود
    download_from_arvan()

    # پردازش
    files = [f for f in os.listdir(DOWNLOAD_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"🗂 تعداد فایل‌ها: {len(files)}")

    for fname in tqdm(files, desc="Processing"):
        in_path = os.path.join(DOWNLOAD_DIR, fname)
        img_bgr = cv2.imread(in_path)
        if img_bgr is None:
            print(f"⚠️ خواندن ناموفق: {fname}")
            continue

        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_final = preprocess_one(img_gray)

        base_name, _ = os.path.splitext(fname)
        out_base = os.path.join(FINAL_OUTPUT_DIR, f"{base_name}.jpg")
        cv2.imwrite(out_base, img_final)

        # Augmentation
        for i in range(NUM_AUGS_PER_IMAGE):
            aug_img = augment(image=img_final)['image']
            aug_name = os.path.join(FINAL_OUTPUT_DIR, f"{base_name}_aug{i+1}.jpg")
            cv2.imwrite(aug_name, aug_img)

    print(f"✅ پردازش تمام شد. خروجی در {FINAL_OUTPUT_DIR}")

if __name__ == "__main__":
    main()
