# preprocess/preprocess.py
# -*- coding: utf-8 -*-
"""
Preprocess X-ray images (Arvan download + ROI-masked pipeline):
 - دانلود از Arvan (S3-compatible) به preprocess/xray_images
 - ساخت ماسک ROI فک/دندان و صفر کردن بیرون آن
 - تبدیل به grayscale، حذف حاشیه‌های روشن، resize/pad به 224x224
 - CLAHE + کاهش نویز (Median/Bilateral/Gaussian)
 - ذخیره نسخه اصلی پردازش‌شده + N تصویر Augmented در preprocess/preprocessed_images
"""

import os
import cv2
import boto3
import numpy as np
import albumentations as A
from tqdm import tqdm

# ============================
# 🔐 تنظیمات Arvan (هاردکُد طبق خواسته‌ی شما)
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
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

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
# 🖼 توابع ماسک/پیش‌پردازش
# ============================
def build_jaw_roi(gray: np.ndarray) -> np.ndarray:
    """
    ماسک ROI ناحیه‌ی فک/دندان: GaussianBlur + Otsu + morphology + بزرگ‌ترین مؤلفه
    خروجی: ماسک 0/255 هم‌ابعاد gray
    """
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: 
        return mask
    c = max(cnts, key=cv2.contourArea)
    jaw = np.zeros_like(mask)
    cv2.drawContours(jaw, [c], -1, 255, -1)
    return jaw

def contour_based_crop(img_gray: np.ndarray) -> np.ndarray:
    # حذف حاشیه‌های سفیدِ خیلی روشن (اگر وجود داشته باشد)
    _, thresh = cv2.threshold(img_gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_gray
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return img_gray[y:y+h, x:x+w]

def resize_or_pad(img: np.ndarray, size=(224, 224)) -> np.ndarray:
    h, w = img.shape[:2]
    th, tw = size
    if h >= th and w >= tw:
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    # پدینگ برای کوچکترها
    dh, dw = max(0, th - h), max(0, tw - w)
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

def preprocess_one(img_gray: np.ndarray) -> np.ndarray:
    """
    پایپ‌لاین نهایی (پس از اعمال ROI mask):
     - حذف حاشیه‌ی سفید
     - Resize/Pad
     - CLAHE
     - Median + Bilateral + Gaussian
    """
    img = contour_based_crop(img_gray)
    img = resize_or_pad(img, TARGET_SIZE)
    img = CLAHE.apply(img)
    img = cv2.medianBlur(img, 5)
    img = cv2.bilateralFilter(img, 9, 75, 75)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img

# افزایش داده روی تصویر خاکستریِ نهایی
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
    # 1) دانلود از Arvan
    download_from_arvan()

    # 2) گردآوری فایل‌های تصویری
    files = [f for f in os.listdir(DOWNLOAD_DIR) if f.lower().endswith(IMG_EXTS)]
    files.sort()
    print(f"🗂 تعداد فایل‌ها: {len(files)}")

    # 3) پردازش + ROI masking
    for fname in tqdm(files, desc="Processing"):
        in_path = os.path.join(DOWNLOAD_DIR, fname)
        img_bgr = cv2.imread(in_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"⚠️ خواندن ناموفق: {fname}")
            continue

        # خاکستری
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # ✅ هماهنگ با predict: ماسک ROI فک/دندان و صفرکردن بیرون
        roi = build_jaw_roi(img_gray)
        img_gray_masked = cv2.bitwise_and(img_gray, img_gray, mask=roi)

        # پیش‌پردازش نهایی
        img_final = preprocess_one(img_gray_masked)

        # ذخیره نسخه اصلی
        base_name, _ = os.path.splitext(fname)
        out_base = os.path.join(FINAL_OUTPUT_DIR, f"{base_name}.jpg")
        cv2.imwrite(out_base, img_final)

        # Augmentation (روی نسخه‌ی نهاییِ تک‌کاناله)
        for i in range(NUM_AUGS_PER_IMAGE):
            aug_img = augment(image=img_final)['image']
            aug_name = os.path.join(FINAL_OUTPUT_DIR, f"{base_name}_aug{i+1}.jpg")
            cv2.imwrite(aug_name, aug_img)

    print(f"✅ پردازش تمام شد. خروجی در «{FINAL_OUTPUT_DIR}»")

if __name__ == "__main__":
    main()
