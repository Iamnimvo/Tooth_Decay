# processing/predict.py
# -*- coding: utf-8 -*-
"""
Predict + Grad-CAM with ROI gating.
اولویت ورودی:
  1) اگر در ROOT/xray_pic/ عکس هست → همه را پردازش کن (برای هر عکس یک فولدر خروجی)
  2) وگرنه از preprocess/preprocessed_images/ حداکثر 100 عکس تصادفی پردازش کن.
خروجی‌ها در processing/predict_output/<image_stem>/ ذخیره می‌شوند:
  - original.jpg
  - prediction_heatmap.jpg
  - prediction_circle.jpg        (دایره قرمز + کانتور سبز)
  - prediction_gradcam.jpg       (overlay + دایره/کانتور)
  - prediction_arrow.jpg
  - prediction_arrow_circle.jpg
"""

import os, cv2, torch, numpy as np, random
from torchvision import transforms
from ToothClassifier import ToothClassifier

# ===================== تنظیمات بصری/تحلیلی =====================
CAM_POWER    = 2.0     # >1: تیزتر کردن ناحیه‌های داغ Grad-CAM
THRESH_PCT   = 92      # آستانه به‌صورت صدک (۹۰–۹۸ پیشنهاد)
HEAT_ALPHA   = 0.35    # شفافیت هیت‌مپ روی تصویر اصلی
MORPH_KERNEL = 5       # کرنل عملیات مورفولوژی
MIN_AREA     = 80      # حداقل مساحت کانتور ROI
MAX_BATCH    = 100     # حداکثر تعداد تصویر از پوشه‌ی preprocess

# ===================== مسیرها =====================
PROC_DIR     = os.path.dirname(__file__)                      # processing/
ROOT_DIR     = os.path.dirname(PROC_DIR)                      # ریشه پروژه
MODEL_PATH   = os.path.join(PROC_DIR, "tooth_classifier.pth")

XRAY_PICK_DIR = os.path.join(ROOT_DIR, "xray_pic")
PREP_IMG_DIR  = os.path.join(ROOT_DIR, "preprocess", "preprocessed_images")

OUT_ROOT    = os.path.join(PROC_DIR, "predict_output")
os.makedirs(OUT_ROOT, exist_ok=True)

# ===================== مدل و دیوایس =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ToothClassifier()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval().to(device)

# ===================== ترنسفورم ورودی =====================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485]*3, std=[0.229]*3),
])

# ===================== ابزارها =====================
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def list_images(d):
    if not os.path.isdir(d): return []
    return [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(IMG_EXTS)]

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def build_jaw_roi(gray: np.ndarray) -> np.ndarray:
    """
    ساخت ماسک ROI برای ناحیه‌ی فک/دندان تا Grad-CAM بیرون فک را نگیرد.
    روش: GaussianBlur + Otsu + morphology + بزرگ‌ترین مؤلفه‌ی متصل
    خروجی: ماسک 0/255 هم‌اندازه‌ی gray
    """
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    # دندان/فک نسبت به پس‌زمینه‌ی سیاه روشن‌ترند
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

def draw_arrow(img, target_pt, color=(0,0,255)):
    h, w = img.shape[:2]; cx, cy = target_pt
    start = (int(0.1*w), cy) if cx >= w//2 else (int(0.9*w), cy)
    cv2.arrowedLine(img, start, (cx, cy), color, 2, tipLength=0.03)

# ===================== پردازش یک تصویر =====================
def process_one_image(img_path: str, out_dir: str):
    ensure_dir(out_dir)

    # --- خواندن و ذخیره اصل ---
    orig = cv2.imread(img_path)
    if orig is None:
        print(f"⚠️ خواندن تصویر ناموفق: {img_path}")
        return
    cv2.imwrite(os.path.join(out_dir, "original.jpg"), orig)

    # --- آماده‌سازی ورودی مدل ---
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    img3 = cv2.merge([gray, gray, gray])
    inp  = transform(img3).unsqueeze(0).to(device)

    # --- پیش‌بینی کلاس ---
    with torch.no_grad():
        logits = model(inp)
    pred = logits.argmax(1).item()
    print("   └─ نتیجه:", "پوسیدگی" if pred == 1 else "سالم")

    # --- Hookهای Grad-CAM روی آخرین لایه کانولوشن ---
    target_layer = model.model.layer4[-1].conv2
    acts, grads = [], []
    def fwd_hook(m, i, o): acts.append(o.detach())
    def bwd_hook(m, gi, go): grads.append(go[0].detach())
    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    # --- Backward برای کلاس پیش‌بینی‌شده ---
    model.zero_grad(set_to_none=True)
    score = model(inp)[0, pred]
    score.backward()

    # --- محاسبه CAM ---
    A = acts[0].squeeze(0).cpu().numpy()     # (C,H,W)
    G = grads[0].squeeze(0).cpu().numpy()    # (C,H,W)
    w = G.mean(axis=(1,2))                   # (C,)
    cam = np.maximum((w[:, None, None] * A).sum(0), 0)
    cam = cv2.resize(cam, (orig.shape[1], orig.shape[0]))
    cam -= cam.min()
    cam /= (cam.max() + 1e-6)
    cam = np.power(cam, CAM_POWER)

    # --- ROI gating: محدودکردن CAM به فک/دندان ---
    roi_mask = build_jaw_roi(gray).astype(np.float32) / 255.0
    cam = cam * roi_mask

    # --- Threshold + Morphology ---
    thr  = np.percentile(cam[cam > 0], THRESH_PCT) if np.any(cam > 0) else 1.0
    mask = (cam >= thr).astype(np.uint8) * 255
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)

    # --- یافتن ROI (کانتور یا fallback به بیشینه) ---
    circle_img = orig.copy()
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center, r, best_cnt = None, None, None

    if cnts:
        cnts = [c for c in cnts if cv2.contourArea(c) >= MIN_AREA]
        if cnts:
            best_cnt = max(cnts, key=cv2.contourArea)
            (x, y), r = cv2.minEnclosingCircle(best_cnt)
            center, r = (int(x), int(y)), int(r)

    if center is None:
        # نقطه‌ی بیشینه داخل ROI
        valid = np.where(roi_mask > 0)
        if valid[0].size:
            # مختصات پیکسل بیشینه‌ی CAM
            yy, xx = np.unravel_index(np.argmax(cam), cam.shape)
            center = (int(xx), int(yy))
        else:
            # اگر ROI هم نبود، وسط تصویر
            h, w = cam.shape
            center = (w // 2, h // 2)
        r = max(int(0.05 * max(orig.shape[:2])), 12)

    # --- ترسیم دایره/کانتور ---
    cv2.circle(circle_img, center, r, (0, 0, 255), 2)
    if best_cnt is not None:
        cv2.drawContours(circle_img, [best_cnt], -1, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(out_dir, "prediction_circle.jpg"), circle_img)

    # --- هیت‌مپ و همپوشانی ---
    heat = (cam * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig, 1.0 - HEAT_ALPHA, heat, HEAT_ALPHA, 0)
    cv2.circle(overlay, center, r, (0, 0, 255), 2)
    if best_cnt is not None:
        cv2.drawContours(overlay, [best_cnt], -1, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(out_dir, "prediction_heatmap.jpg"), heat)
    cv2.imwrite(os.path.join(out_dir, "prediction_gradcam.jpg"), overlay)

    # --- فلش‌ها ---
    arrow = orig.copy()
    draw_arrow(arrow, center, (0,0,255))
    cv2.imwrite(os.path.join(out_dir, "prediction_arrow.jpg"), arrow)

    arrow_circle = circle_img.copy()
    draw_arrow(arrow_circle, center, (0,0,255))
    cv2.imwrite(os.path.join(out_dir, "prediction_arrow_circle.jpg"), arrow_circle)

    # --- پاک کردن hook ها ---
    h1.remove(); h2.remove()

# ===================== اجرای اصلی =====================
def main():
    # 1) اگر xray_pic/ عکس دارد → همه را پردازش کن
    pick_imgs = list_images(XRAY_PICK_DIR)
    if pick_imgs:
        print(f"📂 xray_pic: {len(pick_imgs)} تصویر پیدا شد.")
        for p in pick_imgs:
            stem = os.path.splitext(os.path.basename(p))[0]
            out_dir = os.path.join(OUT_ROOT, stem)
            print(f"→ پردازش: {p}")
            process_one_image(p, out_dir)
        print(f"✅ تمام شد. خروجی‌ها در: {OUT_ROOT}")
        return

    # 2) در غیر این صورت از preprocessed_images حداکثر MAX_BATCH تصویر تصادفی
    all_pre = list_images(PREP_IMG_DIR)
    if not all_pre:
        raise FileNotFoundError("⛔ هیچ تصویری در preprocess/preprocessed_images پیدا نشد.")

    k = min(MAX_BATCH, len(all_pre))
    batch = random.sample(all_pre, k)
    print(f"📂 preprocessed_images: {len(all_pre)} تصویر | انتخاب تصادفی {k} عدد")

    for p in batch:
        stem = os.path.splitext(os.path.basename(p))[0]
        out_dir = os.path.join(OUT_ROOT, stem)
        print(f"→ پردازش: {p}")
        process_one_image(p, out_dir)

    print(f"✅ تمام شد. {k} فولدر خروجی داخل: {OUT_ROOT}")

if __name__ == "__main__":
    main()
