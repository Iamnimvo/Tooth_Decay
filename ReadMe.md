
###`README.md`

```markdown
# 🦷 Tooth Decay Detection – Dental X-ray Pipeline

این پروژه یک **پایپ‌لاین کامل پردازش و یادگیری عمیق** برای تشخیص پوسیدگی دندان از روی تصاویر رادیوگرافی (X-ray) است.  
مراحل به ترتیب:
1. 📥 دانلود و پیش‌پردازش تصاویر (حذف نویز، نوار سفید، بهبود کنتراست، افزایش داده)  
2. 🧠 آموزش یک مدل طبقه‌بندی (ResNet18) برای تشخیص «سالم» یا «پوسیده»  
3. 🔍 پیش‌بینی و تولید خروجی‌های تصویری با استفاده از **Grad-CAM** (دایره، کانتور، فلش و هیت‌مپ)

---

## 📂 ساختار پروژه

```

Tooth\_Decay/
│
├── main.py                     # اجرای کل مراحل (preprocess + train + predict)
├── requirements.txt            # لیست کتابخانه‌ها
├── README.md                   # این فایل
├── .gitignore
│
├── preprocess/
│   ├── preprocess.py           # دانلود و پیش‌پردازش تصاویر
│   ├── xray\_images/            # تصاویر خام دانلودشده از Arvan
│   └── preprocessed\_images/    # تصاویر پردازش‌شده (224x224، augmentation)
│
├── processing/
│   ├── ToothClassifier.py      # تعریف مدل ResNet18
│   ├── train\_model.py          # آموزش مدل
│   ├── predict.py              # پیش‌بینی + Grad-CAM
│   ├── tooth\_classifier.pth    # وزن‌های مدل ذخیره‌شده (بعد از train)
│   └── predict\_output/         # خروجی‌های پیش‌بینی
│       ├── <image1>/
│       │   ├── original.jpg
│       │   ├── prediction\_circle.jpg
│       │   ├── prediction\_heatmap.jpg
│       │   ├── prediction\_gradcam.jpg
│       │   ├── prediction\_arrow\.jpg
│       │   └── prediction\_arrow\_circle.jpg
│       └── <image2>/ ...
│
└── xray\_pic/                   # (اختیاری) اگر عکسی در این پوشه قرار بگیرد،
\# همان عکس‌ها به جای داده‌های پیش‌پردازش‌شده
\# برای predict استفاده می‌شوند.

````

---

## ⚙️ نصب و اجرا

### 1) نصب کتابخانه‌ها
ابتدا پکیج‌ها را نصب کنید (ترجیحاً در محیط مجازی):

```bash
pip install -r requirements.txt
````

> ⚠️ توجه: نسخه‌ی `numpy<2.0` انتخاب شده چون PyTorch فعلاً با NumPy 2.x ناسازگار است.

برای نصب PyTorch روی GPU/CUDA باید دستور مناسب را از [وب‌سایت PyTorch](https://pytorch.org/get-started/locally/) بگیرید. مثال (CUDA 11.8):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

### 2) اجرای کامل پروژه

با اجرای یک دستور، کل pipeline انجام می‌شود:

```bash
python main.py
```

این مراحل انجام می‌شود:

1. دانلود و پیش‌پردازش تصاویر از Arvan → `preprocess/preprocessed_images/`
2. آموزش مدل → `processing/tooth_classifier.pth`
3. پیش‌بینی و تولید خروجی‌های Grad-CAM → `processing/predict_output/`

---

### 3) حالت پیش‌بینی

* اگر داخل پوشه‌ی `xray_pic/` تصویری قرار دهید → همان‌ها استفاده می‌شوند.
* اگر این پوشه خالی باشد → به‌صورت تصادفی **۱۰۰ تصویر** از `preprocess/preprocessed_images/` انتخاب می‌شود و برای هر تصویر یک فولدر خروجی ساخته می‌شود.

---

## 🔍 خروجی‌های Grad-CAM

برای هر تصویر ورودی چندین خروجی ساخته می‌شود:

* **original.jpg** → تصویر اصلی
* **prediction\_circle.jpg** → ناحیه مشکوک با دایره قرمز و کانتور سبز
* **prediction\_heatmap.jpg** → هیت‌مپ Grad-CAM
* **prediction\_gradcam.jpg** → هیت‌مپ روی تصویر اصلی با دایره/کانتور
* **prediction\_arrow\.jpg** → فلش اشاره به ناحیه مشکوک
* **prediction\_arrow\_circle.jpg** → ترکیب دایره + فلش

---

## 🧪 نکات فنی

* معماری استفاده‌شده: **ResNet18**
* داده‌ها با استفاده از **Albumentations** افزایش داده شده‌اند.
* پیش‌پردازش شامل: برش نوار سفید، CLAHE، فیلتر Median + Bilateral، Gaussian Blur
* برای جلوگیری از اشتباه روی پس‌زمینه‌ی سیاه، از **ROI Mask** (ناحیه فک/دندان) استفاده شده است.

---

## ❗ Troubleshooting (رفع خطاها)

🔴 **خطای NumPy با PyTorch**

* پیام: *"A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x..."*
* راه‌حل:

  ```bash
  pip install "numpy<2.0"
  ```

🔴 **خطای دانلود وزن ResNet18**

* اگر اینترنت یا تحریم مشکل ایجاد کرد، می‌توانید:

  * وزن‌های ResNet18 را دستی دانلود کنید و در مسیر
    `~/.cache/torch/hub/checkpoints/resnet18-*.pth` قرار دهید.
  * یا در کد `ToothClassifier.py` پارامتر `use_imagenet_weights=False` بدهید.

🔴 **دانلود از Arvan کار نمی‌کند**

* مطمئن شوید کلیدهای `access_key` و `secret_key` درست هستند.
* پوشه‌ی `xray_images/` را چک کنید که تصاویر ذخیره شده باشند.
* اگر همچنان مشکل بود، `DOWNLOAD_FROM_S3 = False` تنظیم کنید و تصاویر را دستی در `preprocess/xray_images/` بریزید.

🔴 **GPU پیدا نشد / فقط CPU استفاده می‌شود**

* مطمئن شوید PyTorch GPU نصب کرده‌اید (با دستور مخصوص CUDA).
* بررسی کنید کارت گرافیک توسط `nvidia-smi` شناسایی شود.
* اگر GPU ندارید → روی CPU هم اجرا می‌شود، فقط کندتر.

🔴 **مصرف زیاد RAM/CPU در زمان اجرا**

* تعداد `num_workers` در DataLoader (`train_model.py`) را کاهش دهید (حتی به 0).
* سایز Batch (`BATCH_SIZE`) را کوچک‌تر کنید.
* مرورگر یا برنامه‌های سنگین دیگر را همزمان باز نکنید.

---

## 🎯 مسیرهای بعدی

* بهبود دیتاست با برچسب‌های دقیق‌تر (Bounding Box یا Mask دندان‌ها)
* جایگزینی مدل طبقه‌بندی با مدل‌های **Object Detection** (YOLOv8, Faster R-CNN) برای تعیین دقیق‌تر محل پوسیدگی
* استفاده از **Focal Loss** برای بالانس کلاس‌ها

---

```

