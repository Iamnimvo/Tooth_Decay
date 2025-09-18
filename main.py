# main.py
# -*- coding: utf-8 -*-

import subprocess
import sys
import os

ROOT_DIR = os.path.dirname(__file__)

# مسیر اسکریپت‌ها
PREPROCESS = os.path.join(ROOT_DIR, "preprocess", "preprocess.py")
TRAIN      = os.path.join(ROOT_DIR, "processing", "train_model.py")
PREDICT    = os.path.join(ROOT_DIR, "processing", "predict.py")

def run_step(script_path: str, title: str):
    print(f"\n{'='*60}\n▶️  {title}\n{'='*60}")
    try:
        result = subprocess.run([sys.executable, script_path], check=True)
        if result.returncode == 0:
            print(f"✅ {title} با موفقیت اجرا شد.\n")
    except subprocess.CalledProcessError as e:
        print(f"⛔ خطا در {title}: {e}\n")
        sys.exit(1)

def main():
    run_step(PREPROCESS, "مرحله 1: پیش‌پردازش تصاویر")
    run_step(TRAIN, "مرحله 2: آموزش مدل")
    run_step(PREDICT, "مرحله 3: پیش‌بینی و Grad-CAM")

    print("\n🎉 همه مراحل با موفقیت انجام شدند.")

if __name__ == "__main__":
    main()
