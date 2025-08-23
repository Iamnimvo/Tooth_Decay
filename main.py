import subprocess
import os

# مسیرهای پوشه‌ها
preprocess_dir = "./preprocess"
processing_dir = "./processing"

# اجرای کد پیش‌پردازش
print("⚙️ Running preprocess.py...")
subprocess.run(["python", os.path.join(preprocess_dir, "preprocess.py")])

# اجرای کد آموزش مدل
print("⚙️ Running train_model.py...")
subprocess.run(["python", os.path.join(processing_dir, "train_model.py")])

# اجرای کد مدل
print("⚙️ Running ToothClassifier.py...")
subprocess.run(["python", os.path.join(processing_dir, "ToothClassifier.py")])

# اجرای کد پیش‌بینی
print("⚙️ Running predict.py...")
subprocess.run(["python", os.path.join(processing_dir, "predict.py")])

print("✅ همه مراحل انجام شد.")