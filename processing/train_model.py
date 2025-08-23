import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import cv2
import pandas as pd

# تنظیمات مسیرها
final_output_dir = "./preprocess/preprocessed_images"
results_file = "results.csv"

# تعریف مدل
class ToothClassifier(nn.Module):
    def __init__(self):
        super(ToothClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # 2 کلاس: پوسیدگی یا عدم پوسیدگی

    def forward(self, x):
        return self.model(x)

# تنظیمات پردازش داده‌ها
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485]*3, std=[0.229]*3)
])

# آماده‌سازی داده‌ها
image_paths = [os.path.join(final_output_dir, fname) for fname in os.listdir(final_output_dir) if fname.lower().endswith((".jpg", ".jpeg", ".png"))]
labels = [1 if "disease" in fname else 0 for fname in os.listdir(final_output_dir)]  # برچسب‌ها

# ایجاد DataLoader
dataset = [(transform(cv2.imread(image_path)), label) for image_path, label in zip(image_paths, labels)]
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# مدل و تنظیمات آموزش
model = ToothClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# آموزش مدل
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0  # برای شمارش پیش‌بینی‌های درست
    total_predictions = 0  # برای شمارش کل پیش‌بینی‌ها

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)

        # محاسبه loss
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # محاسبه پیش‌بینی‌های درست
        _, predicted = torch.max(outputs, 1)  # پیش‌بینی‌های مدل
        correct_predictions += (predicted == labels).sum().item()  # شمارش پیش‌بینی‌های درست
        total_predictions += labels.size(0)  # تعداد کل پیش‌بینی‌ها

        running_loss += loss.item()

    # محاسبه دقت
    accuracy = (correct_predictions / total_predictions) * 100  # دقت به درصد

    # چاپ زیان و دقت در هر اپوک
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

# ذخیره مدل آموزش‌دیده
torch.save(model.state_dict(), "tooth_classifier.pth")
print("Model saved to tooth_classifier.pth")