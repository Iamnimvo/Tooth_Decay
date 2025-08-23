import torch
import cv2
from torchvision import transforms
from ToothClassifier import ToothClassifier  # فرض می‌کنیم که مدل شما در همین مسیر است

# بارگذاری مدل ذخیره‌شده
model = ToothClassifier()
model.load_state_dict(torch.load("./processing/tooth_classifier.pth")) 
model.eval()  # مدل را در حالت ارزیابی قرار می‌دهیم

# پیش‌پردازش تصویر جدید
image_path = "./processing/new_image.jpg"  # مسیر تصویر جدید
img = cv2.imread(image_path)

# پیش‌پردازش تصویرُ
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485]*3, std=[0.229]*3)
])

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # تبدیل به گراسیلSS
img_3ch = cv2.merge([img_gray] * 3)  # تبدیل به سه کاناله
img_tensor = transform(img_3ch).unsqueeze(0)  # تبدیل به تنسور برای مدل

# پیش‌بینی با مدل
with torch.no_grad():
    outputs = model(img_tensor)  # پیش‌بینی
    _, predicted = torch.max(outputs, 1)  # دریافت پیش‌بینی‌ها

# چاپ نتیجه
if predicted.item() == 1:
    print("پوسیدگی دندان تشخیص داده شد.")
else:
    print("دندان بدون پوسیدگی است.")