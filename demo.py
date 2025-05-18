import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# ----------------------------
# 1. MODELİ TANIMLA ve YÜKLE
# ----------------------------
model = models.densenet121(pretrained=False)
model.classifier = torch.nn.Linear(model.classifier.in_features, 2)  # 2 sınıf için

# Ağırlıkları yükle
model.load_state_dict(torch.load("densenet_epoch3.pt", map_location=torch.device('cpu')))
model.eval()

# ----------------------------
# 2. GÖRÜNTÜYÜ YÜKLE
# ----------------------------
img_path = "IM-0172-0001.jpeg"
image = Image.open(img_path).convert("RGB")  # RGB'ye çeviriyoruz
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    ])
input_tensor = transform(image).unsqueeze(0)

# ----------------------------
# 3. TAHMİN ET
# ----------------------------
with torch.no_grad():
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)
    label = "Hastalıklı" if predicted.item() == 1 else "Normal"

# ----------------------------
# 4. SONUCU GÖSTER
# ----------------------------
plt.imshow(image, cmap='gray')
plt.title(f"Tahmin: {label}")
plt.axis('off')
plt.show()
