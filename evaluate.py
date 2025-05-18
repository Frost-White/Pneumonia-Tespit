import torch
import os
from sklearn.metrics import classification_report, f1_score
from models.cnn_model_5kat import PneumoniaCNN
from utils.dataset_loader import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, _, test_loader = get_dataloaders("dataset")

weights_folder = "5katmanagırlık"
best_f1 = 0.0
best_model_name = ""

for weight_file in sorted(os.listdir(weights_folder)):
    if weight_file.endswith(".pt"):
        model = PneumoniaCNN().to(device)
        model.load_state_dict(torch.load(os.path.join(weights_folder, weight_file)))
        model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"{weight_file} -> F1 Score: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = weight_file

print(f"\nBest model: {best_model_name} with F1 Score: {best_f1:.4f}")

# En iyi modeli detaylı classification report ile göster
model = PneumoniaCNN().to(device)
model.load_state_dict(torch.load(os.path.join(weights_folder, best_model_name)))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\nDetailed Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["Normal", "Pneumonia"]))
