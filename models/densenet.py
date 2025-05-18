import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

# DenseNet121 modelini yükle
def get_densenet_model(num_classes=2):
    model = densenet121(weights=DenseNet121_Weights.DEFAULT)  # pretrained=True daha hızlı convergence sağlar
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)  # Son katmanı 2 sınıfa göre değiştiriyoruz
    return model
