import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class PneumoniaResNet50(nn.Module):
    def __init__(self):
        super(PneumoniaResNet50, self).__init__()
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)  # YENİ SİSTEM 

        # İlk konvolüsyon katmanını 1 kanallı yapıyoruz
        self.model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Son tam bağlantı katmanını 2 sınıfa göre değiştiriyoruz
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x):
        return self.model(x)
