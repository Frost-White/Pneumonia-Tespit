import torch.nn as nn
import torchvision.models as models

class PneumoniaResNet(nn.Module):
    def __init__(self):
        super(PneumoniaResNet, self).__init__()
        self.model = models.resnet18(pretrained=True)

        # 1 KANALLI X-ray görüntüler için ilk katmanı değiştirdik
        self.model.conv1 = nn.Conv2d(
            in_channels=1,  # 3 yerine 1 kanal
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Son katmanı da 2 sınıfa göre ayarlıyoruz
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x):
        return self.model(x)
