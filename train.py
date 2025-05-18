import torch
import torch.nn as nn
from torch.optim import Adam
from models.cnn_model_5kat import PneumoniaCNN
from utils.dataset_loader import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, _ = get_dataloaders("dataset", batch_size=32)

model = PneumoniaCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader)}")
    torch.save(model.state_dict(), f"5katmanagırlık/5kat{epoch}.pt")



