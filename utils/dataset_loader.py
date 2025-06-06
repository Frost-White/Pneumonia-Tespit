from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder(f"{data_dir}/train", transform=transform)
    val_data = datasets.ImageFolder(f"{data_dir}/val", transform=transform)
    test_data = datasets.ImageFolder(f"{data_dir}/test", transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_loader, val_loader, test_loader
