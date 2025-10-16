import torch.nn as nn
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms


class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2,2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2,2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2,2)
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(nn.Linear(128*16*16, 256), nn.ReLU(), nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5))
        self.fc3 = nn.Linear(128, 2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    trans = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    train_set = ImageFolder(root='data/train',transform=trans)
    valid_set = ImageFolder(root='data/valid',transform=trans)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    train_N = len(train_loader.dataset)
    valid_N = len(valid_loader.dataset)
        
    model = cnn()
    model.to(device)
    LossFunction = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters())
    model = model.to(device)

    def get_batch_accuracy(output, y,n):
        pred = output.argmax(dim=1)
        correct = pred.eq(y).sum().item()
        return correct / n      


    def train():
        loss = 0
        accuracy = 0
        
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            optimizer.zero_grad()
            batch_Loss = LossFunction(output,y)
            batch_Loss.backward()
            optimizer.step()
            loss += batch_Loss.item()
            accuracy += get_batch_accuracy(output,y,train_N)
        print('Train - Loss: {:.4f} Accuracy: {:.4f}'.format(loss,accuracy))

    def validate():
        loss = 0
        accuracy = 0

        model.eval()
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss += LossFunction(output,y).item()
                accuracy += get_batch_accuracy(output,y,valid_N)
        print('valid - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))

    epochs = 20

    for epoch in range(epochs):
        print('Epoch: {}'.format(epoch))
        train()
        validate()

    torch.save(model.state_dict(), "catdog_model.pth")
