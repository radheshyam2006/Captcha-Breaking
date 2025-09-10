import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import pandas as pd
import os
import numpy as np
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

imagedir=input("Enter directory name:")
csvfile = os.path.join(imagedir, "labels.csv")
epochs = int(input("Enter number of epochs:"))

df = pd.read_csv(csvfile, header=None, names=["ImageName", "TextLabel"])

CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
cti = {}
for idx, char in enumerate(CHARACTERS):
    cti[char] = idx + 1

cti["<PAD>"] = 0  
itc = {}
for char, idx in cti.items():
    itc[idx] = char


df["TextLabel"] = df["TextLabel"].fillna("")  
MAX_TEXT_LEN = max(df["TextLabel"].apply(lambda x: len(str(x)))) 


def setup(model, lr=0.001, step_size=5, gamma=0.1):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return criterion, optimizer, scheduler

class TID(Dataset):
    def __init__(self, csvfile, imagedir, transform=None):
        self.transform = transform
        self.imagedir = imagedir
        self.data = pd.read_csv(csvfile, header=None, names=["ImageName", "TextLabel"])
        self.data["TextLabel"] = self.data["TextLabel"].fillna("").astype(str)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.imagedir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert("L")

        if self.transform:
            image = self.transform(image)

        
        text = self.data.iloc[idx, 1]
        encoded_text = []
        for char in text:
            if char in cti:
                encoded_text.append(cti[char])

        encoded_text += [0] * (MAX_TEXT_LEN - len(encoded_text))  

        return image, torch.tensor(encoded_text)


transform = transforms.Compose([
    transforms.Resize((32, 128)),  
    transforms.ToTensor(),
    transforms.RandomRotation(5), 
    transforms.Normalize((0.5,), (0.5,))
])

dataset = TID(csvfile, imagedir, transform)

tsize = int(0.8 * len(dataset))
vsize = int(0.1 * len(dataset))
testsize = len(dataset) - tsize - vsize  

train_dataset, val_dataset, testdata = random_split(dataset, [tsize, vsize, testsize])


tload = DataLoader(train_dataset, batch_size=20, shuffle=True)
vload = DataLoader(val_dataset, batch_size=20, shuffle=False)
testload = DataLoader(testdata, batch_size=20, shuffle=False)


class impmodel(nn.Module):
    def __init__(self, num_chars):
        super(impmodel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  
        )
        
        self.rnn = nn.LSTM(64 * 8, 128, bidirectional=True, num_layers=2, dropout=0.2, batch_first=True)
        self.fc = nn.Linear(256, num_chars)  

    def forward(self, x):
        x = self.cnn(x) 
        x = x.permute(0, 3, 1, 2).contiguous() 
        x = x.view(x.size(0), x.size(1), -1)  
        
        x, _ = self.rnn(x)  
        x = self.fc(x)  
        
        return x.permute(1, 0, 2)  

num_classes = len(CHARACTERS) + 1  
# print(num_classes)
model = impmodel(num_classes)




criterion, optimizer, scheduler = setup(model)




for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, labels in tload:
        optimizer.zero_grad()
        outputs = model(images)

        probability = F.log_softmax(outputs, dim=2)  # CTC loss expects log probabilities
        ilen = torch.full((images.size(0),), probability.size(0), dtype=torch.long)
        tlen = torch.sum(labels != 0, dim=1)

        loss = criterion(probability, labels, ilen, tlen)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    model.eval()
    total_chars = 0
    correct_chars = 0
    with torch.no_grad():
        for images, labels in vload:
            outputs = model(images)
            outputs = outputs.argmax(dim=2)  

            for i in range(outputs.size(1)):  
                prediction = ""
                for idx in outputs[:, i]:
                    if idx.item() != 0:
                        prediction += itc[idx.item()]

                realword = ""
                for idx in labels[i]:
                    if idx.item() != 0:
                        realword += itc[idx.item()]


                total_chars += len(realword)
                for a, b in zip(prediction, realword):
                    if a == b:
                        correct_chars += 1


    acc =100 * correct_chars / total_chars if total_chars > 0 else 0
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(tload)}, Val acc: {acc:.4f}")

def redorgreen(path):
    image = cv2.imread(path)
    (blue,green, red) = cv2.split(image)
    sumg = np.sum(green)
    sumr = np.sum(red)
    if sumg > sumr:
        return 0
    elif sumr > sumg:
        return 1


def predict_text(image_path, model):
    model.eval()
    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0)  

    with torch.no_grad():
        output = model(image)

    output = output.squeeze(1)  
    output = output.argmax(dim=1)  

    predword = []
    prev_char = None
    for idx in output:
        char_idx = idx.item()
        if char_idx != 0 and char_idx != prev_char:  
            predword.append(itc[char_idx])
        prev_char = char_idx
    
    x=redorgreen(image_path)
    if x==1:
        predword = predword[::-1]

    return "".join(predword)

model.eval()
total_chars = 0
correct_chars = 0

with torch.no_grad():
    for images, labels in testload:
        outputs = model(images)
        outputs = outputs.argmax(dim=2) 

        for i in range(outputs.size(1)):  
                prediction = ""
                for idx in outputs[:, i]:
                    if idx.item() != 0:
                        prediction += itc[idx.item()]

                realword = ""
                for idx in labels[i]:
                    if idx.item() != 0:
                        realword += itc[idx.item()]

                total_chars += len(realword)
                for a, b in zip(prediction, realword):
                    if a == b:
                        correct_chars += 1
                


test_acc = 100 * correct_chars / total_chars if total_chars > 0 else 0
print(f"Test acc: {test_acc:.4f}%")



