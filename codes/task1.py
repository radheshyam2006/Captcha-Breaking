import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import os
from PIL import Image


DATASET_DIR = input("Enter the directory name: ")
CSV_FILE = os.path.join(DATASET_DIR, "labels.csv")
epochs = int(input("Enter number of epochs: "))

df = pd.read_csv(CSV_FILE)

df["TextLabel"] = df["TextLabel"].str.lower()
df["ImageName"] = df["ImageName"].str.lower()


samelabels = df["TextLabel"].unique()
label_map = dict.fromkeys(samelabels)
for idx, word in enumerate(label_map):
    label_map[word] = idx
df["Label"] = df["TextLabel"].map(label_map)


class DATA(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.root_dir = root_dir
        self.data["TextLabel"] = self.data["TextLabel"].str.lower()
        self.data["ImageName"] = self.data["ImageName"].str.lower()
        
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        
        name = self.data.iloc[idx]["ImageName"]
        path = os.path.join(self.root_dir, name)
        if not os.path.exists(path):
            print(f"Error: {path} not found!")
            return None 

        image = Image.open(path).convert("RGB")
        label = label_map[self.data.iloc[idx]["TextLabel"]]

        if self.transform:
            image = self.transform(image)

        return image, label, name


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])


dataset = DATA(csv_file=CSV_FILE, root_dir=DATASET_DIR, transform=transform)
vsize = int(0.15 * len(dataset))
tsize = len(dataset) - vsize
tdata, vdata = random_split(dataset, [tsize, vsize])

vload = DataLoader(vdata, batch_size=16, shuffle=False)
tload = DataLoader(tdata, batch_size=16, shuffle=True)



class CNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

num_classes = len(samelabels)
model = CNNClassifier(num_classes)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)

def get_loss_function():
    return nn.CrossEntropyLoss()

def get_optimizer(model, lr=0.001):
    return optim.Adam(model.parameters(), lr=lr)

criterion = get_loss_function()
optimizer = get_optimizer(model)

start_epoch = 0


def training(model, tload, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for ba in tload:
        if ba is None:
            continue
        
        images, labels, _ = ba
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / len(tload), 100 * correct / total

for epoch in range(start_epoch, epochs):
    tloss, tacc = training(model, tload, criterion, optimizer, device)
    print(f"Epoch {epoch+1}/{epochs} | Loss: {tloss:.4f} | Accuracy: {tacc:.2f}%")


model.eval()
predictions = []
correct, total = 0, 0

with torch.no_grad():
    for d in vload:
        
        images, labels, names = d
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        for i in range(len(labels)):
            prediction = samelabels[predicted[i].item()]
            predictions.append([names[i], prediction])
            if predicted[i] == labels[i]:
                correct += 1
            total += 1

val_accuracy = 100 * correct / total
print(f"Validation Accuracy: {val_accuracy:.2f}%")

predictions_df = pd.DataFrame(predictions, columns=["Image Name", "Predicted Label"])
predictions_df.to_csv("predictions.csv", index=False)
print("Predictions saved to predictions.csv")
