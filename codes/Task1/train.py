import argparse
import os
import time
import numpy as np
from tqdm import tqdm
import json
import random
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms

from models import resnet18

from contextlib import redirect_stdout

# set seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU情况
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# load data
parser = argparse.ArgumentParser()
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=30)

args = parser.parse_args()
num_workers = args.num_workers
batch_size = args.batch_size

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

train_set = torchvision.datasets.CIFAR10(root='../../data', train=True, download=True, transform=train_transform)
val_set = torchvision.datasets.CIFAR10(root='../../data', train=True, download=True, transform=test_transform)

train_idx, val_idx = train_test_split(
    np.arange(len(train_set)),
    shuffle=True,
    test_size=0.2,
    random_state=42
)

train_set = Subset(train_set, train_idx)
val_set = Subset(val_set, val_idx)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

test_set = torchvision.datasets.CIFAR10(root='../../data', train=False, download=True, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
print("Dataset has been loaded!")

# init model
model = resnet18.ResNet(dropout_rate=0.1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
model.to(device)
# print(model)
print("Model has been initialized!")

# init optimizer and loss function
# optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.0)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.2)

# train
epochs = args.epochs
log_iter = 100
patience = 5
acc_delta = 0.1
best_acc = 0.0
print("Begin training!")
print("-"*20)

train_loss = []
train_avg_loss = []
valid_acc = []

log_dir = f'work_dir/{time.strftime("%Y%m%d%H%M", time.localtime())}'
os.makedirs(log_dir, exist_ok=True)

with open(f"{log_dir}/model_architecture.txt", 'w') as f:
    with redirect_stdout(f):
        print(model)

writer = SummaryWriter(log_dir=log_dir)

total_train_time = 0
for epoch in range(epochs):
    # train
    start_time = time.time()
    model.train()
    print(f"[Train]Epoch {epoch+1}...")
    total_loss = 0
    train_total = 0
    train_correct = 0
    for iter, (X_train, y_train) in tqdm(enumerate(train_loader)):
        X_train, y_train = X_train.to(device), y_train.to(device)
        optimizer.zero_grad()
        output = model(X_train)
        _, preds = torch.max(output, 1)
        train_total += len(X_train)
        train_correct += (preds == y_train).sum().item()
        loss = loss_fn(output, y_train)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

        if not iter % log_iter:
            print(f"[Train]iter:{iter}, loss:{loss.item()}")
            train_loss.append({'iter': epoch * len(train_loader) + iter, 'loss': loss.item()})
            writer.add_scalar('Train/batch_loss', loss.item(), epoch * len(train_loader) + iter)

    print(f"[Train]Average loss:{total_loss/len(train_loader):.4f}")
    train_avg_loss.append({'epoch': epoch + 1, 'loss': total_loss/len(train_set)})
    writer.add_scalar('Train/avg_loss', total_loss / len(train_loader), epoch)
    
    train_acc = 100 * train_correct / train_total
    print(f"[Train]Accuracy:{train_acc:.4f}")
    writer.add_scalar('Train/acc', train_acc, epoch)

    # valid
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for X_val, y_val in tqdm(val_loader):
            X_val, y_val = X_val.to(device), y_val.to(device)
            output = model(X_val)
            _, preds = torch.max(output, 1)
            total += len(X_val)
            correct += (preds == y_val).sum().item()
    acc = 100 * correct / total

    print(f"[Valid]Accuracy:{acc:.4f}")
    valid_acc.append({'epoch': epoch + 1, 'accuracy': acc})
    writer.add_scalar('Valid/acc', acc, epoch)

    scheduler.step()

    # save model
    if acc > best_acc + acc_delta:
        best_acc = acc
        torch.save(model.state_dict(), f"{log_dir}/best_model.pth")
        patience = 5
    else:
        patience -= 1
        if patience <= 0:
            print(f"Early stop at epoch {epoch+1}")
            break

    train_time = time.time() - start_time
    print(f"[Train]Training time at Epoch {epoch+1}:", train_time)
    total_train_time += train_time
    print("-"*20)

print("Training completed!")

# eval
model.eval()
total = 0
correct = 0
with torch.no_grad():
    for X_test, y_test in test_loader:
        X_test, y_test = X_test.to(device), y_test.to(device)
        output = model(X_test)
        _, preds = torch.max(output, 1)
        total += len(X_test)
        correct += (preds == y_test).sum().item()
acc = 100 * correct / total

print(f"[Test]Accuracy:{acc:.4f}")

lod_data = {
    'train_loss': train_loss,
    'train_avg_loss': train_avg_loss,
    'valid_acc': valid_acc
}


with open(f"{log_dir}/log.json", 'w') as f:
    json.dump(lod_data, f)

with open(f"{log_dir}/result.txt", 'w') as f:
    f.write(f'[Train]Total training time:{total_train_time}\n')
    f.write(f'[Train]Training epochs:{epoch+1}\n')
    f.write(f'[Test]Accuracy:{acc:.4f}')




