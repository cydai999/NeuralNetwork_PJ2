import argparse

import numpy as np
import random

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from models import resnet18

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
parser.add_argument('--model_path', type=str, default='best_model.pth')

args = parser.parse_args()

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

test_set = torchvision.datasets.CIFAR10(root='../../data', train=False, download=True, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
print("Dataset has been loaded!")

# init model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

model_path = args.model_path
model = resnet18.ResNet(dropout_rate=0.1)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)

print("Model has been loaded!")

# evaluate
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