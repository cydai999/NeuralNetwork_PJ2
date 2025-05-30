import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm

from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm
from loader.loaders import get_cifar_loader

import argparse

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_random_seeds(42)

# load data
parser = argparse.ArgumentParser()
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--save_dir', type=str, default='vgg')

args = parser.parse_args()

num_workers = args.num_workers
batch_size = args.batch_size
train_loader = get_cifar_loader(root='../../../data', train=True, batch_size=batch_size, num_workers=num_workers)
val_loader = get_cifar_loader(root='../../../data', train=False, batch_size=batch_size, num_workers=num_workers)
print("Dataset has been loaded!")

# train function
def train(model, optimizer, criterion, train_loader, val_loader, save_path, scheduler=None, epochs_n=100):
    model.to(device)
    learning_curve = [0] * epochs_n    # training loss
    train_accuracy_curve = [0] * epochs_n    # training accuracy
    val_accuracy_curve = [0] * epochs_n    # valid accuracy
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)   # batch size
    losses_list = []    # use this to record the loss value of each epoch
    grads_list = []   # use this to record the loss gradient of each epoch
    for epoch in tqdm(range(epochs_n), unit='epoch'):
        # train
        print(f'[Train]Epoch:{epoch+1}')
        model.train()

        loss_list = []  # use this to record the loss value of each step
        total_train = 0
        correct_train = 0

        for data in tqdm(train_loader):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)

            # record loss
            loss_list.append(loss.item())
            learning_curve[epoch] += loss.item()

            # record training accuracy
            total_train += len(y)
            _, pred_y = torch.max(prediction, 1)
            correct_train += (pred_y == y).sum().item()

            # optimize
            loss.backward()
            optimizer.step()

        losses_list.append(loss_list)

        learning_curve[epoch] /= batches_n
        train_accuracy_curve[epoch] = correct_train / total_train
        print(f"[Train]Training loss:{learning_curve[epoch]}")
        print(f"[Train]Training accuracy:{train_accuracy_curve[epoch]}")

        # valid
        model.eval()

        total_valid = 0
        correct_valid = 0
        for data in tqdm(val_loader):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                prediction = model(x)
            _, pred_y = torch.max(prediction, 1)
            total_valid += len(y)
            correct_valid += (pred_y == y).sum().item()
        valid_acc = correct_valid / total_valid
        val_accuracy_curve[epoch] = valid_acc
        print(f"[Valid]Valid accuracy:{val_accuracy_curve[epoch]}")

        if valid_acc > max_val_accuracy:
            max_val_accuracy = valid_acc
            max_val_accuracy_epoch = epoch + 1

        # schedule the learning rate
        if scheduler is not None:
            scheduler.step()

    print("Training completed!")
    print(f"The best accuracy on valid set is {max_val_accuracy} in epoch {max_val_accuracy_epoch}.")

    # plot
    plt.subplot(2, 1, 1)
    plt.plot(learning_curve)
    plt.title('Learning curve')
    plt.subplot(2, 2, 3)
    plt.plot(train_accuracy_curve)
    plt.title('Train accuracy')
    plt.subplot(2, 2, 4)
    plt.plot(val_accuracy_curve)
    plt.title('Valid accuracy')
    plt.tight_layout()
    plt.savefig(save_path)

    return losses_list, grads_list

# Train
epo = args.epochs
save_dir = args.save_dir
os.makedirs(save_dir, exist_ok=True)

model = VGG_A()
lr = args.lr
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
criterion = nn.CrossEntropyLoss()
fg_save_path = os.path.join(save_dir, f'curve_{lr}.png')
loss, grads = train(model, optimizer, criterion, train_loader, val_loader, fg_save_path, epochs_n=epo)

np.savetxt(os.path.join(save_dir, f'loss_{lr}.txt'), loss, fmt='%s', delimiter=' ')