# train_lenet5_kmnist_noisy.py
# -*- coding: utf-8 -*-
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.sigmoid = nn.Sigmoid()
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(6, 16, kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c5 = nn.Conv2d(16, 120, kernel_size=5)
        self.flatten = nn.Flatten()
        self.f6 = nn.Linear(120, 84)
        self.f7 = nn.Linear(84, num_classes)

    def forward(self, x, return_features=False):
        x = self.sigmoid(self.c1(x))
        x = self.s2(x)
        x = self.sigmoid(self.c3(x))
        x = self.s4(x)
        x = self.c5(x)
        x = self.flatten(x)
        features = self.f6(x)
        output = self.f7(features)
        if return_features:
            return output, features
        return output


def train_one_noise(train_root, save_path, epochs=20, batch_size=128, lr=1e-3,
                    val_ratio=0.1, seed=42, num_workers=4):

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Train root:", train_root)
    print("Save path :", save_path)

    transform_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomCrop(28, padding=2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    transform_eval = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    full_dataset = datasets.ImageFolder(root=train_root, transform=transform_train)
    num_classes = len(full_dataset.classes)
    print(f"Found {len(full_dataset)} images, {num_classes} classes: {full_dataset.classes}")

    val_len = int(len(full_dataset) * val_ratio)
    train_len = len(full_dataset) - val_len
    train_set, val_set = random_split(
        full_dataset, [train_len, val_len],
        generator=torch.Generator().manual_seed(seed)
    )
    val_set.dataset.transform = transform_eval

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    model = LeNet5(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = -1.0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()

        train_acc = total_correct / len(train_set)
        train_loss = total_loss / len(train_set)

        model.eval()
        val_correct = 0
        val_loss_sum = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                val_loss_sum += loss.item() * imgs.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()

        val_acc = val_correct / len(val_set)
        val_loss = val_loss_sum / len(val_set)

        print(f"Epoch {epoch}/{epochs} | "
              f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.4f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  ✅ Saved best model to {save_path} (val_acc={best_val_acc:.4f})")

    print(f"Done. Best val_acc = {best_val_acc:.4f}")
    return best_val_acc


def main():
    parser = argparse.ArgumentParser()

    # 你的截图：/ywt/dataset/{NoiseType}/KMNIST/train
    parser.add_argument("--data_root", type=str, required=True,
                        help="例如 /NVNUYucw/ywt/dataset")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)

    # ALL / RandomLabelNoise / RandomDataNoise / SpecificDataNoise / SpecificLabelNoise
    parser.add_argument("--noise_type", type=str, default="ALL")
    args = parser.parse_args()

    noise_list = [
        "RandomLabelNoise",
        "RandomDataNoise",
        "SpecificDataNoise",
        "SpecificLabelNoise",
    ]
    if args.noise_type != "ALL":
        noise_list = [args.noise_type]

    for noise in noise_list:
        train_root = os.path.join(args.data_root, noise, "KMNIST", "train")
        save_path  = os.path.join(args.data_root, noise, "KMNIST", "LeNet5.pth")

        if not os.path.isdir(train_root):
            print(f"\n❌ Skip {noise}: train_root not found -> {train_root}")
            continue

        print("\n" + "=" * 80)
        print(f"Training on noise type: {noise}")
        print("=" * 80)

        train_one_noise(
            train_root=train_root,
            save_path=save_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            val_ratio=args.val_ratio,
            seed=args.seed,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
