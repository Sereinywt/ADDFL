# train_lenet5_kmnist.py
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
import time


# =============================
# 固定随机种子
# =============================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================
# 完全匹配你项目里的 LeNet5
# =============================
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


# =============================
# 主函数
# =============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", type=str, required=True,
                        help="KMNIST 训练集路径，如 /NVNUYucw/ywt/dataset/OriginalTrainData/KMNIST/train")
    parser.add_argument("--save_path", type=str, required=True,
                        help="保存模型的路径，如 /NVNUYucw/ywt/dataset/OriginalTrainData/KMNIST/LeNet5.pth")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # =============================
    # 数据增强与归一化（KMNIST）
    # =============================
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

    # 读取 KMNIST（ImageFolder 结构）
    full_dataset = datasets.ImageFolder(root=args.train_root, transform=transform_train)
    num_classes = len(full_dataset.classes)
    print(f"Found {len(full_dataset)} images, {num_classes} classes: {full_dataset.classes}")

    # 训练 / 验证划分
    val_len = int(len(full_dataset) * args.val_ratio)
    train_len = len(full_dataset) - val_len
    train_set, val_set = random_split(
        full_dataset, [train_len, val_len],
        generator=torch.Generator().manual_seed(args.seed)
    )
    val_set.dataset.transform = transform_eval

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # =============================
    # 模型、损失函数、优化器
    # =============================
    model = LeNet5(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # =============================
    # 训练循环
    # =============================
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
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

        # =============================
        # 验证
        # =============================
        model.eval()
        val_correct = 0
        val_loss_sum = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                val_loss_sum += loss.item() * imgs.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()

        val_acc = val_correct / len(val_set)
        val_loss = val_loss_sum / len(val_set)

        print(f"Epoch {epoch}/{args.epochs} | "
              f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.4f}")

        # 保存最优模型
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"  ✅ Saved best model to {args.save_path} (val_acc={best_val_acc:.4f})")

    print(f"Training done. Best val_acc = {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
