#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, random, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# ---------------------------
# 工具函数
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_resnet18_1ch(num_classes):
    """适配 1 通道 28x28 的 ResNet-18"""
    model = models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

# ---------------------------
# 训练与验证
# ---------------------------
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(imgs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, total_correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, total_correct / total

# ---------------------------
# 主函数
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Train ResNet18 on EMNIST ImageFolder")
    parser.add_argument("--train_root", type=str, required=True,
                        help="EMNIST train 文件夹路径，如 /NVNUYucw/ywt/dataset/OriginalTrainData/EMNIST/train")
    parser.add_argument("--save_path", type=str, required=True,
                        help="模型保存路径，如 /NVNUYucw/ywt/dataset/OriginalTrainData/EMNIST/ResNet18.pth")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---------------- 数据增强 ----------------
    tf_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomCrop(28, padding=2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    tf_eval = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_dataset = datasets.ImageFolder(root=args.train_root, transform=tf_train)
    num_classes = len(full_dataset.classes)
    print(f"共找到 {len(full_dataset)} 张图像, 类别数: {num_classes}")
    print("类别列表:", full_dataset.classes)

    val_len = int(len(full_dataset) * args.val_ratio)
    train_len = len(full_dataset) - val_len
    train_set, val_set = random_split(full_dataset, [train_len, val_len],
                                      generator=torch.Generator().manual_seed(args.seed))
    val_set.dataset.transform = tf_eval

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, persistent_workers=True)

    # ---------------- 模型定义 ----------------
    model = build_resnet18_1ch(num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # ---------------- 训练循环 ----------------
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.4f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"  ✅ Saved best model to {args.save_path} (val_acc={best_val_acc:.4f})")

    print(f"训练完成，最佳验证准确率 = {best_val_acc:.4f}")

if __name__ == "__main__":
    main()
