# -*- coding: utf-8 -*-
import os
import argparse
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch import Tensor


# =============================
# 固定随机种子
# =============================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================
# ResNet18 for small images (KMNIST 1x28x28)
# =============================
class BasicBlock18(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: Tensor):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet18(nn.Module):
    def __init__(self, num_classes=10, in_channels=1):
        super().__init__()
        self.inplanes = 64

        # 适配 KMNIST：灰度输入 1 通道；小图用 3x3 stride=1；不使用 maxpool
        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.Identity()

        self.layer1 = self._make_layer(BasicBlock18,  64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock18, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock18, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock18, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc      = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# =============================
# 训练一个噪声类型
# =============================
def train_one(train_root, save_path, epochs, batch_size, lr, val_ratio, seed, num_workers):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Train root:", train_root)
    print("Save path :", save_path)

    # KMNIST 常见增强
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

    full_dataset = ImageFolder(root=train_root, transform=transform_train)
    num_classes = len(full_dataset.classes)
    print(f"Found {len(full_dataset)} images, {num_classes} classes: {full_dataset.classes}")

    val_len = int(len(full_dataset) * val_ratio)
    train_len = len(full_dataset) - val_len
    train_set, val_set = random_split(
        full_dataset, [train_len, val_len],
        generator=torch.Generator().manual_seed(seed)
    )
    val_set.dataset.transform = transform_eval

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    model = ResNet18(num_classes=num_classes, in_channels=1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = -1.0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for i, (imgs, labels) in enumerate(train_loader, 1):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            pred = logits.argmax(1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

            if i % 100 == 0:
                print(f"  Epoch {epoch}/{epochs} Step {i}/{len(train_loader)} Loss {loss.item():.4f}")

        train_acc = correct / total
        train_loss = total_loss / total

        model.eval()
        v_correct, v_total, v_loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                v_loss_sum += loss.item() * imgs.size(0)
                pred = logits.argmax(1)
                v_total += labels.size(0)
                v_correct += (pred == labels).sum().item()

        val_acc = v_correct / v_total
        val_loss = v_loss_sum / v_total

        print(f"Epoch {epoch}/{epochs} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.4f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  ✅ Saved best to {save_path} (val_acc={best_val_acc:.4f})")

    print(f"Done. Best val_acc={best_val_acc:.4f}. Time={(time.time()-start)/60:.2f} min")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                        help="例如 /NVNUYucw/ywt/dataset")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--noise_type", type=str, default="ALL",
                        help="RandomLabelNoise / RandomDataNoise / SpecificDataNoise / SpecificLabelNoise / ALL")
    args = parser.parse_args()

    noise_list = ["RandomLabelNoise", "RandomDataNoise", "SpecificDataNoise", "SpecificLabelNoise"]
    if args.noise_type != "ALL":
        noise_list = [args.noise_type]

    for noise in noise_list:
        train_root = os.path.join(args.data_root, noise, "KMNIST", "train")
        save_path  = os.path.join(args.data_root, noise, "KMNIST", "ResNet18.pth")

        if not os.path.isdir(train_root):
            print(f"❌ Skip {noise}: not found -> {train_root}")
            continue

        print("\n" + "=" * 80)
        print(f"Training ResNet18 on {noise}")
        print("=" * 80)

        train_one(
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
