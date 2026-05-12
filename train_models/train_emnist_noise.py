# train_emnist_noisy_all.py
import os
import sys
import time
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch import Tensor


# ============================================================
# 1) 模型定义：LeNet5 + 1-channel ResNet18
# ============================================================
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.sigmoid = nn.Sigmoid()
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.c5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)

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


class ResNet18_1ch(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock18,  64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock18, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock18, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock18, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
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


def get_model(model_name: str, num_classes: int) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "lenet5":
        return LeNet5(num_classes=num_classes)
    if model_name == "resnet18":
        return ResNet18_1ch(num_classes=num_classes)
    raise ValueError(f"Unknown model: {model_name}")


# ============================================================
# 2) 训练一个 noisy 模型（单次）
# ============================================================
def train_one(
    train_root: str,
    save_path: str,
    model_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    val_split: float,
    num_workers: int,
    seed: int,
):
    # 可复现
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[TrainOne] device={device}, model={model_name}, train_root={train_root}")
    if not os.path.isdir(train_root):
        raise FileNotFoundError(f"train_root not found: {train_root}")

    # EMNIST / KMNIST 类似：灰度 28x28
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    full_dataset = ImageFolder(root=train_root, transform=transform)
    num_classes = len(full_dataset.classes)
    dataset_size = len(full_dataset)
    if dataset_size == 0:
        raise RuntimeError(f"Empty dataset at: {train_root}")

    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    split = int(np.floor(val_split * dataset_size))
    val_indices = indices[:split]
    train_indices = indices[split:]

    train_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_indices),
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(val_indices),
        num_workers=num_workers,
        pin_memory=True,
    )

    model = get_model(model_name, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = -1.0
    best_state = None

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        # train
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for i, (x, y) in enumerate(train_loader):
            # ✅ 只打印第一个 batch 的信息（不 break）
            if i == 0:
                print("batch type:", type([x, y]), "len:", 2, flush=True)
                print("x:", type(x), getattr(x, "shape", None), flush=True)
                print("y:", type(y), getattr(y, "shape", None), flush=True)

            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * x.size(0)
            pred = out.argmax(dim=1)
            train_total += y.size(0)
            train_correct += (pred == y).sum().item()

            # ✅ batch 级别进度打印
            if (i + 1) % 200 == 0 or (i + 1) == len(train_loader):
                train_loss_now = train_loss_sum / max(train_total, 1)
                train_acc_now = train_correct / max(train_total, 1)
                print(
                    f"Epoch {epoch:03d}/{epochs} | "
                    f"Step {i+1:05d}/{len(train_loader)} | "
                    f"loss {loss.item():.4f} | "
                    f"avg_loss {train_loss_now:.4f} acc {train_acc_now:.4f}",
                    flush=True,
                )
        train_loss = train_loss_sum / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # val
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss_sum += loss.item() * x.size(0)
                pred = out.argmax(dim=1)
                val_total += y.size(0)
                val_correct += (pred == y).sum().item()

        val_loss = val_loss_sum / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f} Acc {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # 保存 best
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if best_state is None:
        best_state = model.state_dict()
    torch.save(best_state, save_path)

    dt_min = (time.time() - t0) / 60.0
    print(f"[TrainOne] ✅ saved best model: {save_path} | best_val_acc={best_val_acc:.4f} | time={dt_min:.2f}min")


# ============================================================
# 3) 主函数：循环 4 种噪声 × EMNIST × (LeNet5/ResNet18)
# ============================================================
def ensure_out_dirs(out_root: str, model_name: str):
    # 你 CIFAR10 的目录结构：feature / mutmodel / results / train
    for sub in ["feature", "mutmodel", "results", "train"]:
        os.makedirs(os.path.join(out_root, sub, model_name), exist_ok=True)


def main():
    parser = argparse.ArgumentParser("Train EMNIST noisy models for 4 noise types (LeNet5/ResNet18)")

    # dataset 根目录（你当前环境用这个）
    parser.add_argument("--dataset_root", type=str, default="/NVNUYucw/ywt/dataset")

    # 输出根目录（你的实验工程目录）
    parser.add_argument("--out_root", type=str, default="./dataset")

    # 固定：EMNIST
    parser.add_argument("--dataset_name", type=str, default="EMNIST")

    # 4种噪声（你也可以改成只跑其中几种）
    parser.add_argument(
        "--noise_types",
        nargs="+",
        default=["RandomLabelNoise", "RandomDataNoise", "SpecificDataNoise", "SpecificLabelNoise"],
    )

    # 两个模型
    parser.add_argument(
        "--models",
        nargs="+",
        default=["LeNet5", "ResNet18"],
    )

    # 训练超参
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    ds = args.dataset_name

    for noise in args.noise_types:
        # 训练集路径来自 dataset： /dataset/<NoiseType>/EMNIST/train
        train_root = os.path.join(args.dataset_root, noise, ds, "train")
        if not os.path.isdir(train_root):
            print(f"[WARN] skip: train_root not found: {train_root}")
            continue

        # 你的输出路径： ./dataset/<NoiseType>/EMNIST
        out_root = os.path.join(args.out_root, noise, ds)
        os.makedirs(out_root, exist_ok=True)

        for model_name in args.models:
            # 创建和 CIFAR10 一样的结构（给后续 baselines / feature 用）
            ensure_out_dirs(out_root, model_name)

            # 模型文件保存到： ./dataset/<NoiseType>/EMNIST/<Model>.pth
            save_path = os.path.join(out_root, f"{model_name}.pth")

            # 开始训练
            train_one(
                train_root=train_root,
                save_path=save_path,
                model_name=model_name,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                val_split=args.val_split,
                num_workers=args.num_workers,
                seed=args.seed,
            )


if __name__ == "__main__":
    main()
