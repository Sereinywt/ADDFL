# train_resnet20_aligned.py
import os
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm


# ------------------------------
# ResNet-20 for CIFAR10 (键名与 adversirial.py 对齐)
# conv1 -> nn.Sequential(conv,bn)  => conv1.0 / conv1.1
# block 主分支 -> left.*           => left.0/left.1/left.2(ReLU)/left.3/left.4
# 投影捷径 -> shortcut.*           => shortcut.0/shortcut.1
# 最终全连接 -> fc                 => fc.weight / fc.bias
# ------------------------------
class BasicBlockLeft(nn.Sequential):
    """left.0 conv -> left.1 bn -> left.2 ReLU -> left.3 conv -> left.4 bn"""
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.add_module('0', nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False))
        self.add_module('1', nn.BatchNorm2d(planes))
        self.add_module('2', nn.ReLU(inplace=True))
        self.add_module('3', nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False))
        self.add_module('4', nn.BatchNorm2d(planes))


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.left = BasicBlockLeft(in_planes, planes, stride)
        # 投影捷径 (1x1 conv + BN)：当尺寸/通道变化时使用
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class ResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 注意：conv1 必须是 Sequential，形成 conv1.0 / conv1.1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
        )
        self.in_planes = 16
        self.layer1 = self._make_layer(16, num_blocks=3, stride=1)
        self.layer2 = self._make_layer(32, num_blocks=3, stride=2)
        self.layer3 = self._make_layer(64, num_blocks=3, stride=2)
        self.fc = nn.Linear(64, num_classes)

        # Kaiming 初始化（可选）
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5)) if 'math' in globals() else None

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def get_dataloader(data_root, batch_size, num_workers):
    # 与常见 CIFAR10 统计一致；若你的主流程使用不同 Normalize，请保持一致
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    train_dir = os.path.join(data_root, "OriginalTrainData", "CIFAR10", "train")
    dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return loader, len(dataset.classes)


def save_model(model, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 保存为纯 state_dict（无 module. 前缀）
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, save_path)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 模型已保存到: {save_path}")

    # 自检：严格加载应无 missing/unexpected
    m2 = ResNet20()
    missing, unexpected = m2.load_state_dict(torch.load(save_path, map_location="cpu"), strict=False)
    print(f"自检 -> missing: {missing}, unexpected: {unexpected}")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[INFO] 设备: {device}")

    train_loader, num_classes = get_dataloader(args.data_root, args.batch_size, args.num_workers)
    print(f"[INFO] 类别数: {num_classes}（ImageFolder 目录名顺序决定标签）")

    model = ResNet20(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4
    )
    # MultiStep 学习率衰减：80/120 epoch 衰减 0.1
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp))

    print("[INFO] 开始训练 ResNet-20 (CIFAR10 对齐版)")
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, total=len(train_loader), ncols=100,
                    desc=f"Epoch {epoch}/{args.epochs} | LR={scheduler.get_last_lr()[0]:.3f}")

        for images, labels in pbar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and args.amp)):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            global_step += 1

            pbar.set_postfix(loss=f"{running_loss/total:.3f}", acc=f"{100.0*correct/total:.2f}%")

        scheduler.step()

        if args.save_every > 0 and epoch % args.save_every == 0:
            epoch_path = os.path.join(args.save_dir, f"ResNet_epoch{epoch}.pth")
            save_model(model, epoch_path)

    # 训练结束保存“干净模型”到默认主程序可发现的位置
    final_path = os.path.join(args.save_dir, "ResNet.pth")
    save_model(model, final_path)


def parse_args():
    parser = argparse.ArgumentParser("Train ResNet-20 for CIFAR10 with adversirial.py key alignment")
    parser.add_argument("--data_root", type=str, default="./dataset", help="根数据目录，脚本将使用 <root>/OriginalTrainData/CIFAR10/train")
    parser.add_argument("--save_dir", type=str, default="./dataset/OriginalTrainData/CIFAR10", help="保存目录")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", help="开启混合精度")
    parser.add_argument("--cpu", action="store_true", help="仅用 CPU 训练")
    parser.add_argument("--save_every", type=int, default=0, help="每 N 个 epoch 额外保存一次，0 表示只保存最终模型")
    return parser.parse_args()


if __name__ == "__main__":
    # 可重复性（可选）
    import random, numpy as np, math
    torch.backends.cudnn.benchmark = True
    seed = 2025
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    args = parse_args()
    train(args)
