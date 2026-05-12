import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch import Tensor

import os
import sys
import time
import argparse
import numpy as np

# =================================================================
# --- 1. LeNet-5 模型定义 (完全按照您的提供) ---
# =================================================================
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 假设输入是 28x28
        # c1(k=5, p=2) -> (28-5+2*2)/1 + 1 = 28. 尺寸保持不变
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        # 激活函数
        self.sigmoid = nn.Sigmoid() 
        # s2(k=2, s=2) -> 28/2 = 14
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # c3(k=5, p=0) -> (14-5)/1 + 1 = 10
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # s4(k=2, s=2) -> 10/2 = 5
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        # c5(k=5, p=0) -> (5-5)/1 + 1 = 1
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        
        self.flatten = nn.Flatten()
        self.f6 = nn.Linear(120, 84)
        self.f7 = nn.Linear(84, num_classes)

    def forward(self, x, return_features=False):
        # x 初始: [N, 1, 28, 28]
        x = self.sigmoid(self.c1(x)) # [N, 6, 28, 28]
        x = self.s2(x)              # [N, 6, 14, 14]
        x = self.sigmoid(self.c3(x)) # [N, 16, 10, 10]
        x = self.s4(x)              # [N, 16, 5, 5]
        x = self.c5(x)              # [N, 120, 1, 1]
        x = self.flatten(x)         # [N, 120]
        features = self.f6(x)       # [N, 84]
        output = self.f7(features)  # [N, num_classes]

        if return_features:
            return output, features
        return output

# =================================================================
# --- 2. ResNet-18 模型定义 (来自您上一个问题, 适配1通道) ---
# =================================================================

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
    def __init__(self, num_classes=10):
        super().__init__()
        self.inplanes = 64

        # 适配 1-channel (KMNIST / EMNIST)
        # 输入 [N, 1, 28, 28]
        # conv1(k=7, s=2, p=3) -> (28-7+2*3)/2 + 1 = 14.5 -> 14
        self.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        # maxpool(k=3, s=2, p=1) -> (14-3+2*1)/2 + 1 = 7
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1) # [N, 64, 7, 7]

        self.layer1 = self._make_layer(BasicBlock18,  64, 2, stride=1) # [N, 64, 7, 7]
        self.layer2 = self._make_layer(BasicBlock18, 128, 2, stride=2) # [N, 128, 4, 4]
        self.layer3 = self._make_layer(BasicBlock18, 256, 2, stride=2) # [N, 256, 2, 2]
        self.layer4 = self._make_layer(BasicBlock18, 512, 2, stride=2) # [N, 512, 1, 1]

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

# =================================================================
# --- 3. 辅助函数: 获取模型 ---
# =================================================================
def get_model(model_name, num_classes):
    if model_name == 'lenet5':
        print(f"使用模型: LeNet5 (您自定义的版本, {num_classes} 类)")
        return LeNet5(num_classes=num_classes)
    elif model_name == 'resnet18':
        print(f"使用模型: ResNet18 (适配 1-channel, {num_classes} 类)")
        return ResNet18(num_classes=num_classes)
    else:
        raise ValueError(f"未知的模型: {model_name}")

# =================================================================
# --- 4. 主函数 (训练逻辑) ---
# =================================================================
def main():
    # --- Argparse 设置 (符合您的截图和命令) ---
    parser = argparse.ArgumentParser(description="Train EMNIST Letters with ResNet18 / LeNet5")
    parser.add_argument(
        "--train_root",
        type=str,
        required=True,
        help="EMNIST Letters 训练集路径, 如 /.../EMNIST/train",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="保存模型的路径, 如 /.../EMNIST/LeNet5.pth",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["resnet18", "lenet5"],
        help="选择模型: resnet18 或 lenet5",
    )
    parser.add_argument("--epochs", type=int, default=40, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=128, help="批量大小")
    
    # --- 关键：修正学习率 ---
    # 您之前 0.01 的学习率导致了训练失败 (准确率卡在 3.8%)
    # 0.001 是一个对 Adam 更安全、更标准的起始值。
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.001, 
        help="学习率 (默认: 0.001)"
    )
    
    parser.add_argument("--val_split", type=float, default=0.1, help="从训练集中拆分用于验证的比例")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader 使用的进程数")
    
    args = parser.parse_args()

    print("--- 训练配置 ---")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("--------------------")
    
    # --- 设备设置 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # --- 数据预处理 ---
    # EMNIST 是 28x28 的灰度图
    # 1. Grayscale(1): 确保 ImageFolder 加载为 1 通道
    # 2. ToTensor(): 转换为 [0, 1] 的张量
    # 3. Normalize: 归一化到 [-1, 1]
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # --- 加载数据集和拆分 ---
    try:
        full_dataset = ImageFolder(root=args.train_root, transform=transform)
    except FileNotFoundError:
        print(f"错误: 找不到训练路径: {args.train_root}")
        sys.exit(1)
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        sys.exit(1)

    num_classes = len(full_dataset.classes)
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(args.val_split * dataset_size))
    
    # 随机打乱索引
    np.random.seed(42) # 保证可复现
    np.random.shuffle(indices)
    
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        full_dataset, 
        batch_size=args.batch_size, 
        sampler=train_sampler, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        full_dataset, 
        batch_size=args.batch_size, 
        sampler=val_sampler, 
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"总图片数: {dataset_size}, 类别数: {num_classes}")
    print(f"训练集大小: {len(train_indices)}, 验证集大小: {len(val_indices)}")
    print(f"发现 {num_classes} 个类别: {full_dataset.classes}")

    # --- 初始化模型, 损失函数, 优化器 ---
    model = get_model(args.model, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --- 训练循环 ---
    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(args.epochs):
        # --- 训练 ---
        model.train()
        running_loss_train = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss_train += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss_train = running_loss_train / total_train
        epoch_acc_train = correct_train / total_train

        # --- 验证 ---
        model.eval()
        running_loss_val = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss_val += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        epoch_loss_val = running_loss_val / total_val
        epoch_acc_val = correct_val / total_val

        # 打印日志
        print(f'Epoch {epoch+1:03d}/{args.epochs} | '
              f'Train Loss {epoch_loss_train:.4f} Acc {epoch_acc_train:.4f} | '
              f'Val Loss {epoch_loss_val:.4f} Acc {epoch_acc_val:.4f}')

        # --- 保存最佳模型 ---
        if epoch_acc_val > best_val_acc:
            best_val_acc = epoch_acc_val
            try:
                # 确保保存目录存在
                save_dir = os.path.dirname(args.save_path)
                if not os.path.exists(save_dir) and save_dir != "":
                    os.makedirs(save_dir, exist_ok=True)
                
                torch.save(model.state_dict(), args.save_path)
                print(f'  ✅ Saved best model to {args.save_path} (val_acc={best_val_acc:.4f})')
            except Exception as e:
                print(f"保存模型时出错: {e}")

    end_time = time.time()
    print(f"--- 训练完成 ---")
    print(f"总耗时: {(end_time - start_time) / 60:.2f} 分钟")
    print(f"最佳验证准确率: {best_val_acc:.4f}")
    print(f"最佳模型已保存至: {args.save_path}")

if __name__ == "__main__":
    main()