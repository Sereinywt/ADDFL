# train_vgg16.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import os

# --- VGG-16 for CIFAR-10 模型定义 ---
cfg = {
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        # CIFAR-10 VGG的分类层可以简化
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        # 添加一个平均池化层来确保输出尺寸为 1x1
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def VGG16():
    return VGG('VGG16')

# --- 主脚本 ---
if __name__ == '__main__':
    # --- 参数配置 ---
    TARGET_SAVE_PATH = './dataset/OriginalTrainData/CIFAR10/VGG.pth'
    NUM_EPOCHS = 160
    BATCH_SIZE = 128
    LEARNING_RATE = 0.1

    # --- 数据准备 ---
    print("[VGG-16] 正在准备数据集...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    train_dir = './dataset/OriginalTrainData/CIFAR10/train'
    trainset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # --- 模型和训练设置 ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[VGG-16] 使用设备: {device}")
    model = VGG16().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)

    # --- 训练循环 ---
    print("--- 开始训练 VGG-16 ---")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} LR={scheduler.get_last_lr()[0]:.3f}")
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar.set_postfix(loss=f'{running_loss/(batch_idx+1):.3f}', acc=f'{100.*correct/total:.2f}%')
        scheduler.step()

    print("[VGG-16] 训练完成！")
    os.makedirs(os.path.dirname(TARGET_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), TARGET_SAVE_PATH)
    print(f"模型已成功保存到: {TARGET_SAVE_PATH}")