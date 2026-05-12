import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch import Tensor
import os
import time

# --- 1. 配置参数 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {DEVICE}")

# --- 路径配置 (根据您的提供) ---
TRAIN_DIR = '/NVNUYucw/ywt/dataset/OriginalTrainData/CIFAR10/train'
MODEL_SAVE_PATH = '/NVNUYucw/ywt/dataset/OriginalTrainData/CIFAR10/ResNet18.pth'

# 假设的测试集路径 (强烈建议)
TEST_DIR = '/NVNUYucw/ywt/dataset/OriginalTestData/CIFAR10/test' 

# --- 超参数 ---
NUM_EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_CLASSES = 10

# =================================================================
# --- 2. 模型定义 (基于您的 ResNet18，已适配 CIFAR-10) ---
# =================================================================

# BasicBlock18 (来自您的代码, 无需修改)
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


# ResNet18 (来自您的代码, 关键修改以适配 CIFAR-10)
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.inplanes = 64

        # -----------------------------------------------------------------
        # --- 关键修改 (适配 CIFAR-10) ---
        
        # 1. 输入通道: 1 -> 3 (KMNIST 是 1, CIFAR-10 是 3)
        # 2. 卷积核: 7 -> 3 (适配 32x32 图像)
        # 3. 步长: 2 -> 1
        # 4. 填充: 3 -> 1
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False) # <--- 修改
        
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        
        # 5. 移除 MaxPool (对于 32x32 太激进了)
        self.maxpool = nn.Identity() # <--- 修改 (原: nn.MaxPool2d(3, stride=2, padding=1))
        # -----------------------------------------------------------------

        # 保持 ResNet-18 的 4 个 stage (2, 2, 2, 2)
        self.layer1 = self._make_layer(BasicBlock18,  64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock18, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock18, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock18, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc      = nn.Linear(512, num_classes) # <--- 您的代码是 512, 保持一致

    def _make_layer(self, block, planes, blocks, stride):
        downsample = None
        # 注意: 您的 BasicBlock18 expansion=1, 所以
        # self.inplanes != planes * block.expansion
        # 应该简化为 self.inplanes != planes
        if stride != 1 or self.inplanes != planes: 
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes # 在 BasicBlock 中 expansion=1

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor):
        # 您的前向传播 (已适配)
        x = self.maxpool(self.relu(self.bn1(self.conv1(x)))) # maxpool 已是 Identity
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# =================================================================
# --- 3. 数据预处理和加载 ---
# =================================================================

# CIFAR-10 标准化
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 检查训练路径
if not os.path.exists(TRAIN_DIR):
    print(f"错误: 训练路径不存在: {TRAIN_DIR}")
    exit()

# 加载训练数据集
try:
    train_dataset = ImageFolder(root=TRAIN_DIR, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    print(f"成功从 {TRAIN_DIR} 加载训练数据。")
except Exception as e:
    print(f"加载训练数据时出错: {e}")
    exit()

# 加载测试数据集 (如果存在)
test_loader = None
if os.path.exists(TEST_DIR):
    try:
        test_dataset = ImageFolder(root=TEST_DIR, transform=transform_test)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        print(f"成功从 {TEST_DIR} 加载测试数据。")
    except Exception as e:
        print(f"加载测试数据时出错: {e}")
else:
    print(f"--- 警告: 未找到测试集 {TEST_DIR}。将跳过评估。 ---")


# =================================================================
# --- 4. 初始化模型、损失函数和优化器 ---
# =================================================================

model = ResNet18(num_classes=NUM_CLASSES)
model = model.to(DEVICE)
print("模型结构 (已适配 CIFAR-10):")
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# =================================================================
# --- 5. 训练循环 ---
# =================================================================

print("开始训练...")
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    model.train() # 设置模型为训练模式
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    train_accuracy = 100 * correct_train / total_train
    train_avg_loss = running_loss / len(train_loader)
    
    print(f'--- Epoch {epoch+1} 结束 ---')
    print(f'训练集平均损失: {train_avg_loss:.4f}, 训练集准确率: {train_accuracy:.2f}%')

    # --- 验证 (如果测试集存在) ---
    if test_loader:
        model.eval() 
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
        test_accuracy = 100 * correct_test / total_test
        print(f'测试集准确率: {test_accuracy:.2f}%')
    print(f'-----------------------')

end_time = time.time()
print(f'训练完成! 总耗时: {(end_time - start_time) / 60:.2f} 分钟')

# =================================================================
# --- 6. 保存模型 ---
# =================================================================
try:
    save_dir = os.path.dirname(MODEL_SAVE_PATH)
    if not os.path.exists(save_dir) and save_dir != "":
        os.makedirs(save_dir, exist_ok=True)
    
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f'模型已成功保存到: {MODEL_SAVE_PATH}')
except Exception as e:
    print(f'保存模型时出错: {e}')