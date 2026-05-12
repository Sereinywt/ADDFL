import torch
from torch import nn
from torchvision.transforms import transforms
from utils.dataset import dataset
from utils.models import *
from wavemix.classification import WaveMix

import json
import random
import os
from tqdm import tqdm

# ======== 配置区域 ========
dataset_name = 'EMNIST'
model_name = 'LeNet1'
train_root = '/root/ywt/dataset/CaseStudyData/EMNIST/train_clean'
test_root  = '/root/ywt/dataset/OriginalTestData/EMNIST/test'
class_path = './dataset/emnist_classes.json'
image_size = (28, 28, 1)
lr = 0.001
epoches = 50
batch_size = 64    #
# =========================

# ✅ 加载模型
if model_name == 'WaveMix':
    model = WaveMix(
        num_classes=26,
        depth=4,
        mult=2,
        ff_channel=48,
        final_dim=48,
        dropout=0.3,
        level=3,
        patch_size=4,
    )
else:
    model = eval(model_name)()

# ✅ 图像预处理
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

# ✅ 构造 data_s 全量分配函数
def build_data_s(root_path):
    with open(class_path, 'r') as f:
        classes = json.load(f)
    num_classes = len(classes)
    model = LeNet1(num_classes=num_classes)
    result = {}
    IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp']  # 允许的图片扩展名

    for cls in classes.keys():
        class_dir = os.path.join(root_path, cls)
        if not os.path.exists(class_dir):
            print(f"⚠️ 类别路径不存在：{class_dir}，跳过该类")
            continue
        img_list = []
        for img in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img)
            # 跳过文件夹和隐藏文件、非图片文件
            if (
                os.path.isdir(img_path)
                or img.startswith('.')
                or not any(img.lower().endswith(ext) for ext in IMG_EXTENSIONS)
            ):
                continue
            img_list.append(os.path.join(cls, img))
        result[cls] = img_list
    return result


# ✅ 加载训练数据
print(f'📂 训练集路径: {train_root}')
train_data_s = build_data_s(train_root)
train_data = dataset(
    root=train_root,
    classes_path=class_path,
    transform=transform,
    image_size=image_size,
    image_set='',
    data_s=train_data_s
)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16)

# ✅ 加载测试数据
print(f'📂 测试集路径: {test_root}')
test_data_s = build_data_s(test_root)
test_data = dataset(
    root=test_root,
    classes_path=class_path,
    transform=transform,
    image_size=image_size,
    image_set='',
    data_s=test_data_s
)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

# ✅ 设置训练相关组件
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'🚀 使用设备: {device}')
model.to(device)

# ✅ 训练模型
for epoch in range(epoches):
    model.train()
    running_loss = 0.0
    for i, (images, labels, *_) in enumerate(train_loader):
        outputs = model(images.to(device))
        loss = criterion(outputs, labels.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if (i + 1) % 100 == 0:
            print(f'🧱 Epoch [{epoch + 1}/{epoches}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    avg_loss = running_loss / len(train_loader)
    print(f'📉 Epoch {epoch + 1} 平均训练损失: {avg_loss:.4f}')

    # ✅ 测试
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels, *_ in test_loader:
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

    acc = 100 * correct / total
    print(f'🎯 Epoch {epoch + 1} 测试集准确率：{acc:.2f}%')

# ✅ 保存模型参数（推荐命名为model_clean.pth）
save_path = f'./dataset/CaseStudyData/{dataset_name}/model_clean.pth'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model.state_dict(), save_path)
print(f'✅ 干净模型已保存至：{save_path}')
