# models.py (重构优化 & 兼容加载版)

import os
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from collections import OrderedDict
from einops.layers.torch import Rearrange
from wavemix import Level4Waveblock, Level3Waveblock, Level2Waveblock, Level1Waveblock

# ---------------------------------------------------------
# 小工具：更宽松地加载 state_dict（线性层命名 remap、strict=False）
# ---------------------------------------------------------
def load_state_dict_flexible(model: nn.Module, state_or_path, remap_linear_to_fc: bool = True, strict: bool = False):
    """
    用于把外部权重加载进当前模型：
    - 支持传入路径或已加载的 state(dict) / checkpoint(dict)
    - 可选地把 'linear.' 前缀键重命名为 'fc.'（很多 CIFAR ResNet 权重习惯用 linear）
    - 默认 strict=False，避免小改动导致报错
    """
    if isinstance(state_or_path, (str, os.PathLike)):
        if not os.path.exists(state_or_path):
            print(f"[WARN] checkpoint 不存在：{state_or_path}，跳过加载。")
            return
        state = torch.load(state_or_path, map_location="cpu")
    else:
        state = state_or_path

    # 兼容各种保存格式
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    if remap_linear_to_fc:
        state = OrderedDict((k.replace("linear.", "fc."), v) for k, v in state.items())

    missing, unexpected = model.load_state_dict(state, strict=strict)
    print("[load_state_dict_flexible] missing keys:", missing)
    print("[load_state_dict_flexible] unexpected keys:", unexpected)


# =========================
# --- LeNet 家族 ---
# =========================

class LeNet1(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 4, 5)
        self.tanh = nn.Tanh()
        self.s2 = nn.AvgPool2d(2)
        self.c3 = nn.Conv2d(4, 12, 5)
        self.s4 = nn.AvgPool2d(2)
        # 输入维度 12 * 4 * 4 是针对 28x28 输入计算的
        self.fc = nn.Linear(12 * 4 * 4, num_classes)

    def forward(self, x, return_features=False):
        x = self.c1(x)
        x = self.tanh(x)
        x = self.s2(x)
        x = self.c3(x)
        x = self.tanh(x)
        x = self.s4(x)
        features = x.view(x.size(0), -1)
        output = self.fc(features)

        if return_features:
            return output, features
        return output

class LeNet1_NCNV(nn.Module):
    def __init__(self):
        super(LeNet1_NCNV, self).__init__()
        self.c1 = nn.Conv2d(1, 4, 5)
        self.TANH = nn.Tanh()
        self.s2 = nn.AvgPool2d(2)
        self.c3 = nn.Conv2d(4, 12, 5)
        self.s4 = nn.AvgPool2d(2)
        self.fc = nn.Linear(12 * 4 * 4, 10)

    def forward(self, x, feat=False):
        x = self.c1(x)
        x = self.TANH(x)
        x = self.s2(x)
        x = self.c3(x)
        x = self.TANH(x)
        x = self.s4(x)
        fea = x.view(x.size(0), -1)
        x = self.fc(fea)
        if feat:
            return x, fea
        else:
            return x
    
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 假设输入是 28x28，通过 padding 变成 32x32
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

class LeNet5_NCNV(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5_NCNV, self).__init__()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.Sigmoid = nn.Sigmoid()
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.flatten = nn.Flatten()
        self.f6 = nn.Linear(120, 84)
        # ★ 这里用传进来的 num_classes，而不是写死 10
        self.f7 = nn.Linear(84, num_classes)

    def forward(self, x, feat=False):
        # x输入为32*32*1， 输出为28*28*6
        x = self.Sigmoid(self.c1(x))
        # x输入为28*28*6， 输出为14*14*6
        x = self.s2(x)
        # x输入为14*14*6， 输出为10*10*16
        x = self.Sigmoid(self.c3(x))
        # x输入为10*10*16， 输出为5*5*16
        x = self.s4(x)
        # x输入为5*5*16， 输出为1*1*120
        x = self.c5(x)
        x = self.flatten(x)
        fea = self.f6(x)
        x = self.f7(fea)
        if feat:
            return x, fea
        else:
            return x

# =========================
# --- ResNet 家族 (CIFAR 风格) ---
# =========================

class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 这一段你已经改成 conv1/bn1/conv2/bn2 的就保持这样
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
class ResidualBlock_Left(nn.Module):
    """
    对齐你 ckpt 的命名：
      left.0 conv
      left.1 bn
      left.2 relu (无参数)
      left.3 conv
      left.4 bn
      shortcut.0 conv1x1 (可选)
      shortcut.1 bn (可选)
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False), # left.0
            nn.BatchNorm2d(out_channels),                                                   # left.1
            nn.ReLU(inplace=True),                                                          # left.2
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),      # left.3
            nn.BatchNorm2d(out_channels),                                                   # left.4
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),         # shortcut.0
                nn.BatchNorm2d(out_channels),                                               # shortcut.1
            )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)

class ResNet18_CIFAR_CustomKey(nn.Module):
    """
    对齐你现在 CIFAR10 ckpt 的 key 命名和 fc_in=64：
      conv1.0/conv1.1
      layer1/2/3  每层 3 个 block（你 ckpt 里确实有 layer1.2 / layer2.2 / layer3.2）
      fc: Linear(64 -> num_classes)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False),  # conv1.0
            nn.BatchNorm2d(16),                                    # conv1.1
            nn.ReLU(inplace=True),                                 # conv1.2 (无参数)
        )
        self.layer1 = self._make_layer(ResidualBlock_Left, 16, 3, stride=1)
        self.layer2 = self._make_layer(ResidualBlock_Left, 32, 3, stride=2)
        self.layer3 = self._make_layer(ResidualBlock_Left, 64, 3, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, channels, s))
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        features = out.view(out.size(0), -1)
        logits = self.fc(features)
        if return_features:
            return logits, features
        return logits
    
class ResNet_(nn.Module):
    """
    CIFAR 版三阶段 ResNet（conv->layer1..3->avgpool->fc）
    定义为：conv1 = Sequential(conv+bn+relu)，匹配 ckpt 里的 conv1.0 / conv1.1。
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 16

        # ✅ 关键修改：用 Sequential，而不是 conv1 + bn1 + relu 三个属性
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(ResidualBlock, 16, 3, stride=1)
        self.layer2 = self._make_layer(ResidualBlock, 32, 3, stride=2)
        self.layer3 = self._make_layer(ResidualBlock, 64, 3, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, channels, s))
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        # ✅ 这里直接用 conv1(x)，不要再调用 bn1 / relu
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        features = out.view(out.size(0), -1)
        output = self.fc(features)

        if return_features:
            return output, features
        return output

class ResNet_NCNV_(nn.Module):
    def __init__(self, ResidualBlockCls, num_classes=10, in_channels=3):
        super().__init__()
        self.inchannel = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlockCls, 16, 3, stride=1)
        self.layer2 = self.make_layer(ResidualBlockCls, 32, 3, stride=2)
        self.layer3 = self.make_layer(ResidualBlockCls, 64, 3, stride=2)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.inchannel, channels, s))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x, feat=False):
        if x.size(1) == 1 and self.conv1[0].in_channels == 3:
            x = x.repeat(1, 3, 1, 1)

        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        fea = F.adaptive_avg_pool2d(out, (1, 1))
        fea = fea.view(fea.size(0), -1)
        out = self.fc(fea)
        if feat:
            return out, fea     # NCNV 需要 (logits, features)
        else:
            return out

def ResNet18_NCNV(num_classes=10, in_channels=3):
    """
    供 baselines.NCNV 中 eval('ResNet18_NCNV') 调用的工厂函数。
    本质上复用上面的 ResNet_NCNV_ 结构。
    """
    return ResNet_NCNV_(ResidualBlockCls=ResidualBlock,
                        num_classes=num_classes,
                        in_channels=in_channels)


# --- 1. ResNet 基础块 (两个 ResNet 都会用到) ---
class ResNet18_BasicBlock(nn.Module):
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

# --- 2. ResNet18_EMNIST (1-channel, 适用于 KMNIST/EMNIST) ---
class ResNet18_EMNIST(nn.Module):
    def __init__(self, num_classes=10, stem="cifar"):  # stem: "cifar" or "imagenet"
        super().__init__()
        self.inplanes = 64

        if stem == "imagenet":
            # 7x7 + maxpool（匹配 conv1=[64,1,7,7] 的 EMNIST ckpt）
            self.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        else:
            # 3x3 + no maxpool（匹配 conv1=[64,1,3,3] 的 KMNIST ckpt）
            self.conv1 = nn.Conv2d(1, 64, 3, stride=1, padding=1, bias=False)
            self.maxpool = nn.Identity()

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(ResNet18_BasicBlock,  64, 2, stride=1)
        self.layer2 = self._make_layer(ResNet18_BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(ResNet18_BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(ResNet18_BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
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

# --- 3. ResNet18_CIFAR (3-channel, 适用于 CIFAR-10) ---
class ResNet18_CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.inplanes = 64
        # 3-channel, 3x3 kernel (匹配 3-channel CIFAR checkpoint)
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.Identity() # 移除 MaxPool
        self.layer1 = self._make_layer(ResNet18_BasicBlock,  64, 2, stride=1)
        self.layer2 = self._make_layer(ResNet18_BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(ResNet18_BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(ResNet18_BasicBlock, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc      = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes: 
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes, 1, stride=stride, bias=False), nn.BatchNorm2d(planes))
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x); x = torch.flatten(x, 1)
        return self.fc(x)

# =================================================================
# =========================
# --- VGG 家族 ---
# =========================
# 将此代码放入您的 utils/models.py 或其他定义模型的地方，替换掉旧的VGG类
# 这是正确的、带BatchNorm2d的模型定义
import torch.nn as nn

cfg = {
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name='VGG16', num_classes=10):
        super(VGG, self).__init__()
        # 确保特征提取器名为 'features'
        self.features = self._make_layers(cfg[vgg_name])
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
                # 确保这里有 BatchNorm2d
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

class VGG_NCNV(nn.Module):

    def __init__(self, vgg_name='VGG16'):
        super(VGG_NCNV, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(512, 10),
        )
        #         self.classifier = nn.Linear(512,10)

        self._initialize_weight()

    def forward(self, x, feat=False):
        fea = self.features(x)
        # 在进入
        fea = fea.view(fea.size(0), -1)
        out = self.classifier(fea)
        if feat:
            return out, fea
        else:
            return out

    # make layers
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3  # RGB 初始通道为3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]  # kernel_size 为 2 x 2,然后步长为2
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),  # 都是(3.3)的卷积核
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]  # RelU
                in_channels = x  # 重定义通道
        #         layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    # 初始化参数
    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # xavier is used in VGG's paper
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

# =========================
# --- WaveMix 家族 ---
# =========================
from wavemix import Level4Waveblock, Level3Waveblock, Level2Waveblock, Level1Waveblock
import torch.nn as nn
from einops.layers.torch import Rearrange

class WaveMix(nn.Module):
    def __init__(
        self,
        *,
        num_classes=10,
        depth=4,
        mult=2,
        ff_channel=48,
        final_dim=48,
        dropout=0.3,
        level=3,
        initial_conv='patchify',   # 兼容 'pachify' 的写法
        patch_size=4,
        stride=2,
        in_channels=3             # ★ 支持 1/3 通道
    ):
        super().__init__()
        # 兼容拼写
        if initial_conv == 'pachify':
            initial_conv = 'patchify'

        self.level = level                 # ★ forward 中需要用到
        self.initial_conv = initial_conv
        self.patch_size = patch_size

        wave_block = {
            1: Level1Waveblock, 2: Level2Waveblock,
            3: Level3Waveblock, 4: Level4Waveblock
        }[level]

        self.layers = nn.ModuleList([
            wave_block(mult=mult, ff_channel=ff_channel, final_dim=final_dim, dropout=dropout)
            for _ in range(depth)
        ])

        # 初始特征提取
        if initial_conv == 'strided':
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, int(final_dim / 2), kernel_size=3, stride=stride, padding=1),
                nn.Conv2d(int(final_dim / 2), final_dim, kernel_size=3, stride=stride, padding=1)
            )
        elif initial_conv == 'patchify':
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, int(final_dim / 4), kernel_size=3, stride=1, padding=1),
                nn.Conv2d(int(final_dim / 4), int(final_dim / 2), kernel_size=3, stride=1, padding=1),
                nn.Conv2d(int(final_dim / 2), final_dim, kernel_size=patch_size, stride=patch_size),
                nn.GELU(),
                nn.BatchNorm2d(final_dim)
            )
        else:
            raise ValueError(f"initial_conv must be 'strided' or 'patchify', got: {initial_conv}")

        # 池化 + 分类头
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('... () () -> ...'),
            nn.Linear(final_dim, num_classes)
        )

    def forward(self, img, return_features=False):
        features = self.conv(img)

        # ★ 关键补丁：确保特征图尺寸能被 2^level 整除，避免 wavemix 内部 reshape 报错
        need = 1 << self.level   # 2^level
        h, w = features.shape[-2:]
        pad_h = (-h) % need
        pad_w = (-w) % need
        if pad_h or pad_w:
            # 只在右/下方向补零，保持对齐
            features = F.pad(features, (0, pad_w, 0, pad_h))

        # 残差堆叠
        for attn_block in self.layers:
            features = attn_block(features) + features

        logits = self.pool(features)

        if return_features:
            return logits, features.view(features.size(0), -1)
        return logits

class WaveMix_NCNV(nn.Module):
    def __init__(
            self,
            *,
            num_classes=26,
            depth=4,
            mult=2,
            ff_channel=48,
            final_dim=48,
            dropout=0.3,
            level=3,
            initial_conv='pachify',  # or 'strided'
            patch_size=4,
            stride=2,

    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            if level == 4:
                self.layers.append(
                    Level4Waveblock(mult=mult, ff_channel=ff_channel, final_dim=final_dim, dropout=dropout))
            elif level == 3:
                self.layers.append(
                    Level3Waveblock(mult=mult, ff_channel=ff_channel, final_dim=final_dim, dropout=dropout))
            elif level == 2:
                self.layers.append(
                    Level2Waveblock(mult=mult, ff_channel=ff_channel, final_dim=final_dim, dropout=dropout))
            else:
                self.layers.append(
                    Level1Waveblock(mult=mult, ff_channel=ff_channel, final_dim=final_dim, dropout=dropout))

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('... () () -> ...'),
            nn.Linear(final_dim, num_classes)
        )

        if initial_conv == 'strided':
            self.conv = nn.Sequential(
                nn.Conv2d(3, int(final_dim / 2), 3, stride, 1),
                nn.Conv2d(int(final_dim / 2), final_dim, 3, stride, 1)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(3, int(final_dim / 4), 3, 1, 1),
                nn.Conv2d(int(final_dim / 4), int(final_dim / 2), 3, 1, 1),
                nn.Conv2d(int(final_dim / 2), final_dim, patch_size, patch_size),
                nn.GELU(),
                nn.BatchNorm2d(final_dim)
            )

    def forward(self, img, feat=False):
        fea = self.conv(img)

        for attn in self.layers:
            x = attn(fea) + fea

        out = self.pool(x)

        if feat:
            fea = fea.view(fea.size(0), -1)
            return out, fea
        else:
            return out


# =========================
# --- TCDCNN (重构) ---
# =========================

class TCDCNN(nn.Module):
    """
    重构后的TCDCNN，仅包含模型结构定义。
    loss, accuracy等业务逻辑应移至训练脚本中。
    """
    def __init__(self):
        super().__init__()
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5),
            nn.Hardtanh(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 48, kernel_size=3),
            nn.Hardtanh(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 64, kernel_size=3),
            nn.Hardtanh(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=2),
            nn.Hardtanh()
        )
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout()

        # 分类/回归头：当前为10个坐标点的回归
        self.landmark_regressor = nn.Linear(256, 10)

    def forward(self, x, return_features=False):
        x = self.feature_extractor(x)
        features = self.flatten(x)
        features = self.dropout(features)

        output = self.landmark_regressor(features)

        if return_features:
            return output, features
        return output


# =========================
# 使用提示（可选）
# =========================
# 对于 CIFAR-ResNet 权重加载（例如 /NVNUYucw/ywt/dataset/OriginalTrainData/CIFAR10/ResNet.pth）：
# model = ResNet_(num_classes=10)
# load_state_dict_flexible(model, "/NVNUYucw/ywt/dataset/OriginalTrainData/CIFAR10/ResNet.pth", remap_linear_to_fc=True, strict=False)
# 这样即可避免之前的 Missing/Unexpected/size mismatch 报错。
# ===== 弹性加载工具 =====
import os
from collections import OrderedDict

def _filter_and_load(model, state, strict=False):
    """只加载形状匹配的键；其余忽略，避免 size mismatch 报错。"""
    model_dict = model.state_dict()
    new_state = OrderedDict()
    for k, v in state.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            new_state[k] = v
    missing = [k for k in model_dict.keys() if k not in new_state]
    unexpected = [k for k in state.keys() if k not in model_dict]
    model.load_state_dict(new_state, strict=strict)
    print(f"[flex load] loaded: {len(new_state)} keys | missing: {len(missing)} | unexpected: {len(unexpected)}")

def load_resnet_weights_flex(model, ckpt_path):
    if not os.path.exists(ckpt_path):
        print(f"[flex load] checkpoint not found: {ckpt_path}")
        return
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # CIFAR 常见：最终层叫 linear.*，需要映射到 fc.*
    state = OrderedDict((k.replace("linear.", "fc."), v) for k, v in state.items())
    _filter_and_load(model, state, strict=False)

# ===== 把 CIFAR 版 ResNet_ 暴露为 'ResNet' =====
class ResNet(ResNet_):
    pass
