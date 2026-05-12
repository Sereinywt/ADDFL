#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms

# ======================= 与 adversirial.py 期望键名对齐的实现 =======================
class LeNet1Proj(nn.Module):
    """与流水线键名一致：c1, c3, fc"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.c1   = nn.Conv2d(1, 4, 5, stride=1, padding=2)
        self.pool = nn.AvgPool2d(2, 2)
        self.c3   = nn.Conv2d(4, 12, 5, stride=1, padding=2)
        self.fc   = nn.Linear(12*7*7, num_classes)
        self.act  = nn.Tanh()

    def forward(self, x):
        x = self.pool(self.act(self.c1(x)))   # 28->14
        x = self.pool(self.act(self.c3(x)))   # 14->7
        x = torch.flatten(x, 1)
        x = self.fc(self.act(x))
        return x

class LeNet5Proj(nn.Module):
    """常见 LeNet5 结构；键名：c1, c3, fc1, fc2, fc3"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.c1   = nn.Conv2d(1, 6, 5)              # 28->24
        self.pool = nn.AvgPool2d(2, 2)              # 24->12
        self.c3   = nn.Conv2d(6, 16, 5)             # 12->8
        self.fc1  = nn.Linear(16*4*4, 120)          # 8->4 (池化)
        self.fc2  = nn.Linear(120, 84)
        self.fc3  = nn.Linear(84, num_classes)
        self.act  = nn.Tanh()

    def forward(self, x):
        x = self.pool(self.act(self.c1(x)))
        x = self.pool(self.act(self.c3(x)))
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x

def get_model(name: str):
    name = name.lower()
    if name == "lenet1": return LeNet1Proj()
    if name == "lenet5": return LeNet5Proj()
    raise ValueError("model_name must be LeNet1 or LeNet5")

# ================================ 训练主程序 =================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", required=True, help="FASHION/train 目录（按类分文件夹 0..9）")
    ap.add_argument("--model_name", required=True, choices=["LeNet1","LeNet5"])
    ap.add_argument("--out_path",   required=True, help="保存权重的路径，比如 ./dataset/OriginalTrainData/FASHION/LeNet1.pth")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1216)
    # 可调归一化，默认使用 Fashion-MNIST 统计
    ap.add_argument("--mean", type=float, default=0.2860)
    ap.add_argument("--std",  type=float, default=0.3530)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((args.mean,), (args.std,))
    ])
    full = datasets.ImageFolder(args.train_root, transform=tfm)

    # 从 train 切 10% 作验证（若你有独立 val，可自行替换）
    val_len = max(1, int(0.1 * len(full)))
    train_len = len(full) - val_len
    train_set, val_set = random_split(full, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=256,          shuffle=False, num_workers=2, pin_memory=True)

    model = get_model(args.model_name).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    best_acc, best_state = 0.0, None
    for epoch in range(1, args.epochs+1):
        # train
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward(); opt.step()
            loss_sum += loss.item()*y.size(0)
            correct  += (logits.argmax(1) == y).sum().item()
            total    += y.size(0)
        train_loss = loss_sum/total; train_acc = correct/total

        # val
        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                logits = model(x)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)
        val_acc = correct/total if total>0 else 0.0
        print(f"[{args.model_name}] epoch {epoch:02d}/{args.epochs}  "
              f"train_loss {train_loss:.4f}  train_acc {train_acc:.4f}  val_acc {val_acc:.4f}")

        if val_acc >= best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    torch.save(best_state if best_state is not None else model.state_dict(), args.out_path)
    print(f"✅ saved (project-compatible) -> {args.out_path}   best_val_acc={best_acc:.4f}")

    # 也顺带存一份“vanilla”版（可选）
    vanilla_path = os.path.splitext(args.out_path)[0] + ".vanilla.pth"
    torch.save(model.state_dict(), vanilla_path)
    print(f"📝 saved (vanilla state_dict) -> {vanilla_path}")

if __name__ == "__main__":
    main()
