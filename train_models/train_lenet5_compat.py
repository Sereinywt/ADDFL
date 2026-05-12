# train_lenet5_compat.py
import os, math, random, argparse
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

class LeNet5Compat(nn.Module):
    """
    兼容 adversirial.py 期望的键名/结构：
      c1: Conv2d(1, 6, 5) -> ReLU -> MaxPool(2)  # 28->24->12
      c3: Conv2d(6,16, 5) -> ReLU -> MaxPool(2)  # 12->8->4
      c5: Conv2d(16,120, 4)                      # 4->1
      flatten -> f6: Linear(120,84) -> ReLU
               -> f7: Linear(84,num_classes)
    保存时只保存 model.state_dict()，确保包含键：c1, c3, c5, f6, f7
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)    # 28->24
        self.c3 = nn.Conv2d(6, 16, kernel_size=5)    # 12->8
        self.c5 = nn.Conv2d(16, 120, kernel_size=5)  # 4->1
        self.pool = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU(inplace=True)
        self.f6  = nn.Linear(120, 84)
        self.f7  = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.c1(x)))   # 24->12
        x = self.pool(self.relu(self.c3(x)))   # 8->4
        x = self.relu(self.c5(x))              # 1x1x120
        x = torch.flatten(x, 1)                # 120
        x = self.relu(self.f6(x))
        x = self.f7(x)
        return x

def set_seed(s=42):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True

def get_loaders(train_root, bs=128, workers=4, val_ratio=0.1):
    train_tf = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.08,0.08), scale=(0.95,1.05)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5]),
    ])
    full = datasets.ImageFolder(root=train_root, transform=train_tf)
    K = len(full.classes)
    v = max(1, int(len(full)*val_ratio)); t = len(full)-v
    train_ds, val_ds = random_split(full, [t,v])
    val_ds.dataset.transform = eval_tf
    tl = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=workers, pin_memory=True)
    vl = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)
    return tl, vl, K

@torch.inference_mode()
def eval_loop(model, loader, device):
    model.eval(); crit = nn.CrossEntropyLoss()
    loss_sum=0.0; corr=0; tot=0
    for x,y in loader:
        x=x.to(device, non_blocking=True); y=y.to(device, non_blocking=True)
        o=model(x); loss=crit(o,y)
        loss_sum += loss.item()*x.size(0)
        corr += (o.argmax(1)==y).sum().item()
        tot  += x.size(0)
    return loss_sum/tot, corr/tot

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="ywt/dataset/OriginalTrainData/KMNIST")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    train_dir = os.path.join(args.data_root, "train")
    assert os.path.isdir(train_dir), f"未找到数据目录：{train_dir}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tl, vl, K = get_loaders(train_dir, bs=args.batch_size, workers=args.workers, val_ratio=args.val_ratio)

    model = LeNet5Compat(num_classes=K).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    save_path = os.path.join(args.data_root, "LeNet5.pth")
    os.makedirs(args.data_root, exist_ok=True)
    best=0.0
    print(f"开始训练：设备={device}，保存={save_path}")

    for ep in range(1, args.epochs+1):
        model.train(); run_loss=0.0; run_corr=0; run_tot=0
        for x,y in tl:
            x=x.to(device, non_blocking=True); y=y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                o=model(x); loss=crit(o,y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run_loss += loss.item()*x.size(0)
            run_corr += (o.argmax(1)==y).sum().item()
            run_tot  += x.size(0)

        tr_loss = run_loss/run_tot; tr_acc = run_corr/run_tot
        val_loss, val_acc = eval_loop(model, vl, device)
        sch.step()
        print(f"[{ep:03d}/{args.epochs}] train_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f}")

        if val_acc > best:
            best = val_acc
            # 关键：只保存纯 state_dict（键：c1,c3,c5,f6,f7）
            torch.save(model.state_dict(), save_path)
            print(f"✅ 新最佳 {best:.4f}，已保存 state_dict 到 {save_path}")

    print(f"完成。最佳验证准确率：{best:.4f}，权重：{save_path}")

if __name__ == "__main__":
    main()
