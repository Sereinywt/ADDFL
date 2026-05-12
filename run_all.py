# run_all.py —— 一次性跑完配置的组合，跑完即退出；支持已完成跳过；防多实例
import os
import sys
import json
import shlex
import subprocess

PY = sys.executable or "python"

# -------- 单实例文件锁，避免重复启动 --------
try:
    import fcntl
    _LOCK_PATH = "/tmp/run_all.lock"
    _lock_fd = open(_LOCK_PATH, "w")
    fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    _lock_fd.write(str(os.getpid()))
    _lock_fd.flush()
except Exception:
    print("[INFO] run_all.py 已在运行，当前实例自动退出。")
    sys.exit(0)

# ====== 你要批量跑的组合（自行增删）======
NOISE_TYPES = [
    "RandomLabelNoise",
    "RandomDataNoise",
    "SpecificDataNoise",
    "SpecificLabelNoise",
]
DATASETS = ["MNIST", "CIFAR10", "KMNIST","EMNIST"]
MODELS   = ["LeNet1", "LeNet5", "ResNet18", "VGG"]

# ====== 数据根目录 ======
ROOT_D1 = "/NVNUYucw/ywt/dataset"

# ====== 类文件与模型参数 ======
CLASS_JSON = {
    "MNIST":   os.path.join(ROOT_D1, "mnist_classes.json"),
    "KMNIST":  os.path.join(ROOT_D1, "kmnist_classes.json"),
    "EMNIST":  os.path.join(ROOT_D1, "emnist_classes.json"),
    "CIFAR10": os.path.join(ROOT_D1, "cifar10_classes.json"),
    "AGNEWS":  os.path.join(ROOT_D1, "agnews_classes.json"),
   # "FASHION": os.path.join(ROOT_D1, "fashion_mnist_classes.json"),
}
MODEL_ARGS = {
    "MNIST":   os.path.join(ROOT_D1, "mnist_model_args.pth"),
    "KMNIST":  os.path.join(ROOT_D1, "kmnist_model_args.pth"),
    "EMNIST":  os.path.join(ROOT_D1, "emnist_model_args.pth"),
    "CIFAR10": os.path.join(ROOT_D1, "cifar10_model_args.pth"),
    "AGNEWS":  os.path.join(ROOT_D1, "agnews_model_args.pth"),
    "FASHION": os.path.join(ROOT_D1, "fashion_mnist_model_args.pth"),
}

# ====== 输入尺寸映射 ======
IMAGE_SIZE = {
    ("MNIST",   "LeNet1"): "(28, 28, 1)",
    ("MNIST",   "LeNet5"): "(28, 28, 1)",
    ("KMNIST",  "LeNet5"): "(28, 28, 1)",
    ("KMNIST", "ResNet18"): "(32, 32, 3)",
    ("CIFAR10", "ResNet18"): "(32, 32, 3)",
    ("CIFAR10", "VGG"):    "(32, 32, 3)",
    ("EMNIST",  "LeNet5"): "(28, 28, 1)",
    ("EMNIST", "ResNet18"): "(32, 32, 1)",
}
DEFAULT_SIZE_OF_DATASET = {
    "MNIST":   "(28, 28, 1)",
    "CIFAR10": "(32, 32, 3)",
    "EMNIST":  "(28, 28, 1)",
    "KMNIST":  "(28, 28, 1)",
}

# ====== 允许组合 ======
ALLOWED_COMBINATIONS = {
    "MNIST":   ["LeNet1", "LeNet5"],
    "CIFAR10": ["ResNet18", "VGG"],
    "EMNIST":  ["LeNet5", "ResNet18"],
    "KMNIST":  ["ResNet18", "LeNet5"], 
}

def model_clean_path_of(dataset: str, model: str) -> str:
    """干净模型默认位置：优先 <model>.pth，回退 model_clean.pth"""
    p1 = os.path.join(ROOT_D1, "OriginalTrainData", dataset, f"{model}.pth")
    p2 = os.path.join(ROOT_D1, "OriginalTrainData", dataset, "model_clean.pth")
    return p1 if os.path.exists(p1) else p2

def is_result_ready(noise: str, dataset: str, model: str) -> bool:
    """判断该组合是否已有 noManual 结果（任一文件存在即视为完成）"""
    base = os.path.join("dataset", noise, dataset, "results", model)
    targets = ["noManual_sorted_score_list.json", "noManual_results_list.json"]
    return any(os.path.exists(os.path.join(base, t)) for t in targets)

def exists_required(noise: str, dataset: str, model: str) -> bool:
    """检查必需输入是否齐全；不齐则跳过该组合。"""
    train_noisy = os.path.join(ROOT_D1, noise, dataset, "train")
    train_clean = os.path.join(ROOT_D1, "OriginalTrainData", dataset, "train")
    test_root   = os.path.join(ROOT_D1, "OriginalTestData", dataset, "test")
    class_json  = CLASS_JSON.get(dataset, "")
    model_args  = MODEL_ARGS.get(dataset, "")
    clean_w     = model_clean_path_of(dataset, model)
    name2fault1 = os.path.join("dataset",  noise, dataset, "train", "name2isfault.json")
    name2fault2 = os.path.join("dataset1", noise, dataset, "train", "name2isfault.json")

    ok = all([
        os.path.isdir(train_noisy),
        os.path.isdir(train_clean),
        os.path.isdir(test_root),
        os.path.exists(class_json),
        os.path.exists(model_args),
        os.path.exists(clean_w),
        os.path.exists(name2fault1) or os.path.exists(name2fault2),
    ])
    if not ok:
        print(f"[SKIP] 缺少必要文件：noise={noise} dataset={dataset} model={model}")
        print(f"  train_noisy:  {train_noisy}  ({'OK' if os.path.isdir(train_noisy) else 'MISSING'})")
        print(f"  train_clean:  {train_clean}  ({'OK' if os.path.isdir(train_clean) else 'MISSING'})")
        print(f"  test_root:    {test_root}    ({'OK' if os.path.isdir(test_root) else 'MISSING'})")
        print(f"  class_json:   {class_json}   ({'OK' if os.path.exists(class_json) else 'MISSING'})")
        print(f"  model_args:   {model_args}   ({'OK' if os.path.exists(model_args) else 'MISSING'})")
        print(f"  clean_weight: {clean_w}      ({'OK' if os.path.exists(clean_w) else 'MISSING'})")
        print(f"  name2isfault: {name2fault1} / {name2fault2}")
    return ok

def run_one(noise: str, dataset: str, model: str):
    image_size = IMAGE_SIZE.get((dataset, model), DEFAULT_SIZE_OF_DATASET.get(dataset, "(28, 28, 1)"))

    args = [
        PY, "-B", "adversarial.py",
        "--class_path", CLASS_JSON[dataset],
        "--model_args", MODEL_ARGS[dataset],
        "--train_noisy_root", os.path.join(ROOT_D1, noise, dataset, "train"),
        "--train_clean_root", os.path.join(ROOT_D1, "OriginalTrainData", dataset, "train"),
        "--model_clean_path", model_clean_path_of(dataset, model),
        "--test_root", os.path.join(ROOT_D1, "OriginalTestData", dataset, "test"),
        "--dataset", os.path.join(".", "dataset", noise, dataset),
        "--model_name", model,
        "--image_size", image_size,
        "--image_set", "",
        "--hook_layer", "conv",
        "--rm_ratio", "0.05",
        "--retrain_epoch", "10",
        "--retrain_bs", "64",
        "--slice_num", "1",
        "--ablation", "None",
        "--dataset_name", dataset,
        "--noise_type",   noise,
    ]

    print("\n==============================")
    print(f"Run: noise={noise}  dataset={dataset}  model={model}  image_size={image_size}")
    print("==============================")
    subprocess.run(args, check=False)

def main():
    ran, skipped_ready, skipped_missing = 0, 0, 0

    for dataset in DATASETS:
        allowed = set(ALLOWED_COMBINATIONS.get(dataset, []))
        for model in MODELS:
            if model not in allowed:
                continue
            for noise in NOISE_TYPES:
                if not exists_required(noise, dataset, model):
                    skipped_missing += 1
                    continue
                if is_result_ready(noise, dataset, model):
                    print(f"[SKIP] 已有结果：{noise} | {dataset} | {model}")
                    skipped_ready += 1
                    continue
                run_one(noise, dataset, model)
                ran += 1

    # 统一汇总一次
    print("\n==== 所有任务执行完毕，尝试进行汇总 ====")
    try:
        subprocess.run([PY, "eval_all.py"], check=False)
    except Exception as e:
        print(f"[WARN] 汇总脚本 eval_all.py 执行失败：{e}")

    print(f"\n[SUMMARY] 新执行: {ran} | 跳过(已有结果): {skipped_ready} | 跳过(缺文件): {skipped_missing}")
    print("✅ run_all.py 单次执行完成")

if __name__ == "__main__":
    main()
