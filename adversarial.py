import argparse
import gc
import json
import os
import random
import sys
import time
import warnings
from tqdm import tqdm
import numpy as np

from pyod.models.vae import VAE
from sklearn.cluster import KMeans
from torch import nn, utils
import torch
from utils.dataset import dataset
from utils.models import *
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from utils.models import ResNet18_EMNIST, ResNet18_CIFAR

# ========= 不要在这里使用 args =========
# （删除了顶端那段：with open(args.class_path,'r')... 以及 data_s['Aa'] 的打印）


def build_data_s_from_folder(root, class_path):
    import os, json, glob

    # 读取类映射：{"0":0, "1":1,...} 或 {"Aa":0, "Bb":1,...}
    with open(class_path, 'r', encoding='utf-8') as f:
        class_map = json.load(f)

    # 按 label id 排序，保证顺序和训练时一致
    classes = sorted(class_map.keys(), key=lambda k: class_map[k])

    # ✅ 不再限制必须是数字类，EMNIST 的 Aa/Bb/... 也可以
    # 如果你还想保留一个简单提示，可以开这个：
    # bad = [k for k in classes if not k.isalnum()]
    # if bad:
    #     print(f"[WARN] 一些类别名包含非字母数字字符: {bad}")

    data_s = {}
    for cname in classes:
        class_dir = os.path.join(root, cname)
        if not os.path.isdir(class_dir):
            # 没有这个类目录，就设成空列表
            data_s[cname] = []
            continue

        files = []
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp'):
            files.extend(glob.glob(os.path.join(class_dir, ext)))

        # 保存相对路径： "Aa/00001.png" 这种形式
        data_s[cname] = [os.path.join(cname, os.path.basename(p)) for p in sorted(files)]

    total = sum(len(v) for v in data_s.values())
    print(f'[build_data_s_from_folder] 总计：{total} images loaded')
    return data_s

class Adversarial():
    def __init__(self, args):
        self.image_list = []
        self.gt_list = []  # 防止属性不存在
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.args = args
        classes = self.load_json(self.args.class_path)
        self.class_num = len(classes.keys())
        self.modelargs = torch.load(self.args.model_args)
        # 确保模型构造参数里包含正确的类别数（按你的 LeNet1 构造器字段名来）
        # 常见字段名：'num_classes' 或 'n_classes' 或 'out_dim'
        if 'num_classes' in self.modelargs:
            self.modelargs['num_classes'] = self.class_num
        elif 'n_classes' in self.modelargs:
            self.modelargs['n_classes'] = self.class_num
        elif 'out_dim' in self.modelargs:
            self.modelargs['out_dim'] = self.class_num
        else:
            # 如果 model_args 里没有这些键，直接加一个最常见的
            self.modelargs['num_classes'] = self.class_num

        # 输出目录就绪
        for sub in ['feature', 'results',  'mutmodel']:
            p = os.path.join(self.args.dataset, f'{sub}/{self.args.model_name}')
            os.makedirs(p, exist_ok=True)

    def pgd_attack(self, model, images, labels, epsilon=0.03, alpha=0.007, iters=7):
        ori_images = images.data
        for i in range(iters):
            images.requires_grad = True
            outputs = model(images)
            model.zero_grad()
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            adv_images = images + alpha * images.grad.sign()
            eta = torch.clamp(adv_images - ori_images, min=-epsilon, max=epsilon)
            images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
        return images
    def _sanitize_init_kwargs(self, model_class):
        # """从 self.modelargs 里只保留构造函数能接收的键，并注入正确的类别数。"""
        import inspect

        raw = dict(self.modelargs) if isinstance(self.modelargs, dict) else {}
        # 这些键是训练/预处理用的，绝不能传进构造函数
        ban = {'optimizer', 'scheduler', 'transform', 'loss_fn', 'criterion', 'augmentation'}
        for k in list(raw.keys()):
            if k in ban:
                raw.pop(k)

        # 仅保留构造函数签名里出现的参数
        try:
            sig = inspect.signature(model_class.__init__)
            valid = set(sig.parameters.keys()) - {'self'}
            kwargs = {k: v for k, v in raw.items() if k in valid}
        except Exception:
            # 如果拿不到签名，就先空 kwargs，后续用 fc 替换兜底
            kwargs = {}

        # 注入（或覆盖）类别数
        for cand in ['num_classes', 'n_classes', 'out_dim', 'num_outputs', 'classes']:
            if cand in kwargs or (cand in valid if 'valid' in locals() else False):
                kwargs[cand] = self.class_num
                break

        return kwargs

    def _build_model(self):
        ds = str(getattr(self.args, "dataset_name", "")).upper()
        model_name = self.args.model_name

        if isinstance(model_name, str) and model_name == "ResNet18":
            # 这里根据数据集选真正的 ResNet18 实现
            if ds == "CIFAR10":
                model_class = ResNet18_CIFAR          # 3 通道 CIFAR 模型
            elif ds in ("KMNIST", "EMNIST"):
                model_class = ResNet18_EMNIST         # 1 通道 KMNIST/EMNIST 模型
            else:
                raise ValueError(f"暂时没有为数据集 {ds} 定义 ResNet18 模型，请自己加一个映射")
        else:
            # 其他模型名保持原逻辑
            model_class = eval(model_name) if isinstance(model_name, str) else model_name

        # 用你原来的参数清洗逻辑
        kwargs = self._sanitize_init_kwargs(model_class)
        try:
            model = model_class(**kwargs)
        except TypeError as e:
            print(f"[WARN] _build_model: 用 kwargs 构造 {model_class.__name__} 失败：{e}，改用无参构造。")
            model = model_class()

        if getattr(self, "use_cuda", False):
            model = model.cuda()
        return model

        # 兜底：把最后一层强制改成 class_num（常见命名：fc / classifier / linear / output）
        def _fix_last_linear(m):
            if hasattr(m, 'fc') and isinstance(m.fc, nn.Linear):
                if m.fc.out_features != self.class_num:
                    m.fc = nn.Linear(m.fc.in_features, self.class_num)
                return True
            if hasattr(m, 'classifier') and isinstance(m.classifier, nn.Linear):
                if m.classifier.out_features != self.class_num:
                    m.classifier = nn.Linear(m.classifier.in_features, self.class_num)
                return True
            if hasattr(m, 'linear') and isinstance(m.linear, nn.Linear):
                if m.linear.out_features != self.class_num:
                    m.linear = nn.Linear(m.linear.in_features, self.class_num)
                return True
            if hasattr(m, 'output') and isinstance(m.output, nn.Linear):
                if m.output.out_features != self.class_num:
                    m.output = nn.Linear(m.output.in_features, self.class_num)
                return True
            return False

        _fix_last_linear(model)
        return model


    def adv_train(self, model, data_loader, attack_steps=7):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        for images, labels, _ in tqdm(data_loader, desc="PGD Adversarial Training"):
            images, labels = images.to(self.device), labels.to(self.device)
            with torch.no_grad():
                outputs = model(images)
                losses = nn.CrossEntropyLoss(reduction='none')(outputs, labels)
                weights = torch.where(losses < 2.0, torch.ones_like(losses), 0.3 * torch.ones_like(losses))
            adv_images = self.pgd_attack(model, images, labels, epsilon=0.03, alpha=0.007, iters=attack_steps)
            adv_outputs = model(adv_images)
            loss = (nn.CrossEntropyLoss(reduction='none')(adv_outputs, labels) * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return model

    def load_json(self, load_path):
        with open(load_path) as f:
            data = json.load(f)
        return data

    def save_as_json(self, data, save_path):
        data_json = json.dumps(data, indent=4)
        with open(save_path, 'w') as file:
            file.write(data_json)

    def getrandomfeature(self, model, model_adv, class_num, fake_per_class=10):
        import scipy.stats
        import numpy as np
        import torch
    
        modelargs = torch.load(self.args.model_args)
    
        try:
            org_entropy_list = self.load_json(os.path.join(self.args.dataset, 'feature', self.args.model_name, 'org_entropy_list.json'))
            fgsm_entropy_list = self.load_json(os.path.join(self.args.dataset, 'feature', self.args.model_name, 'fgsm_entropy_list.json'))
            org_maxconf_list = self.load_json(os.path.join(self.args.dataset, 'feature', self.args.model_name, 'org_maxconf_list.json'))
            fgsm_maxconf_list = self.load_json(os.path.join(self.args.dataset, 'feature', self.args.model_name, 'fgsm_maxconf_list.json'))
            diff_feature_list = self.load_json(os.path.join(self.args.dataset, 'feature', self.args.model_name, 'diff_feature_list.json'))
            diff_softmax_L2_list = [x[0] for x in diff_feature_list]
            diff_loss_list = [x[1] for x in diff_feature_list]
            diff_entropy_list = [x[2] for x in diff_feature_list]
            diff_maxconf_list = [x[3] for x in diff_feature_list]
            pred_equal_list = [x[4] for x in diff_feature_list]
        except Exception:
            org_entropy_list = [2.0] * 1000
            fgsm_entropy_list = [2.0] * 1000
            org_maxconf_list = [0.9] * 1000
            fgsm_maxconf_list = [0.9] * 1000
            diff_softmax_L2_list = [0.2] * 1000
            diff_loss_list = [0.1] * 1000
            diff_entropy_list = [0.05] * 1000
            diff_maxconf_list = [0.1] * 1000
            pred_equal_list = [1] * 1000
    
        org_entropy_mean, org_entropy_std = np.mean(org_entropy_list), np.std(org_entropy_list)
        fgsm_entropy_mean, fgsm_entropy_std = np.mean(fgsm_entropy_list), np.std(fgsm_entropy_list)
        org_maxconf_mean, org_maxconf_std = np.mean(org_maxconf_list), np.std(org_maxconf_list)
        fgsm_maxconf_mean, fgsm_maxconf_std = np.mean(fgsm_maxconf_list), np.std(fgsm_maxconf_list)
        diff_softmax_L2_mean, diff_softmax_L2_std = np.mean(diff_softmax_L2_list), np.std(diff_softmax_L2_list)
        diff_loss_mean, diff_loss_std = np.mean(diff_loss_list), np.std(diff_loss_list)
        diff_entropy_mean, diff_entropy_std = np.mean(diff_entropy_list), np.std(diff_entropy_list)
        diff_maxconf_mean, diff_maxconf_std = np.mean(diff_maxconf_list), np.std(diff_maxconf_list)
        pred_equal_prob = np.mean(pred_equal_list)
    
        def model_out(model, X, Y):
            model.to(self.device)
            model.eval()
            loss_fn = modelargs['loss_fn']
            softmax_func = nn.Softmax(dim=1)
            with torch.no_grad():
                X = X.to(self.device)
                if (
                    self.args.model_name == 'ResNet18'
                    and str(self.args.dataset_name).upper() in ('KMNIST', 'EMNIST', 'MNIST')
                    and X.dim() == 4
                    and X.size(1) == 3
                ):
                    X = X[:, 0:1, :, :]
    
                if self.args.model_name == 'TCDCNN':
                    X = X.float()
                    y = Y.float().to(self.device)
                else:
                    y = torch.from_numpy(np.array([Y])).long().to(self.device)
    
                out = model(X)
                if self.args.model_name == 'TCDCNN':
                    soft_output = out
                    loss = model.loss([out], [y]).cpu().numpy().item()
                    sfout = soft_output.cpu().numpy()[0].tolist()
                else:
                    soft_output = softmax_func(out)
                    loss = loss_fn(soft_output, y).cpu().numpy().item()
                    sfout = soft_output.cpu().numpy()[0].tolist()
            return sfout, loss
    
        image_size = eval(self.args.image_size)
        Feature = []
    
        for i in range(class_num):
            for _ in range(fake_per_class):
                # 注意：X 要放到这里，每次 fake 都重新随机生成
                if image_size is None and self.args.model_name != 'TCDCNN':
                    X = torch.randint(0, 95805, (1, 100))
                elif self.args.model_name == 'TCDCNN':
                    X = torch.rand(1, 1, 40, 40)
                else:
                    X = torch.rand(1, image_size[2], image_size[0], image_size[1])
    
                if self.args.model_name == 'TCDCNN':
                    label = np.zeros((1, 10))
                    for j in range(10):
                        label[0, j] = 0 + (40 - 0) * np.random.random()
                    label = label.astype('float64')
                    label = torch.from_numpy(label)
                else:
                    label = i
    
                sfout_org, loss_org = model_out(model, X, label)
                sfout_fgsm, loss_fgsm = model_out(model_adv, X, label)
    
                if self.args.model_name == 'TCDCNN':
                    gt = label.cpu().numpy()[0].tolist()
                else:
                    gt = np.zeros(self.class_num)
                    gt[i % self.class_num] = 1
                    gt = gt.tolist()
    
                org_entropy = float(np.random.normal(org_entropy_mean, org_entropy_std))
                fgsm_entropy = float(np.random.normal(fgsm_entropy_mean, fgsm_entropy_std))
                org_maxconf = float(np.clip(np.random.normal(org_maxconf_mean, org_maxconf_std), 0, 1))
                fgsm_maxconf = float(np.clip(np.random.normal(fgsm_maxconf_mean, fgsm_maxconf_std), 0, 1))
                diff_softmax_L2 = float(np.abs(np.random.normal(diff_softmax_L2_mean, diff_softmax_L2_std)))
                diff_loss = float(np.random.normal(diff_loss_mean, diff_loss_std))
                diff_entropy = float(np.random.normal(diff_entropy_mean, diff_entropy_std))
                diff_maxconf = float(np.random.normal(diff_maxconf_mean, diff_maxconf_std))
                pred_equal = int(np.random.rand() < pred_equal_prob)
    
                tmp_feature = [
                    *sfout_org, *gt, *sfout_fgsm, loss_org, loss_fgsm,
                    org_entropy, fgsm_entropy, org_maxconf, fgsm_maxconf,
                    diff_softmax_L2, diff_loss, diff_entropy, diff_maxconf, pred_equal
                ]
                Feature.append(tmp_feature)
    
        return Feature

    def Iteration(self, Feature):
        import time, os, numpy as np
        from sklearn.linear_model import LogisticRegression

        t0 = time.time()
        Feature = np.asarray(Feature)

        # ===================== 1. 检查 image_list =====================
        nm_image_list_path = os.path.join(
            self.args.dataset, 'feature', self.args.model_name, 'image_list.json'
        )
        if (not hasattr(self, 'image_list')) or (not self.image_list) or (len(self.image_list) != Feature.shape[0]):
            if os.path.exists(nm_image_list_path):
                self.image_list = self.load_json(nm_image_list_path)
            else:
                print('❌ 缺失 image_list.json 且无法补救，请检查特征提取和保存流程！')
                exit(1)

        # ===================== 2. 检查 gt_list =====================
        nm_gt_list_path = os.path.join(
            self.args.dataset, 'feature', self.args.model_name, 'gt_list.json'
        )
        if (not hasattr(self, 'gt_list')) or (not self.gt_list) or (len(self.gt_list) != Feature.shape[0]):
            if os.path.exists(nm_gt_list_path):
                self.gt_list = self.load_json(nm_gt_list_path)
                if isinstance(self.gt_list, dict):
                    print("⚠️ 检测到 gt_list 为 dict，将自动转换为 list。")
                    self.gt_list = list(self.gt_list.values())
            else:
                print('❌ 缺失 gt_list.json，请检查特征提取和保存流程！')
                exit(1)

        # ===================== 3. 加载 random_index =====================
        rand_idx_path = os.path.join(
            self.args.dataset, "feature", self.args.model_name, "random_index.json"
        )

        print("\n====== [DEBUG] Feature & gt_list 状态 ======")
        print("Feature type:", type(Feature), "Feature shape:", getattr(Feature, "shape", None))
        gt_list = getattr(self, "gt_list", None)
        print("gt_list type:", type(gt_list), "len:", len(gt_list) if gt_list is not None else 0)
        print("===========================================\n")

        # 强制要求 Feature_Summary 已经生成 random_index.json
        if not os.path.exists(rand_idx_path):
            raise FileNotFoundError(
                f"❌ 关键文件缺失: {rand_idx_path}。\n"
                "请确保先执行 Feature_Summary（它会为每个类别选出 1 个最优样本并生成 random_index.json）。"
            )

        random_index = self.load_json(rand_idx_path)
        if isinstance(random_index, dict):
            print('ERROR: 当前 random_index.json 还是旧版，请彻底手动删除后再运行！')
            exit(1)

        random_index = list(map(int, random_index))
        TOPK_CLEAN_PER_CLASS = 10
        sample_feature = np.asarray(Feature)[random_index]    # 每类 top-k 个 Max-Conf 真实样本
        
        expected_min = self.class_num
        expected_max = self.class_num * TOPK_CLEAN_PER_CLASS
        if not (expected_min <= sample_feature.shape[0] <= expected_max):
            print(f"⚠️ 警告：期望真实样本数在 [{expected_min}, {expected_max}] 之间，"
                  f"但实际有 {sample_feature.shape[0]} 条，请检查 random_index.json")

        # ===================== 4. 构造模型 & 随机特征 =====================
        # 这里不再使用 model_class，直接用已有的 _build_model
        model = self._build_model()
        model.load_state_dict(
            torch.load(
                os.path.join(self.args.dataset, "mutmodel", self.args.model_name, "model_clean.pth"),
                map_location=self.device
            )
        )

        model_adv = self._build_model()
        model_adv.load_state_dict(
            torch.load(
                os.path.join(self.args.dataset, "mutmodel", self.args.model_name, "model_adv.pth"),
                map_location=self.device
            )
        )

        random_feat = np.asarray(
            self.getrandomfeature(model, model_adv, self.class_num, fake_per_class=10)
        )

        if sample_feature.shape[1] != random_feat.shape[1]:
            raise ValueError(
                f"Dim mismatch between sample_feature ({sample_feature.shape}) "
                f"and random_feat ({random_feat.shape})"
            )

        # 拼成 LR 训练集：前半真实样本，后半随机样本
        sample_feature = np.concatenate([sample_feature, random_feat], axis=0)
        n_rand = random_feat.shape[0]
        n_real = sample_feature.shape[0] - n_rand

        print(f"Training samples: real = {n_real} (Max-Conf), random = {n_rand} (Fake)")
        Y = np.concatenate([np.zeros(n_real), np.ones(n_rand)]).astype(int)

        # ===================== 5. 全体样本打分 =====================
        img_list = np.asarray(self.image_list)
        gt_list = np.asarray(self.gt_list)

        idx_shuffle = np.random.permutation(len(Feature))
        Feature_shuffled = Feature[idx_shuffle]
        img_shuffled = img_list[idx_shuffle]
        # gt_shuffled = gt_list[idx_shuffle]   # 如暂时不用可以注释掉

        lr = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced')
        lr.fit(sample_feature, Y)

        scores = lr.predict_proba(Feature_shuffled)[:, 1]   # 分数越大，越像“随机 / 假样本”
        idx_sorted = np.argsort(-scores)

        Feature_sorted = Feature_shuffled[idx_sorted]
        img_sorted = img_shuffled[idx_sorted]
        score_sorted = scores[idx_sorted].tolist()

        if self.args.model_name != "TCDCNN":
            img_sorted = np.array([os.path.basename(p) for p in img_sorted])

        sorted_pairs = [[img, score] for img, score in zip(img_sorted, score_sorted)]

        elapsed = time.time() - t0
        print(f"Iteration finished in {elapsed:.2f}s")

        return (img_sorted.tolist(), Feature_sorted.tolist(), score_sorted, elapsed, sorted_pairs)

    
    def run(self, data_s):
        adversarial_time = {'Select Subset': -1, 'Mutation&Extraction': -1, 'Initialize Susp': -1, 'all': -1}

        # 数据加载
        if isinstance(data_s, dict):
            datasets = dataset(root=self.args.train_noisy_root, classes_path=self.args.class_path,
                               transform=self.modelargs['transform'],
                               image_size=eval(self.args.image_size),
                               image_set=self.args.image_set, data_s=data_s)
            data_loader = torch.utils.data.DataLoader(datasets, batch_size=256, shuffle=False, num_workers=4,
                                                      pin_memory=True)
        else:
            data_loader = data_s

        print(f"Debug: Current model_name is: {self.args.model_name}")

        nm_results_path = os.path.join(
            self.args.dataset, 'results', self.args.model_name, 'Adversarial_results_list.json'
        )
        nm_feature_path = os.path.join(
            self.args.dataset, 'feature', self.args.model_name, 'Adversarial_full_Feature.json'
        )
        nm_score_path = os.path.join(
            self.args.dataset, 'results', self.args.model_name, 'Adversarial_sorted_score_list.json'
        )
        nm_image_list_path = os.path.join(
            self.args.dataset, 'feature', self.args.model_name, 'image_list.json'
        )

        if all([
            os.path.exists(nm_results_path),
            os.path.exists(nm_feature_path),
            os.path.exists(nm_score_path),
            self.args.ablation == 'None'
        ]):
            print('adversarial 结果已存在，直接加载。')
            adversarial_results_list = self.load_json(nm_results_path)
            adversarial_full_Feature = self.load_json(nm_feature_path)
            adversarial_sorted_score_list = self.load_json(nm_score_path)
            if not self.image_list and os.path.exists(nm_image_list_path):
                self.image_list = self.load_json(nm_image_list_path)
        else:
            print('adversarial 结果不存在，开始自动处理...')
            print("[DEBUG] Feature_Summary 用到的 noisy_root 路径：", self.args.train_noisy_root)
            print("[DEBUG] 该目录存在吗？", os.path.exists(self.args.train_noisy_root))
        
            if self.args.ablation == 'None':
                full_feature_path = os.path.join(
                    self.args.dataset, 'feature', self.args.model_name, 'full_Feature.json'
                )
            else:
                feature_type = self.args.ablation
                full_feature_path = os.path.join(
                    self.args.dataset, 'feature', self.args.model_name, f'full_Feature_{feature_type}.json'
                )
        
            if os.path.exists(full_feature_path):
                full_Feature = self.load_json(full_feature_path)
                print('Feature loaded')
            else:
                full_Feature, Select_Subset_time, Mutation_Extraction_time = self.Feature_Summary(data_loader)
                adversarial_time['Select Subset'] = Select_Subset_time
                adversarial_time['Mutation&Extraction'] = Mutation_Extraction_time
                self.save_as_json(full_Feature, full_feature_path)
        
            print('start adversarial iteration')
            img_sorted, Feature_sorted, score_sorted, Initialize_Susp_time, sorted_pairs = self.Iteration(full_Feature)
            print('adversarial iteration finished')
            adversarial_time['Initialize Susp'] = Initialize_Susp_time
        
            adversarial_results_list = img_sorted
            adversarial_full_Feature = Feature_sorted
            adversarial_sorted_score_list = sorted_pairs
        
            if self.args.ablation == 'None':
                self.save_as_json(img_sorted, nm_results_path)
                self.save_as_json(Feature_sorted, nm_feature_path)
                self.save_as_json(sorted_pairs, nm_score_path)
                if self.image_list:
                    self.save_as_json(self.image_list, nm_image_list_path)
        
        print("Debug: Finished adversarial results loading/generation.")

        # === 插入开始：统计前三阶段总时长（可选写盘） ===
        adversarial_time['all'] = 0
        for k in ['Select Subset', 'Mutation&Extraction', 'Initialize Susp']:
            if adversarial_time.get(k, -1) != -1:
                adversarial_time['all'] += adversarial_time[k]

        # 如果你想把时间写到文件（可选）：
        if all(adversarial_time.get(k, -1) != -1 for k in ['Select Subset', 'Mutation&Extraction', 'Initialize Susp']):
            adversarial_time_path = os.path.join(
                self.args.dataset, 'results', self.args.model_name, 'adversarial_time.json'
            )
            os.makedirs(os.path.dirname(adversarial_time_path), exist_ok=True)
            self.save_as_json(adversarial_time, adversarial_time_path)
        # === 插入结束 ===

        if self.args.model_name == 'WaveMix':
            return adversarial_results_list, adversarial_sorted_score_list, adversarial_time

        return adversarial_results_list, adversarial_sorted_score_list, adversarial_time

    def Feature_Summary(self, data_s, image_paths=None):
        import time, numpy as np, random, torch, scipy.stats
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)

        t0 = time.time()  # ← 新增：计时起点
        model_clean = self._build_model()
        model_clean.load_state_dict(torch.load(self.args.model_clean_path, map_location=self.device))
        model_clean.to(self.device); model_clean.eval()
        torch.save(model_clean.state_dict(), os.path.join(self.args.dataset, 'mutmodel', self.args.model_name, 'model_clean.pth'))

        noisy_root  = self.args.train_noisy_root
        class_path  = self.args.class_path
        noisy_data_s = build_data_s_from_folder(noisy_root, class_path)
        datasets_noisy = dataset(root=noisy_root, classes_path=class_path,
                                 transform=self.modelargs['transform'],
                                 image_size=eval(self.args.image_size),
                                 image_set='', data_s=noisy_data_s)
        data_loader_noisy = torch.utils.data.DataLoader(datasets_noisy, batch_size=128, shuffle=True, num_workers=4)

        model_adv_path = os.path.join(self.args.dataset, 'mutmodel', self.args.model_name, 'model_adv.pth')
        if os.path.exists(model_adv_path):
            model_adv = self._build_model()
            model_adv.load_state_dict(torch.load(model_adv_path, map_location=self.device))
            model_adv.to(self.device); model_adv.eval()
            print('已加载缓存对抗模型')
        else:
            print('对抗训练生成对抗模型...')
            model_adv = self.adv_train(model_clean, data_loader_noisy, attack_steps=7)
            torch.save(model_adv.state_dict(), model_adv_path)
            model_adv.eval()

        if isinstance(data_s, dict):
            datasets_eval = dataset(root=noisy_root, classes_path=class_path,
                                    transform=self.modelargs['transform'],
                                    image_size=eval(self.args.image_size),
                                    image_set='', data_s=data_s)
            data_loader = torch.utils.data.DataLoader(datasets_eval, batch_size=128, shuffle=False, num_workers=4)
        else:
            data_loader = data_s

        ORG, ADV = self.get_feature(model_clean, model_adv, data_loader)
        image_list, gt_list, org_SFM_list, org_Loss_list, org_entropy_list, org_maxconf_list = zip(*ORG)
        adv_SFM_list,  adv_Loss_list,  adv_entropy_list,  adv_maxconf_list = zip(*ADV)
        assert all(isinstance(x, int) for x in gt_list[:10]), f"gt_list 错误: {[type(x) for x in gt_list[:10]]} {gt_list[:10]}"

        TOPK_CLEAN_PER_CLASS = 10

        # ===== 选择 clean anchors：单独计时 =====
        t_select0 = time.time()

        TOPK_CLEAN_PER_CLASS = 10
        random_index = []
        for i in range(self.class_num):
            idx_this_class = np.where(np.array(gt_list) == i)[0]
            if len(idx_this_class) == 0:
                continue

            conf_list = np.array([org_SFM_list[j][i] for j in idx_this_class], dtype=float)
            topk = min(TOPK_CLEAN_PER_CLASS, len(idx_this_class))

            selected_local = np.argsort(-conf_list)[:topk]
            selected_global = idx_this_class[selected_local]
            random_index.extend([int(x) for x in selected_global])

        self.save_as_json(
            random_index,
            os.path.join(self.args.dataset, 'feature', self.args.model_name, 'random_index.json')
        )

        Select_Subset_time = time.time() - t_select0
        print(f"[INFO] clean anchors selected: {len(random_index)} "
              f"(~{TOPK_CLEAN_PER_CLASS} per class, total classes={self.class_num})")
        print(f"[INFO] clean anchors selected: {len(random_index)} "
              f"(~{TOPK_CLEAN_PER_CLASS} per class, total classes={self.class_num})")

        gt_one_hot_path = os.path.join(self.args.dataset, 'feature', self.args.model_name, 'gt_one_hot_list.json')
        if os.path.exists(gt_one_hot_path):
            gt_one_hot_list = self.load_json(gt_one_hot_path)
        else:
            gt_one_hot = np.zeros((len(gt_list), self.class_num))
            for i, gt in enumerate(gt_list): gt_one_hot[i][gt] = 1
            gt_one_hot_list = gt_one_hot.tolist()
            self.save_as_json(gt_one_hot_list, gt_one_hot_path)

        gt_list_path = os.path.join(self.args.dataset, 'feature', self.args.model_name, 'gt_list.json')
        self.save_as_json(list(gt_list), gt_list_path)

        print('\nfeature extraction finished, start to SUMMARY...')
        Feature = []
        for img, gt, org_SFM, adv_SFM, org_Loss, adv_Loss, org_entropy, adv_entropy, org_maxconf, adv_maxconf in \
                zip(image_list, gt_one_hot_list, org_SFM_list, adv_SFM_list, org_Loss_list, adv_Loss_list,
                    org_entropy_list, adv_entropy_list, org_maxconf_list, adv_maxconf_list):
            org_pred = int(np.argmax(org_SFM)); adv_pred = int(np.argmax(adv_SFM))
            pred_equal = 1 if org_pred == adv_pred else 0
            diff_softmax_L2 = float(np.linalg.norm(np.array(org_SFM) - np.array(adv_SFM)))
            diff_loss       = float(adv_Loss - org_Loss)
            diff_entropy    = float(adv_entropy - org_entropy)
            diff_maxconf    = float(adv_maxconf - org_maxconf)
            tmp_feature = [*org_SFM, *gt, *adv_SFM, org_Loss, adv_Loss, org_entropy, adv_entropy,
                           org_maxconf, adv_maxconf, diff_softmax_L2, diff_loss, diff_entropy, diff_maxconf, pred_equal]
            Feature.append(tmp_feature)

        print(f'SUMMARY finished. Total samples in Feature: {len(Feature)}')
        from collections import Counter
        print("Label distribution in gt_list:", dict(Counter(gt_list)))

        # ===== 选择 clean anchors：单独计时 =====
        t_select0 = time.time()

        TOPK_CLEAN_PER_CLASS = 10
        random_index = []
        for i in range(self.class_num):
            idx_this_class = np.where(np.array(gt_list) == i)[0]
            if len(idx_this_class) == 0:
                continue

            conf_list = np.array([org_SFM_list[j][i] for j in idx_this_class], dtype=float)
            topk = min(TOPK_CLEAN_PER_CLASS, len(idx_this_class))

            selected_local = np.argsort(-conf_list)[:topk]
            selected_global = idx_this_class[selected_local]
            random_index.extend([int(x) for x in selected_global])

        self.save_as_json(
            random_index,
            os.path.join(self.args.dataset, 'feature', self.args.model_name, 'random_index.json')
        )
        print(f"[INFO] clean anchors selected: {len(random_index)} "
              f"(~{TOPK_CLEAN_PER_CLASS} per class, total classes={self.class_num})")

        total_feature_summary_time = time.time() - t0
        Mutation_Extraction_time = total_feature_summary_time - Select_Subset_time

        self.image_list = list(image_paths) if image_paths is not None else list(image_list)
        self.gt_list    = list(gt_list)
        nm_image_list_path = os.path.join(self.args.dataset, 'feature', self.args.model_name, 'image_list.json')
        self.save_as_json(list(self.image_list), nm_image_list_path)

        print('Feature_Summary样本:', Feature[0])
        print('Feature_Summary特征数:', len(Feature[0]))
        return Feature, Select_Subset_time, Mutation_Extraction_time

        self.image_list = list(image_paths) if image_paths is not None else list(image_list)
        self.gt_list    = list(gt_list)
        nm_image_list_path = os.path.join(self.args.dataset, 'feature', self.args.model_name, 'image_list.json')
        self.save_as_json(list(self.image_list), nm_image_list_path)

        print('Feature_Summary样本:', Feature[0])
        print('Feature_Summary特征数:', len(Feature[0]))
        return Feature, Select_Subset_time, Mutation_Extraction_time


    def get_feature(self, model, model_adv, data_loader):
        import scipy.stats
        model.to(self.device); model_adv.to(self.device)
        model.eval(); model_adv.eval()
        ORG, ADV = [], []
        softmax = torch.nn.Softmax(dim=1)
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

        with torch.no_grad():
            for images, labels, image_paths in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                out_org = model(images); out_adv = model_adv(images)
                sf_org = softmax(out_org).cpu().numpy()
                sf_adv = softmax(out_adv).cpu().numpy()
                loss_org = loss_fn(out_org, labels).cpu().numpy()
                loss_adv = loss_fn(out_adv, labels).cpu().numpy()

                if isinstance(image_paths, tuple):
                    image_paths = list(image_paths)

                for i in range(images.size(0)):
                    entropy_org = float(scipy.stats.entropy(sf_org[i] + 1e-8))
                    max_conf_org = float(np.max(sf_org[i]))
                    entropy_adv = float(scipy.stats.entropy(sf_adv[i] + 1e-8))
                    max_conf_adv = float(np.max(sf_adv[i]))
                    ORG.append((image_paths[i], int(labels[i].cpu()), sf_org[i].tolist(), float(loss_org[i]), entropy_org, max_conf_org))
                    ADV.append((sf_adv[i].tolist(), float(loss_adv[i]), entropy_adv, max_conf_adv))
        print("sf_org shape:", sf_org.shape, "loss_org shape:", loss_org.shape)
        return ORG, ADV


def data_slice(class_path, data_root, slice_num=1):
    import random, os, json
    random.seed(2023)
    with open(class_path, 'r') as f:
        classes = list(json.load(f).keys())
    classes = [c for c in classes if c.isdigit()]

    result = {i: {} for i in range(int(slice_num))}
    for cname in classes:
        cdir = os.path.join(data_root, cname)
        if not os.path.isdir(cdir):
            result[0][cname] = []
            continue
        img_list = [fname for fname in os.listdir(cdir)
                    if not fname.startswith('.') and os.path.isfile(os.path.join(cdir, fname))]
        random.shuffle(img_list)
        slice_len = len(img_list) // int(slice_num) if int(slice_num) > 0 else len(img_list)
        for i in range(int(slice_num)):
            beg, end = i * slice_len, (i + 1) * slice_len if i < int(slice_num) - 1 else len(img_list)
            result[i][cname] = img_list[beg:end]
    print('data slice done with slice num:', slice_num)
    return result


if __name__ == "__main__":
    import argparse, os, json, random

    # 统一工作目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print("✅ 当前工作目录（working directory）是：", os.getcwd())

    # ========= 第1阶段：只解析控制维度 =========
    p0 = argparse.ArgumentParser(add_help=False)
    p0.add_argument("--dataset_name", default="MNIST")             # 例：MNIST / EMNIST
    p0.add_argument("--noise_type",   default="RandomLabelNoise")  # 例：RandomLabelNoise / RandomDataNoise / ...
    p0.add_argument("--model_name",   default="LeNet5")            # 例：LeNet1 / LeNet5 / ...
    known, _ = p0.parse_known_args()

    DS    = known.dataset_name
    NOISE = known.noise_type
    MODEL = known.model_name

    # 输入尺寸映射（需要时自行扩充）
    IMAGE_SIZE_MAP = {
        ("MNIST", "LeNet1"): "(28, 28, 1)",
        ("MNIST", "LeNet5"): "(28, 28, 1)",
        # ("EMNIST","LeNet1"): "(28, 28, 1)",
        # ...
    }
    image_size_default = IMAGE_SIZE_MAP.get((DS, MODEL), "(28, 28, 1)")

    # dataset 根（按你的环境）
    ROOT_D1 = "/NVNUYucw/ywt/dataset"

    # 基于三元组动态推导的默认路径（懒填充用）
    def _default_clean_weight():
        p1 = f"{ROOT_D1}/OriginalTrainData/{DS}/{MODEL}.pth"
        p2 = f"{ROOT_D1}/OriginalTrainData/{DS}/model_clean.pth"
        return p1 if os.path.exists(p1) else p2

    # ========= 第2阶段：完整参数表（可被显式覆盖） =========
    parser = argparse.ArgumentParser(parents=[p0])

    # 读入（来自 dataset）
    parser.add_argument('--model_clean_path', default=_default_clean_weight())
    parser.add_argument('--train_clean_root', default=f'{ROOT_D1}/OriginalTrainData/{DS}/train')
    parser.add_argument('--train_noisy_root', default=f'{ROOT_D1}/{NOISE}/{DS}/train')
    parser.add_argument('--test_root',        default=f'{ROOT_D1}/OriginalTestData/{DS}/test')

    # 写出（到 ./dataset/...）
    parser.add_argument('--dataset',    default=f'./dataset/{NOISE}/{DS}', help='output root')
    parser.add_argument('--model',      default=f'./dataset/{NOISE}/{DS}/{MODEL}.pth')
    # parser.add_argument('--model_name', default=MODEL)

    # 其它必需文件（通常固定在 dataset）
    parser.add_argument('--class_path', default=f'{ROOT_D1}/mnist_classes.json')
    parser.add_argument('--image_size', default=image_size_default)
    parser.add_argument('--model_args', default=f'{ROOT_D1}/mnist_model_args.pth')

    # 训练/控制参数
    parser.add_argument('--image_set',     default='')
    parser.add_argument('--hook_layer',    default='conv')
    parser.add_argument('--rm_ratio',      default=0.05, type=float)
    parser.add_argument('--retrain_epoch', default=10,   type=int)
    parser.add_argument('--retrain_bs',    default=64,   type=int)
    parser.add_argument('--slice_num',     default=1,    type=int)
    parser.add_argument('--ablation',      default='None')
    parser.add_argument('--seed', default=2023, type=int)

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ===== Debug：确认关键输入 =====
    with open(args.class_path, 'r') as f:
        ks = list(json.load(f).keys())
    print('[DEBUG] dataset_name =', DS, 'noise_type =', NOISE, 'model_name =', args.model_name)
    print('[DEBUG] class_path   =', args.class_path, '->', ks[:10])
    print('[DEBUG] train_noisy_root =', args.train_noisy_root)
    print('[DEBUG] output root     =', args.dataset)

    # ===== 构建 data_s（MNIST/EMNIST 都按类名子目录收集）=====
    if DS != 'MTFL':
        data_s = build_data_s_from_folder(args.train_noisy_root, args.class_path)
    else:
        data_s = None

    # ===== 跑起来 =====
    df = Adversarial(args)
    adversarial_results_list, adversarial_sorted_score_list, adversarial_time = df.run(data_s)