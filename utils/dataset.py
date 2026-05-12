# utils/dataset.py

import json
import os
import pickle
from os.path import isfile

import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as T
from matplotlib import pyplot as plt
from torchvision.transforms import transforms
from torchtext.data.utils import get_tokenizer
from collections import Counter


def save_as_json(data_, save_path):
    data_json = json.dumps(data_, indent=4)
    with open(save_path, 'w') as file:
        file.write(data_json)


def load_json(load_path):
    with open(load_path, 'r') as f:
        data_ = json.load(f)
    return data_


# =========================
# Simple vocabulary builder
# =========================
class Vocab:
    def __init__(self, datalist_full_paths, min_freq=5):
        """
        datalist_full_paths: [(absolute_txt_path, label), ...]
        只对以 .txt 结尾的样本统计词频并建表
        """
        print("[Vocab] Creating vocabulary (min_freq=%d) ..." % min_freq)
        tokenizer = get_tokenizer('basic_english')

        counter = Counter()
        for p, _ in datalist_full_paths:
            if isinstance(p, str) and p.endswith('.txt'):
                try:
                    with open(p, 'r', encoding='utf-8') as f:
                        txt = f.read()
                except UnicodeDecodeError:
                    with open(p, 'r', encoding='latin-1') as f:
                        txt = f.read()
                counter.update(tokenizer(txt))

        # special tokens
        self.pad = '<pad>'
        self.unk = '<unk>'

        # id mapping
        self.token_to_idx = {self.pad: 0, self.unk: 1}
        for tok, freq in counter.most_common():
            if freq >= min_freq and tok not in self.token_to_idx:
                self.token_to_idx[tok] = len(self.token_to_idx)

        print(f"[Vocab] Done. vocab_size={len(self.token_to_idx)}")

    def __len__(self):
        return len(self.token_to_idx)

    def transform(self, sentence_tokens, max_len=100):
        ids = [self.token_to_idx.get(t, self.token_to_idx[self.unk]) for t in sentence_tokens]
        if len(ids) > max_len:
            ids = ids[:max_len]
        else:
            ids += [self.token_to_idx[self.pad]] * (max_len - len(ids))
        return ids


class dataset(data.Dataset):
    def __init__(self, root, classes_path, transform=None, image_size=None, image_set='train',
                 specific_label=None, ignore_list=None, data_s=None, baseline=None, dataset_name=None):
        """
        dataset_name: 用于触发文本数据（如 AGNEWS）的词表逻辑；不区分大小写
        """
        if ignore_list is None:
            ignore_list = []

        self.root = root
        self.classes_path = classes_path
        self.transform = transform
        self.image_size = image_size
        self.image_set = image_set
        self.dataset_name = (dataset_name or "").lower()

        # 避免重复拼接一层 image_set（例如 root 已是 .../train 而 image_set='train'）
        if (not image_set) or self.root.rstrip('/').endswith(image_set):
            self.image_root = self.root
        else:
            self.image_root = os.path.join(self.root, image_set)

        # =========================
        # MTFL 分支
        # =========================
        if self.root.split('/')[-1] == 'MTFL':
            self.is_mtfl = True  # <<<<<< 加这个
            # 读取并裁剪出 40x40 灰度图，lms 为 (N,10)
            self.datalist, self.images, self.lms = self.getMTFL()

            # 默认 transform：Grayscale->Resize(40,40)->ToTensor()
            if self.transform is None:
                self.transform = T.Compose([
                    T.Grayscale(num_output_channels=1),
                    T.Resize((40, 40)),
                    T.ToTensor(),
                ])

            # 过滤 ignore_list
            delIndexList = []
            for i in range(len(self.datalist)):
                if self.datalist[i] in ignore_list:
                    delIndexList.append(i)

            for i in reversed(delIndexList):
                self.datalist = np.delete(self.datalist, i)
                self.images   = np.delete(self.images, i, axis=0)
                self.lms      = np.delete(self.lms, i, axis=0)

            print("{} data load, {} data delete".format(len(self.datalist), len(ignore_list)))
            # 文本相关属性占位
            self.vocab = None
            self.tokenizer = None

        # =========================
        # 其他数据集（MNIST/CIFAR10/AGNEWS 等）
        # =========================
        else:
            self.is_mtfl = False  # <<<<<< 以及这里
            assert data_s is not None, 'data_s must be specified'
            print("[dataset.py] image_root =", self.image_root)
            if not os.path.exists(self.image_root):
                import glob
                nearby = glob.glob(os.path.join(os.path.dirname(self.image_root), "*"))
                raise FileNotFoundError(
                    f"dataset root path does not exist:\n  {self.image_root}\n"
                    f"Parent dir listing:\n  {nearby}"
                )
            assert os.path.exists(self.classes_path), "dataset classes path does not exist!"
            with open(self.classes_path, 'r') as f:
                self.classes = json.load(f)

            class_keys = list(self.classes.keys())

            # 组装 datalist: [(relative_path, label_id), ...]
            self.datalist = []
            if specific_label is None:
                for name in class_keys:
                    img_list = [p for p in data_s.get(name, []) if p not in ignore_list]
                    self.datalist.extend(list(zip(img_list, [self.classes[name]] * len(img_list))))
            else:
                name = specific_label
                img_list = [p for p in data_s.get(name, []) if p not in ignore_list]
                self.datalist.extend(list(zip(img_list, [self.classes[name]] * len(img_list))))

            # 文本——词表逻辑（AGNEWS）
            self.vocab = None
            self.tokenizer = None
            if self.dataset_name == 'agnews':
                print("[Dataset] AGNEWS detected. Preparing vocabulary ...")
                vocab_path = "./dataset/vocab.pkl"
                self.tokenizer = get_tokenizer('basic_english')

                if os.path.exists(vocab_path):
                    print(f"[Dataset] Loading vocab from {vocab_path}")
                    with open(vocab_path, 'rb') as f:
                        self.vocab = pickle.load(f)
                else:
                    print(f"[Dataset] {vocab_path} not found. Building new vocab ...")
                    full_path_datalist = [
                        (os.path.join(self.image_root, rel_path), label)
                        for (rel_path, label) in self.datalist
                        if isinstance(rel_path, str) and rel_path.endswith('.txt')
                    ]
                    self.vocab = Vocab(full_path_datalist)
                    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
                    with open(vocab_path, 'wb') as f:
                        pickle.dump(self.vocab, f)
                    print(f"[Dataset] Vocabulary saved to {vocab_path}")

            print("{} data load, {} data delete".format(len(self.datalist), len(ignore_list)))
            print('[DEBUG][Dataset __init__] 当前datalist前3:', self.datalist[:3])

    def getMTFL(self):
        """
        读取 MTFL 的 training.txt / annotation.txt，
        - 自动兼容 Windows 反斜杠路径
        - 对 bbox 裁剪后缩放到 40x40 灰度
        - 返回: imagenames (N,), images (N,1,40,40), l (N,10)
        """
        # 规范路径
        img_root = os.path.join(self.root, self.image_set)  # 原始图根：.../MTFL/train
        datalmPath = os.path.join(img_root, f"{self.image_set}ing.txt")  # .../train/training.txt
        annotationPath = os.path.join(img_root, "annotation.txt")

        # 读 landmark 行（部分脚本在 train 有多 1 列无用数据，用两个分支兼容）
        if self.image_set == 'train':
            i, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, g, s, gl, p, _ = np.genfromtxt(
                datalmPath, delimiter=" ", unpack=True)
        else:
            i, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, g, s, gl, p = np.genfromtxt(
                datalmPath, delimiter=" ", unpack=True)

        # 第一列再单独读一次为字符串（图片相对路径），并把 '\' -> '/'
        i = np.genfromtxt(datalmPath, delimiter=" ", usecols=0, dtype=str, unpack=True)
        i = np.array(i, dtype=str)
        i = np.char.replace(i, '\\', '/')

        # 读 bbox
        bb1, bb2, bb3, bb4, bProb = np.genfromtxt(annotationPath, delimiter=" ", unpack=True)

        # 坐标缩放到 40x40
        ratio_x = 40 / (bb3 - bb1)
        ratio_y = 40 / (bb4 - bb2)
        l1, l2, l3, l4, l5 = (l1 - bb1) * ratio_x, (l2 - bb1) * ratio_x, (l3 - bb1) * ratio_x, (l4 - bb1) * ratio_x, (l5 - bb1) * ratio_x
        l6, l7, l8, l9, l10 = (l6 - bb2) * ratio_y, (l7 - bb2) * ratio_y, (l8 - bb2) * ratio_y, (l9 - bb2) * ratio_y, (l10 - bb2) * ratio_y

        # 仅保留存在的文件
        onlyfiles = [f for f in i if isfile(os.path.join(img_root, f))]
        File_length = len(onlyfiles)

        images, indexes, imagenames = [], [], []
        for n in range(File_length):
            try:
                abs_path = os.path.join(img_root, onlyfiles[n])
                temp = cv2.imread(abs_path)
                if temp is None:
                    indexes.append(n); continue
                gray = cv2.cvtColor(temp.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                x1, y1, x2, y2 = int(bb1[n]), int(bb2[n]), int(bb3[n]), int(bb4[n])
                crop_img = gray[y1:y2, x1:x2]
                if crop_img.shape[0] < 40 or crop_img.shape[1] < 40:
                    indexes.append(n); continue
                resized = cv2.resize(crop_img, (40, 40), interpolation=cv2.INTER_AREA)  # (40,40)
                resized = resized.reshape(1, 40, 40)  # (1,40,40)
                imagenames.append(onlyfiles[n])
                images.append(resized)
            except Exception:
                indexes.append(n)

        images = np.array(images)  # (N,1,40,40)

        # 同步删除无效样本的标签
        for index in reversed(indexes):
            i   = np.delete(i, index)
            l1  = np.delete(l1, index);  l2  = np.delete(l2, index);  l3  = np.delete(l3, index);  l4  = np.delete(l4, index);  l5  = np.delete(l5, index)
            l6  = np.delete(l6, index);  l7  = np.delete(l7, index);  l8  = np.delete(l8, index);  l9  = np.delete(l9, index);  l10 = np.delete(l10, index)
            g   = np.delete(g, index);   s   = np.delete(s, index);   gl  = np.delete(gl, index);   p   = np.delete(p, index)

        File_length = len(images)
        if File_length == 0:
            raise RuntimeError("[MTFL] 有效样本为 0：请检查 training.txt 路径与图像目录是否匹配（已兼容 '\\'→'/'）")

        # 统一成 (N,1) 形状后拼接成 (N,10)
        l1  = np.reshape(l1,  (-1,1)); l2  = np.reshape(l2,  (-1,1)); l3  = np.reshape(l3,  (-1,1)); l4  = np.reshape(l4,  (-1,1)); l5  = np.reshape(l5,  (-1,1))
        l6  = np.reshape(l6,  (-1,1)); l7  = np.reshape(l7,  (-1,1)); l8  = np.reshape(l8,  (-1,1)); l9  = np.reshape(l9,  (-1,1)); l10 = np.reshape(l10, (-1,1))
        l = np.concatenate([l1,l2,l3,l4,l5,l6,l7,l8,l9,l10], axis=1)  # (N,10)

        imagenames = np.array(imagenames)
        assert len(images) == len(l) == len(imagenames), "length of images and labels are not equal"
        return imagenames, images, l

    def __getitem__(self, index):
        # MTFL：返回 Tensor 图像与 float32 的 10 维关键点
        if getattr(self, "is_mtfl", False):
            img_path = self.datalist[index]      # 相对路径
            img_np   = self.images[index]        # (1,40,40) np.uint8
            label_np = self.lms[index]           # (10,) numpy

            # 转为 PIL -> Tensor（确保一定输出 Tensor）
            img = Image.fromarray(np.uint8(img_np.squeeze()))  # (40,40)
            if self.transform is not None:
                img = self.transform(img)        # (1,40,40) torch.float32 in [0,1]
            else:
                img = T.Compose([
                    T.Grayscale(num_output_channels=1),
                    T.Resize((40, 40)),
                    T.ToTensor(),
                ])(img)

            # 关键点转 float32 Tensor
            label = torch.as_tensor(label_np, dtype=torch.float32)  # (10,)

            return img, label, img_path

        # 其他数据集
        img_path_rel, label = self.datalist[index]

        # 文本样本
        if isinstance(img_path_rel, str) and img_path_rel.endswith('.txt'):
            full_path = os.path.join(self.image_root, img_path_rel)
            with open(full_path, 'r', encoding='utf-8') as f:
                text = f.read()
            if self.vocab is None or self.tokenizer is None:
                raise RuntimeError(
                    "Vocabulary/tokenizer not ready for text data. "
                    "Ensure dataset_name='AGNEWS' when constructing the dataset, "
                    "或先构建 ./dataset/vocab.pkl。"
                )
            ids = self.vocab.transform(self.tokenizer(text), max_len=100)
            img = torch.tensor(ids, dtype=torch.long)
            return img, label, img_path_rel

        # 图像样本
        full_path = os.path.join(self.image_root, img_path_rel)
        img = Image.open(full_path) # 先只打开图片，不立即转换

        # --- 关键修复：根据 image_size 动态决定转换模式 ---
        if self.image_size and len(self.image_size) == 3:
            expected_channels = self.image_size[2]
            if expected_channels == 1:
                img = img.convert('L')  # 'L' 代表单通道灰度模式
            else: # expected_channels == 3 或其他情况
                img = img.convert('RGB')
        else:
            # 如果没有提供 image_size 信息，默认转为 RGB (保持旧逻辑的兼容性)
            img = img.convert('RGB')
        # --- 修复结束 ---

        if self.image_size is not None:
            img = img.resize(self.image_size[:2])
        if self.transform is not None:
            img = self.transform(img)
        return img, label, img_path_rel

    def __len__(self):
        return len(self.datalist)


class dataset_NCNV(data.Dataset):
    # 原逻辑保留；修复文本读取编码 + 在缺少词表时报清晰错误
    def __init__(self, root, classes_path, transform=None, image_size=None, image_set='',
                 specific_label=None, ignore_list=None, data_s=None, mode=None, pred=None,
                 probability=None):
        if ignore_list is None:
            ignore_list = []
        assert data_s is not None, 'data_s must be specified'
        self.root = root
        self.classes_path = classes_path
        self.transform = transform
        self.image_size = image_size
        self.image_set = image_set
        if self.image_set == '' or self.root.endswith(self.image_set):
            self.image_root = self.root
        else:
            self.image_root = os.path.join(self.root, self.image_set)
        assert os.path.exists(self.image_root), "dataset root path does not exist!"
        assert os.path.exists(self.classes_path), "dataset classes path does not exist!"
        with open(self.classes_path, 'r') as f:
            self.classes = json.load(f)
        class_keys = list(self.classes.keys())
        if specific_label is None:
            self.datalist = []
            for name in class_keys:
                img_list = []
                for path in data_s.get(name, []):
                    if path not in ignore_list:
                        img_list.append(path)
                datas = zip(img_list, [self.classes[name]] * len(img_list))
                datas = list(datas)
                self.datalist.extend(datas)
        else:
            self.datalist = []
            name = specific_label
            img_list = []
            for path in data_s.get(name, []):
                if path not in ignore_list:
                    img_list.append(path)
            datas = zip(img_list, [self.classes[name]] * len(img_list))
            datas = list(datas)
            self.datalist.extend(datas)
        self.vocab = None
        if os.path.exists("./dataset/vocab.pkl"):
            self.vocab = pickle.load(open("./dataset/vocab.pkl", "rb"))
            self.tokenizer = get_tokenizer('basic_english')
        print("{} data load, {} data delete".format(len(self.datalist), len(ignore_list)))

        self.mode = mode
        if self.mode == "labeled":
            pred_idx = pred.nonzero()[0]
            self.probability = [probability[i] for i in pred_idx]
        elif self.mode == "unlabeled":
            pred_idx = (1 - pred).nonzero()[0]
            self.unlabeled_probability = [probability[i] for i in pred_idx]
        else:
            pred_idx = np.arange(len(self.datalist))
        self.datalist = [self.datalist[i] for i in pred_idx]

    def __getitem__(self, index):
        img_path, label = self.datalist[index]
        if isinstance(img_path, str) and img_path.endswith('.txt'):
            full_path = os.path.join(self.image_root, img_path)
            with open(full_path, 'r', encoding='utf-8') as f:
                text = f.read()
            if self.vocab is None:
                raise RuntimeError("vocab.pkl 不存在，且 dataset_NCNV 不会自动构建词表。"
                                   "请先用 dataset(dataset_name='AGNEWS') 构建一次词表。")
            img = torch.tensor(self.vocab.transform(sentence=self.tokenizer(text), max_len=200),
                               dtype=torch.long)
        else:
            full_path = os.path.join(self.image_root, img_path)
            img = Image.open(full_path)  # 先打开

            # 按 image_size 决定 1 通道还是 3 通道（和上面 dataset.__getitem__ 保持一致）
            if self.image_size and len(self.image_size) == 3:
                expected_channels = self.image_size[2]
                if expected_channels == 1:
                    img = img.convert('L')   # 灰度
                else:
                    img = img.convert('RGB')
            else:
                img = img.convert('RGB')

        if self.image_size is not None and not isinstance(img, torch.Tensor):
            img = img.resize(self.image_size[:2])

        if self.mode == 'labeled':
            prob = self.probability[index]
            if self.transform is None:
                img11, img12, img2 = img[:], img[:], img[:]
            else:
                img11, img12, img2 = self.transform(img), self.transform(img), self.transform(img)
            return img11, img12, img2, label, -1, prob, index
        elif self.mode == 'unlabeled':
            prob = self.unlabeled_probability[index]
            if self.transform is None:
                img11, img12, img2 = img[:], img[:], img[:]
            else:
                img11, img12, img2 = self.transform(img), self.transform(img), self.transform(img)
            return img11, img12, img2, label, -1, prob, index
        else:
            if self.transform is not None and not isinstance(img, torch.Tensor):
                img = self.transform(img)
            return img, label, self.datalist[index][0]

    def __len__(self):
        return len(self.datalist)


if __name__ == '__main__':
    # 仅示例，不影响主体功能
    ds = dataset(root='../dataset/mnist', classes_path='classes.json')
    transform_ = transforms.Compose([transforms.ToTensor()])
    data_loader = data.DataLoader(
        dataset(root='../dataset/mnist', classes_path='classes.json', transform=transform_),
        batch_size=1, shuffle=True, num_workers=0)
    for img_, label_ in data_loader:
        print(img_.shape, label_)
        plt.imshow(img_[0][0], cmap='gray')
        plt.show()
