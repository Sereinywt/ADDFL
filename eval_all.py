# eval_all.py  —— 仅评 noManual，并把运行总时长 runtime_s（可选各阶段时长）写入表格
import os, json, csv, re
import numpy as np
from sklearn.metrics import roc_auc_score
from datetime import datetime
from utils.evaluation import APFD, POBL, RAUC, ROC_AUC

def _norm_fname(s: str) -> str:
    """把各种名字规范成 '12345.png'：
       - '5/moved_9_000021__54235.png' -> '54235.png'
       - 'moved_7_000031__42175.png'  -> '42175.png'
       - '30001.png'                  -> '30001.png'
    """
    base = os.path.basename(s)
    m = re.search(r'__([0-9]+\.png)$', base, flags=re.IGNORECASE)
    return m.group(1) if m else base

def _norm_name2fault(n2f_raw: dict) -> dict:
    """把原始 name2isfault（键可能是带前缀路径/文件名）归一化到 '12345.png'"""
    n2f = {}
    for k, v in n2f_raw.items():
        key = _norm_fname(k)
        val = 1 if (v is True or str(v).lower() in ("1", "true")) else 0
        n2f[key] = val
    return n2f

# ===== 你要评测的组合 =====
NOISE_TYPES = ['RandomLabelNoise','RandomDataNoise','SpecificDataNoise','SpecificLabelNoise']
DATASETS    = ['MNIST', 'CIFAR10','EMNIST','KMNIST','AGNEWS']
MODELS      = ['LeNet1','LeNet5', 'WaveMix','ResNet','ResNet18', 'VGG', 'LSTM', 'BiLSTM']

# 只保留 noManual 的文件名
NOMAN_LIST  = 'noManual_results_list.json'
NOMAN_SCORE = 'noManual_sorted_score_list.json'

# 是否把各阶段时间也写进表格（Select Subset / Mutation&Extraction / Initialize Susp / Update Susp）
WRITE_STAGE_TIMES = False   # 想要列出来就改成 True

def find_name2isfault(noise, dataset):
    main = os.path.join('dataset',  noise, dataset, 'train', 'name2isfault.json')
    if os.path.exists(main): return main
    fb   = os.path.join('dataset1', noise, dataset, 'train', 'name2isfault.json')
    if os.path.exists(fb):   return fb
    return None

def load_json(path):
    with open(path,'r') as f:
        return json.load(f)

def safe_load(path):
    return load_json(path) if os.path.exists(path) else None

def extract_labels_and_scores(score_list, name2isfault, base_dir=None):
    """
    支持：
      1) [['xxx.png', score], ...]
      2) ['xxx.png', ...]                 -> 用排名当分
      3) [0.9, 0.8, ...] (纯分数数组)     -> 需要 base_dir 旁的 noManual_results_list.json 提供文件名
    """
    import os, json
    import numpy as np

    if not score_list:
        return np.array([]), np.array([])

    # 情形1: 成对
    if isinstance(score_list[0], list) and len(score_list[0]) == 2:
        filenames, scores = zip(*score_list)
        norm_names = [_norm_fname(x) for x in filenames]                 # ★ 关键：规范化
        labels = [name2isfault.get(n, 0) for n in norm_names]
        return np.array(labels, int), np.array(scores, float)

    # 情形2: 纯字符串列表 -> 用排名
    if isinstance(score_list[0], str):
        filenames = score_list
        scores = list(range(len(filenames), 0, -1))
        norm_names = [_norm_fname(x) for x in filenames]                 # ★ 关键：规范化
        labels = [name2isfault.get(n, 0) for n in norm_names]
        return np.array(labels, int), np.array(scores, float)

    # 情形3: 纯分数数组 -> 从 noManual_results_list.json 拿文件名
    if isinstance(score_list[0], (int, float)):
        if base_dir is None:
            raise ValueError("pure score list requires base_dir to locate noManual_results_list.json")
        noman_path = os.path.join(base_dir, 'noManual_results_list.json')
        if not os.path.exists(noman_path):
            raise FileNotFoundError(f"缺少 {noman_path}，无法为分数配对文件名")
        filenames = json.load(open(noman_path))
        if len(filenames) != len(score_list):
            raise ValueError("len(filenames) != len(scores)")
        norm_names = [_norm_fname(x) for x in filenames]                 # ★ 关键：规范化
        labels = [name2isfault.get(n, 0) for n in norm_names]
        return np.array(labels, int), np.array(score_list, float)

    raise TypeError("unknown score_list format")

def calc_auc_roc_standalone(base_dir, n2f):
    """
    一个独立的 AUC-ROC 计算器，它不依赖于“统一的格式”。
    
    它会智能地解析 noManual_sorted_score_list.json 文件，
    无论其格式是：
      1) [['xxx.png', score], ...]  (成对格式)
      2) ['xxx.png', ...]         (纯文件名，用排名当分数)
      3) [0.9, 0.8, ...] (纯分数，依赖 _results_list.json)
      
    它使用的评估方法 (roc_auc_score) 与 evaluation.py 一致。
    """
    
    # 1. 优先尝试加载 _sorted_score_list.json
    score_file_path = os.path.join(base_dir, NOMAN_SCORE)
    score_content = safe_load(score_file_path)
    
    # 2. 如果失败，回退到 _results_list.json (用于格式 2)
    if not score_content:
        score_file_path = os.path.join(base_dir, NOMAN_LIST)
        score_content = safe_load(score_file_path)
        
    # 3. 如果两个文件都不存在
    if not score_content:
        print("  [WARN] AUC-ROC: 找不到 _sorted_score_list 或 _results_list 文件。")
        return float('nan')

    try:
        # 4. 使用 eval_all (2).py 自带的健壮解析器
        labels, scores = extract_labels_and_scores(score_content, n2f, base_dir)
        
        if len(labels) == 0:
            print("  [WARN] AUC-ROC: 解析后得到 0 个样本。")
            return float('nan')

        # 5. 检查标签是否只有一类（这会导致 sklearn 失败）
        #
        if len(set(labels.tolist())) < 2:
            print("  [WARN] AUC-ROC: 真实标签中只找到一种类别（例如，全是'有故障'或全是'无故障'）。")
            return float('nan')
            
        # 6. 计算指标 (与 evaluation.py 的方法一致)
        return float(roc_auc_score(labels, scores))
        
    except Exception as e:
        print(f"  [ERROR] Standalone AUC-ROC 计算失败: {e}")
        return float('nan')

# 这是修改后的新版本
def load_time_info(base_dir):
    """
    读取时间：优先 adversarial_time.json，其次 dfaulo_time.json
    返回 (runtime_s, stage_times_dict)
    """
    cand = [
        os.path.join(base_dir, 'adversarial_time.json'),
        os.path.join(base_dir, 'dfaulo_time.json'),
    ]
    tj = None
    for p in cand:
        if os.path.exists(p):
            try:
                tj = load_json(p)
                break
            except Exception:
                pass

    runtime_s = None
    stages = {}
    if tj:
        # 1. 优先尝试读取预先计算好的总时长
        for k in ['all', 'ALL', 'total', 'Total', 'total_seconds']:
            if k in tj:
                try:
                    runtime_s = float(tj[k])
                except Exception:
                    pass
                break
        
        # 2. 收集所有已知的分阶段时长
        #    扩充了已知阶段的key，以匹配 adversarial.py 的输出
        known_stage_keys = [
            'Select Subset', 'Mutation&Extraction', 'Initialize Susp', 'Update Susp', 
            'Adv_Training', 'Feature_Extraction'
        ]
        for k in known_stage_keys:
            if k in tj:
                try:
                    stages[k] = float(tj[k])
                except (ValueError, TypeError):
                    stages[k] = 0.0 # 如果值不是数字，则记为0

        # ### 新增逻辑开始 ###
        # 3. 如果没有找到预设的总时长，并且收集到了分阶段时长，则自动求和
        if runtime_s is None and stages:
            runtime_s = sum(stages.values())
        # ### 新增逻辑结束 ###

    return runtime_s, stages

def eval_one_experiment(base_dir, name2isfault_path):
    """base_dir = ./dataset/<noise>/<dataset>/results/<model>"""
    n2f = load_json(name2isfault_path) # 这是 name2isfault 字典

    # 1. 加载排序后的文件名列表 (用于 APFD, RAUC, POBL)
    results_list_path = os.path.join(base_dir, NOMAN_LIST)
    results_list = safe_load(results_list_path)

    # 初始化指标
    pobl_score = float('nan')
    rauc_score = float('nan')
    apfd_score = float('nan')
    positives = 0
    count = 0

    # 2. 如果 results_list 存在，则计算 APFD, RAUC, POBL
    #    (这些指标依赖于 evaluation.py 中的函数)
    if results_list:
        try:
            # 检查是否有 '有故障' 的样本
            y_true_check = [n2f.get(os.path.basename(x), 0) for x in results_list]
            positives = int(np.sum(y_true_check))
            count = len(results_list)
            
            if positives > 0:
                pobl_score = POBL(results_list, n2f, 0.1)  #
                rauc_score = RAUC(results_list, n2f)      #
                apfd_score = APFD(results_list, n2f)      #
            else:
                print("  [WARN] APFD/RAUC: 真实标签中没有'有故障'的样本。")

        except Exception as e:
            print(f"  [ERROR] APFD/RAUC/POBL 计算失败: {e}")
            
    else:
        print("  [WARN] APFD/RAUC/POBL: 找不到 noManual_results_list.json。")


    # 3. 单独计算 AUC-ROC
    #    (使用我们新加的、不依赖格式的函数)
    roc_auc_score = calc_auc_roc_standalone(base_dir, n2f)


    # 4. 汇总结果
    res_no = {
        'count':     count,
        'positives': positives,
        'PoBL@10%':  round(pobl_score, 6),
        'RAUC':      round(rauc_score, 6),
        'APFD':      round(apfd_score, 6),
        'AUC-ROC':   round(roc_auc_score, 6) # <-- 这个值来自新函数
    }

    # 5. 读取时间信息 (这部分逻辑保持不变)
    runtime_s, stage_times = load_time_info(base_dir)
    return res_no, runtime_s, stage_times

def main():
    rows = []
    print('==== 批量评测开始（仅 noManual）====')

    # 打印顺序：先按 model，再按 noise，再按 dataset
    ALLOWED_COMBINATIONS = {
        "MNIST": ["LeNet1", "LeNet5"],
        "KMNIST": ['ResNet18', "LeNet5"],
        "CIFAR10": ["ResNet18", "VGG"],
        "EMNIST":["LeNet5","ResNet18"],
    }

    for ds in DATASETS:
        for model in MODELS:
            for noise in NOISE_TYPES:
                # 👇 新增过滤逻辑 👇
                if model not in ALLOWED_COMBINATIONS.get(ds, []):
                    continue
                name2isfault_path = find_name2isfault(noise, ds)
                if not name2isfault_path:
                    print(f'[SKIP] 缺少 name2isfault.json: dataset={ds}, noise={noise}')
                    continue
                base_dir = os.path.join('dataset', noise, ds, 'results', model)
                if not os.path.isdir(base_dir):
                    print(f'[SKIP] 结果目录不存在：{base_dir}')
                    continue
                try:
                    res_no, runtime_s, stage_times = eval_one_experiment(base_dir, name2isfault_path)
                except Exception as e:
                    print(f'[ERROR] 计算失败: {base_dir} -> {e}')
                    continue

                print(f'\n=== [{noise} | {ds} | {model}] ===')
                print('🔍 noManual（非人工迭代结果）：')
                print(f"  PoBL@10%: {res_no['PoBL@10%']}")
                print(f"  RAUC:     {res_no['RAUC']}")
                print(f"  AUC-ROC:  {res_no['AUC-ROC']}")
                print(f"  APFD:     {res_no['APFD']}")
                if runtime_s is not None:
                    print(f"  runtime_s: {runtime_s:.3f}s")

                row = {
                    'noise_type': noise,
                    'dataset': ds,
                    'model': model,
                    'mode': 'noManual',
                    **res_no,
                    'runtime_s': runtime_s if runtime_s is not None else '',
                }
                if WRITE_STAGE_TIMES:
                    row['t_select_subset']   = stage_times.get('Select Subset', '')
                    row['t_mut_extract']     = stage_times.get('Mutation&Extraction', '')
                    row['t_init_susp']       = stage_times.get('Initialize Susp', '')
                    row['t_update_susp']     = stage_times.get('Update Susp', '')
                rows.append(row)

    if not rows:
        print('\n⚠️ 未找到任何可评测的实验结果。')
        return

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = 'eval_reports'
    os.makedirs(out_dir, exist_ok=True)
    csv_path  = os.path.join(out_dir, f'eval_summary_{ts}.csv')
    json_path = os.path.join(out_dir, f'eval_summary_{ts}.json')

    # 保存 CSV + JSON
    fieldnames = ['noise_type','dataset','model','mode','count','positives','PoBL@10%','RAUC','AUC-ROC','APFD','runtime_s']
    if WRITE_STAGE_TIMES:
        fieldnames += ['t_select_subset','t_mut_extract','t_init_susp','t_update_susp']

    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(json_path, 'w') as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f'\n✅ 汇总完成（只包含 noManual）：\n  CSV : {csv_path}\n  JSON: {json_path}')

if __name__ == '__main__':
    main()
