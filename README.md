# ADDFL

## Description
This repository contains the implementation of **ADDFL** (dversarial Discrepancy-Based Data Fault Localization for Deep Learning Systems).

ADDFL is a data fault localization framework for deep learning datasets. It detects suspicious training instances by comparing the behavior of an original model and its adversarially trained counterpart on the same input. Instead of relying only on the natural training dynamics of a single model, ADDFL uses adversarial training as an explicit probing mechanism to amplify behavioral differences between clean and faulty samples.

The current implementation supports both single run execution and batch experiments on multiple benchmark settings. It also provides evaluation scripts for comparison with representative baselines, including **CleanLab**, **SimiFeat**, **NCNV**, **DIF**, and **DeepGini**.

## Overview
The ADDFL pipeline consists of four main stages:

1. **Adversarial probing.**  
   A clean model is loaded and adversarially fine tuned with PGD based perturbations to obtain an adversarial counterpart.

2. **Discrepancy feature extraction.**  
   For each training sample, the framework extracts features from both the original model and the adversarially trained model, including softmax outputs, loss, entropy, max confidence, and cross model discrepancy signals.

3. **Reference and proxy construction.**  
   ADDFL selects high confidence samples from each class as reference clean anchors and synthesizes proxy features from random inputs.

4. **Suspiciousness scoring and ranking.**  
   A logistic regression detector is trained to distinguish reference features from proxy features, and its output is used as the suspiciousness score for each sample. Samples are then ranked from most suspicious to least suspicious.

## Installation
Install the required dependencies first:

```bash
pip install -r requirements.txt
```

## Repository Entry Points
The main scripts in the current codebase are:

- `adversarial.py`: core ADDFL pipeline for one dataset model noise setting
- `run_all.py`: batch execution over predefined dataset model noise combinations
- `eval_dfaulo_and_baselines.py`: evaluation and comparison with baselines
- `eval_all.py`: legacy summary script for result aggregation
- `baselines.py`: implementations of CleanLab, SimiFeat, NCNV, DIF, DeepGini, and Random
- `models.py`: supported backbone models

> **Note**  
> In some local copies of this project, the script name may still appear as `adversirial.py` due to an earlier typo. If your local file keeps that name, replace `adversarial.py` with `adversirial.py` in the commands below.

## Quick Start
### Run a single ADDFL experiment
A typical command is:

```bash
python adversarial.py \
  --class_path /path/to/mnist_classes.json \
  --model_args /path/to/mnist_model_args.pth \
  --train_noisy_root /path/to/RandomLabelNoise/MNIST/train \
  --train_clean_root /path/to/OriginalTrainData/MNIST/train \
  --model_clean_path /path/to/OriginalTrainData/MNIST/LeNet5.pth \
  --test_root /path/to/OriginalTestData/MNIST/test \
  --dataset ./dataset/RandomLabelNoise/MNIST \
  --model_name LeNet5 \
  --image_size "(28, 28, 1)" \
  --image_set "" \
  --hook_layer conv \
  --rm_ratio 0.05 \
  --retrain_epoch 10 \
  --retrain_bs 64 \
  --slice_num 1 \
  --ablation None \
  --dataset_name MNIST \
  --noise_type RandomLabelNoise
```

### Run all predefined benchmark combinations
If your environment and dataset paths are already prepared, you can launch the batch experiments by running:

```bash
python run_all.py
```

The predefined batch settings currently cover the following image classification combinations:

- **Datasets:** `MNIST`, `KMNIST`, `EMNIST`, `CIFAR10`
- **Noise types:** `RandomLabelNoise`, `RandomDataNoise`, `SpecificLabelNoise`, `SpecificDataNoise`
- **Models:** `LeNet1`, `LeNet5`, `ResNet18`, `VGG`

### Evaluate ADDFL against baselines
To compare ADDFL with the supported baselines, run:

```bash
python eval_dfaulo_and_baselines.py
```

This script loads ADDFL outputs and baseline rankings, then computes ranking based localization metrics.

## Run ADDFL on a Custom Dataset
To run ADDFL on your own dataset, you need to prepare the following components.

### 1. Prepare the dataset structure
The current implementation expects class organized folders. A noisy training set may be organized as follows:

```text
YourDataset
|-- train
|   |-- class_0
|   |   |-- xxx.png
|   |   |-- yyy.png
|   |-- class_1
|   |   |-- zzz.png
|   |-- ...
|-- classes.json
```

The `classes.json` file should map class names to label indices:

```json
{
    "class_0": 0,
    "class_1": 1,
    "class_2": 2
}
```

For the full benchmark style workflow, the codebase assumes the following directory organization:

```text
ROOT_D1
|-- OriginalTrainData
|   |-- YourDataset
|   |   |-- train
|   |   |   |-- class_0
|   |   |   |-- class_1
|   |   |-- YourModel.pth
|   |   |-- model_clean.pth
|-- OriginalTestData
|   |-- YourDataset
|   |   |-- test
|   |   |   |-- class_0
|   |   |   |-- class_1
|-- RandomLabelNoise
|   |-- YourDataset
|   |   |-- train
|   |   |   |-- class_0
|   |   |   |-- class_1
|-- RandomDataNoise
|   |-- YourDataset
|   |   |-- train
|-- SpecificLabelNoise
|   |-- YourDataset
|   |   |-- train
|-- SpecificDataNoise
|   |-- YourDataset
|   |   |-- train
|-- yourdataset_classes.json
|-- yourdataset_model_args.pth
```

### 2. Prepare the clean model
ADDFL requires a pretrained clean model. In the current implementation, the clean checkpoint is passed through `--model_clean_path`.

The backbone definition should be available in `models.py`. The repository already contains implementations such as:

- `LeNet1`
- `LeNet5`
- `ResNet18`
- `VGG`

If you add a new backbone, make sure it is compatible with the current loading and forward interface used by the code.

### 3. Prepare model configuration
The `model_args.pth` file stores auxiliary information used by the pipeline, especially the transform and loss function. A typical example is:

```python
import torch
import torch.nn as nn
from torchvision import transforms

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

torch.save(
    {
        'transform': transform,
        'loss_fn': nn.CrossEntropyLoss(),
        'optimizer': 'SGD',
        'num_classes': 10,
    },
    'mnist_model_args.pth'
)
```

### 4. Run ADDFL
After preparing the noisy training set, clean model, class file, and model configuration, run `adversarial.py` with the paths corresponding to your dataset.

## Main Parameters
The key parameters in `adversarial.py` are listed below.

- `--dataset_name`: dataset name used to select default paths and model behavior
- `--noise_type`: noise setting, such as `RandomLabelNoise` or `RandomDataNoise`
- `--model_name`: model name defined in `models.py`
- `--model_clean_path`: checkpoint path of the clean model
- `--train_clean_root`: root path of the clean training set
- `--train_noisy_root`: root path of the noisy training set to be inspected
- `--test_root`: root path of the test set
- `--dataset`: output root where ADDFL saves features, models, rankings, and timing files
- `--class_path`: class to index mapping file
- `--image_size`: model input size in `(H, W, C)` format
- `--model_args`: serialized model configuration file containing transform and loss function
- `--image_set`: image subset indicator used by the dataset loader
- `--hook_layer`: representation layer name, currently kept for compatibility with the pipeline
- `--rm_ratio`: retained compatibility parameter from earlier experimental settings
- `--retrain_epoch`: retained compatibility parameter for fine tuning related settings
- `--retrain_bs`: batch size used by several training and baseline procedures
- `--slice_num`: number of data slices used by the loader wrapper
- `--ablation`: ablation flag, default is `None`
- `--seed`: random seed for reproducibility

## Output Files
ADDFL writes its outputs under the directory specified by `--dataset`, usually in the following structure:

```text
./dataset/<NoiseType>/<DatasetName>
|-- feature
|   |-- <ModelName>
|   |   |-- full_Feature.json
|   |   |-- Adversarial_full_Feature.json
|   |   |-- image_list.json
|   |   |-- gt_list.json
|   |   |-- gt_one_hot_list.json
|   |   |-- random_index.json
|-- mutmodel
|   |-- <ModelName>
|   |   |-- model_clean.pth
|   |   |-- model_adv.pth
|-- results
|   |-- <ModelName>
|   |   |-- Adversarial_results_list.json
|   |   |-- Adversarial_sorted_score_list.json
|   |   |-- adversarial_time.json
|   |   |-- CleanLab_results_list.json
|   |   |-- CleanLab_sorted_score_list.json
|   |   |-- SimiFeat_results_list.json
|   |   |-- SimiFeat_sorted_score_list.json
|   |   |-- NCNV_results_list.json
|   |   |-- NCNV_sorted_score_list.json
|   |   |-- DIF_results_list.json
|   |   |-- DIF_sorted_score_list.json
|   |   |-- DeepGini_results_list.json
|   |   |-- DeepGini_sorted_score_list.json
```

Among them:

- `Adversarial_results_list.json` stores the final ranked sample list
- `Adversarial_sorted_score_list.json` stores the ranked suspiciousness scores
- `adversarial_time.json` stores stage level timing information
- `random_index.json` stores the selected high confidence clean anchors
- `model_adv.pth` is the adversarially trained counterpart of the clean model

## Feature Design
The current ADDFL implementation constructs each sample feature from:

- original model softmax output
- one hot observed label
- adversarial model softmax output
- original and adversarial losses
- original and adversarial entropy values
- original and adversarial max confidence values
- discrepancy terms between the two models, including:
  - softmax L2 distance
  - loss difference
  - entropy difference
  - max confidence difference
  - prediction consistency indicator

These features are then fed into a logistic regression detector to obtain suspiciousness scores.

## Evaluation Metrics
The evaluation scripts support the following ranking based metrics:

- `PoBL@10%`
- `RAUC`
- `APFD`
- `AUC-ROC`

The main comparison script is `eval_dfaulo_and_baselines.py`, which aligns ADDFL and baseline outputs into a unified ranking based evaluation process.

## Supported Baselines
The repository currently includes implementations of the following baselines:

- `Dfaulo`
- `CleanLab`
- `SimiFeat`
- `NCNV`
- `DIF`
- `DeepGini`
- `Random`

These baselines are implemented in `eval_dfaulo_and_baselines.py` and can be evaluated together with ADDFL using the provided scripts.

## Notes
1. The current repository still contains some legacy file names and script logic inherited from earlier versions. In particular, `eval_all.py` is closer to an older result summary workflow, while `eval_dfaulo_and_baselines.py` is the more direct script for comparing ADDFL with baselines.
2. The batch scripts assume a fixed dataset root on disk. If you run the code on a different machine, you should first modify the hard coded paths in the scripts.
3. Some helper arguments are kept for compatibility with the existing experimental pipeline even if they are not the core variables of ADDFL itself.

## Citation
If you use this repository in academic work, please cite your ADDFL paper accordingly.
