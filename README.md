# Beyond the Shadow of Bias: From Classification Head Bias to Parameter Redistribution

> Official code for the paper *"Classification-Head Bias in Class-Level Machine Unlearning: Diagnosis, Mitigation, and Evaluation"*

---

## Language / 语言

- [English](README.md)
- [简体中文](README.zh-CN.md)
- [繁體中文](README.zh-TW.md)

---

## How to Reproduce Our Experiments

Our experiments are built upon the codebase of Yichen et al. ([CMF_Unlearning](https://github.com/ycgao1/CMF_Unlearning)). Their code is well-structured and includes many baseline methods, allowing us to focus on our contributions. To avoid copyright disputes, this code repository only provides the core code that is necessary beyond the code contributed by Yichen and others. Below we provide a step-by-step guide to reproduce our results.

### Step 1: Create Conda Environment

```bash
conda create -n bias_unlearn python=3.9.18
conda activate bias_unlearn
```

### Step 2: Download Yichen et al.'s Code

```bash
git clone https://github.com/ycgao1/CMF_Unlearning.git
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

You may try to reproduce Yichen et al.'s results first. Our results are thoroughly presented in the paper, and we guarantee that these results are **reliable**!

### Step 4: Integrate Our Code

1. Carefully read the comments in `config/config.py` and copy the latter part of the code into Yichen et al.'s `config.py` as instructed.
2. Copy the files under the `Unlearn` folder into their `unlearn` folder.
3. Locate the `run_unlearning` function in Yichen et al.'s `main.py`. Call our unlearning functions via conditional branches. Our functions are fully compatible with this interface, so no changes to input/output signatures are needed.
4. After completing the above steps, you may encounter an error indicating that the model does not have the `history_log` attribute. To maximize code generality, we only provide the core unlearning code without logging functions. Do not worry — we will show you how to add this attribute. This attribute is already implemented in Yichen et al.'s code. Simply add the following code in the unlearning function under the `unlearn` folder to compute the metrics and write them into `history_log`. You will also need to pass in missing arguments such as `test_loader`, which are readily available in the `run_unlearning` function.

```python
unlearn_model.eval()
train_retain_acc, train_forget_acc, train_metric = test(
    unlearn_model, device, train_loader, args.unlearn_class,
    args.class_label_names, args.num_classes,
    job_name=args.unlearn_method, set_name="Final Train Set"
)

test_retain_acc, test_forget_acc, test_metric = test(
    unlearn_model, device, test_loader,
    args.unlearn_class, args.class_label_names, args.num_classes,
    job_name=args.unlearn_method, set_name="Test Set"
)

unlearn_model.history_log = {
    'unlearn_time': unlearn_time,
    'test_retain_acc': test_retain_acc,
    'test_forget_acc': test_forget_acc,
    'test_metric': test_metric,
    'train_retain_acc': train_retain_acc,
    'train_forget_acc': train_forget_acc,
    'train_metric': train_metric,
    'args': args,
}
```

5. It is worth noting that Yichen et al.'s code does not include a function for measuring unlearning time. You can add it yourself using the code below. However, you must add this measurement inside the unlearning function under the `unlearn` folder, not in `run_unlearning`. The unlearning functions in the `unlearn` folder already include timing evaluation.

```python
import time
start_time = time.time()
# unlearning operations
unlearn_time = time.time() - start_time
```

### Experimental Setup

| System | Specification               | Software | Version     |
| ------ | --------------------------- | -------- | ----------- |
| OS     | Ubuntu 20.04.6 LTS          | Python   | 3.9.15      |
| CPU    | Intel(R) Xeon(R) Gold 6240R | Torch  | 2.6.0 |
| GPU    | V100S-PCIE-32GB             | CUDA     | 12.8        |

**Table 7: Learning rates of core methods.**

| Method   | Retain | Forget |
| -------- | ------ | ------ |
| Original | 0.1    | -      |
| Retrain  | 0.1    | -      |
| SF       | 0.001  | 0.01   |
| TS-BGM   | 0.001  | 0.01   |
| TS-BGRM  | 0.001  | 0.01   |
| LB-HR    | 0.1    | -      |



**Table 8: Number of epochs used by core methods.**

| Method   | CIFAR10 |      | CIFAR100 |      | Tiny-Imagenet |      |
| -------- | ------- | ---- | -------- | ---- | ------------- | ---- |
|          | 1       | 3    | 1        | 3    | 1             | 3    |
| Original | 150     | 150  | 150      | 150  | 150           | 150   |
| Retrain  | 150     | 150  | 150      | 150  | 150             | 150    |
| SF       | 8       | 8    | 20       | 20   | 10            | 10   |
| TS-BGM   | 1       | 1    | 2        | 12   | 1             | 4    |
| TS-BGRM  | 1       | 1    | 2        | 12   | 4             | 1    |
| LB-HR    | 5       | 6    | 10       | 10   | 5             | 5    |



With the above steps, we believe you can successfully reproduce our experiments!

---

## Paper Introduction

**Beyond the Shadow of Bias: From Classification Head Bias to Parameter Redistribution**

**Abstract:** As deep learning models are trained on massive datasets, they inevitably learn low-quality, outdated, or private information, which may degrade performance or raise security concerns. This paper focuses on class-level machine unlearning. We systematically analyze how gradient-based unlearning updates affect classification head bias and demonstrate that existing methods heavily rely on the bias term of the forget class. Based on this observation, we propose BiasShift, which confines unlearning to the bias term and achieves highly efficient forgetting under conventional metrics. However, BiasShift still suffers from bias dependence and potential privacy risks. Therefore, we shift the unlearning effect from bias to classification head weights and feature extractor layers, proposing TS-BGRM, LB-HR, and FGLU frameworks. We also introduce BSC, MBS, and MBG metrics to quantify bias dependence in unlearning.

---


