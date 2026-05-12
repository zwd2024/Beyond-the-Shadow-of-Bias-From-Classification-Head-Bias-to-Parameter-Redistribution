# Beyond the Shadow of Bias: From Classification Head Bias to Parameter Redistribution

> 论文 *《Classification-Head Bias in Class-Level Machine Unlearning: Diagnosis, Mitigation, and Evaluation》* 的官方代码

---

## 语言 / Language

- [English](README.md)
- [简体中文](README.zh-CN.md)
- [繁體中文](README.zh-TW.md)

---

## 如何复现我们的论文实验

我们的实验基于 Yichen 等人的代码库 ([CMF_Unlearning](https://github.com/ycgao1/CMF_Unlearning))。他们的代码非常完善，包含许多对比实验，我们直接在上面进行更改来完成我们的实验。为了避免版权争议，本代码仓库仅提供了 Yichen 等人代码之外所需的核心代码。。接下来我们会手把手地告诉你如何复现我们的论文。

### 第一步：创建 conda 环境

```markdown
conda create -n bias_unlearn python=3.9.18
conda activate bias_unlearn
```

### 第二步：下载 Yichen 等人的代码

```bash
git clone https://github.com/ycgao1/CMF_Unlearning.git
```

### 第三步：安装依赖包

```bash
pip install -r requirements.txt
```

你可以尝试去复现Yichen等人的代码，我们跑出来的结果在论文中有非常详细的呈现，并且可以保证这些结果是**真实可靠**的！

### 第四步：嵌入我们的代码

1. 仔细阅读 `config/config.py` 的注释，按照注释要求把后半部分的代码复制到 Yichen 等人的 `config.py` 中。
2. 将 `Unlearn` 文件夹下的文件复制到 Yichen 等人的 `unlearn` 文件夹下。
3. 在 Yichen 等人的 `main.py` 中找到 `run_unlearning` 函数，通过选择分支结构调用我们的遗忘方法函数，无需改变输入输出接口。
4. 在完成上述操作后，你还会遇到一个错误，即model没有`history_log`属性，为了增加代码的通用性，我们提供的代码仅仅为基础的遗忘代码，没有相关记录函数，但不用担心，我们会告诉你如何加入这个属性，这个属性在Yichen等人的代码中已经写好了，你只需要在`unlearn`文件下找到相应的遗忘函数，通过下面的代码进行测试后写入`history_log`中，然后传入`test_loader`等缺失的参数，这些在`run_unlearning`函数中都是有的，所以改起来很容易。

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

5. 值得注意的是的是，Yichen等人的文件中并没有测试遗忘时间的函数，你自行加入可以通过下面的代码进行测试，但你需要在`unlearn`文件夹下的函数进行添加，而不能在`run_unlearning`函数里测量，`unlearn`文件夹下的遗忘函数包含了评估时间。

```python
import time
start_time = time.time()
# 遗忘操作
unlearn_time = time.time() - start_time
```

### 实验设备及参数设置

| 系统     | 具体信息                    | 软件    | 版本        |
| -------- | --------------------------- | ------- | ----------- |
| 系统环境 | Ubuntu 20.04.6 LTS          | Python  | 3.9.15      |
| CPU      | Intel(R) Xeon(R) Gold 6240R | Torch | 2.6.0 |
| GPU      | V100S-PCIE-32GB             | CUDA    | 12.8        |

**表7：核心方法的学习率**

| Method   | Retain | Forget |
| -------- | ------ | ------ |
| Original | 0.1    | -      |
| Retrain  | 0.1    | -      |
| SF       | 0.001  | 0.01   |
| TS-BGM   | 0.001  | 0.01   |
| TS-BGRM  | 0.001  | 0.01   |
| LB-HR    | 0.1    | -      |



**表8：核心方法使用epoch数**

<img width="1128" height="265" alt="image" src="https://github.com/user-attachments/assets/15ed52ff-c31d-4582-a034-0c5c21c34281" />

至此，我们相信你能够很顺利地复现我们的实验！

---

## 论文介绍

**Beyond the Shadow of Bias: From Classification Head Bias to Parameter Redistribution**

**摘要：** 随着人工智能模型的发展，模型在海量数据的训练下极易学习到一些低质的、过时的或者隐私数据，从而影响模型性能或者带来一些安全隐患。本文聚焦于类别级遗忘，全面且系统地分析了基于梯度的遗忘更新对模型分类头偏置的影响，并通过实验证明了现有大部分遗忘方法过度依赖遗忘类对应分类头的偏置项。基于此，我们将对遗忘发挥作用的参数局限于遗忘类的分类头偏置，提出了在传统评价指标上实现最高效遗忘的 BiasShift 方法，但该方法与现有大部分方法一样高度依赖分类头偏置，可能会存在一些隐私问题。因此，我们将对遗忘发挥作用的参数由分类头偏置转移到分类头权重、特征提取层参数，提出了 TS-BGRM、LB-HR 和 FGLU 等遗忘框架，并提出了 BSC、MBS、MBG 等评价指标来衡量遗忘对分类头偏置的依赖程度。

---


