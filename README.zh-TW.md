# Beyond the Shadow of Bias: From Classification Head Bias to Parameter Redistribution

> 論文 *《Beyond the Shadow of Bias: From Classification Head Bias to Parameter Redistribution》* 官方代碼

---

## 語言 / Language

- [English](README.md)
- [简体中文](README.zh-CN.md)
- [繁體中文](README.zh-TW.md)

---

## 如何複現我們的論文實驗

我們的實驗基於 Yichen 等人的代碼庫 ([CMF_Unlearning](https://github.com/ycgao1/CMF_Unlearning))。他們的代碼非常完善，包含許多對比實驗，我們直接在上面進行更改來完成我們的實驗。本代碼倉庫提供了 Yichen 等人代碼之外所需的核心代碼。接下來我們會手把手地告訴你如何複現我們的論文。

### 第一步：創建 conda 環境

```bash
conda create -n bias_unlearn python=3.9.18
conda activate bias_unlearn
```

### 第二步：下載 Yichen 等人的代碼

```bash
git clone https://github.com/ycgao1/CMF_Unlearning.git
```

### 第三步：安裝依賴包

```bash
pip install -r requirements.txt
```

你可以嘗試去複現Yichen等人的代碼，我們跑出來的結果在論文中有非常詳細的呈現，並且可以保證這些結果是**真實可靠**的！

### 第四步：嵌入我們的代碼

1. 仔細閱讀 `config/config.py` 的註釋，按照註釋要求把後半部分的代碼複製到 Yichen 等人的 `config.py` 中。
2. 將 `Unlearn` 文件夾下的文件複製到 Yichen 等人的 `unlearn` 文件夾下。
3. 在 Yichen 等人的 `main.py` 中找到 `run_unlearning` 函數，通過選擇分支結構調用我們的遺忘方法函數，無需改變輸入輸出接口。
4. 在完成上述操作後，你還可能會遇到一個錯誤，即model沒有`history_log`屬性，為了增加代碼的通用性，我們提供的代碼僅僅為基礎的遺忘代碼，沒有相關記錄函數，但不用擔心，我們會告訴你如何加入這個屬性，這個屬性在Yichen等人的代碼中已經寫好了，你只需要在`unlearn`文件下找到相應的遺忘函數，通過下面的代碼進行測試後寫入`history_log`中，然後傳入`test_loader`等缺失的參數，這些在`run_unlearning`函數中都是有的，所以改起來很容易。

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

5. 值得注意的是的是，Yichen等人的文件中並沒有測試遺忘時間的函數，你自行加入可以通過下面的代碼進行測試，但你需要在`unlearn`文件夾下的函數進行添加，而不能在`run_unlearning`函數裡測量，`unlearn`文件夾下的遺忘函數包含了評估時間。

```python
import time
start_time = time.time()
# 遺忘操作
unlearn_time = time.time() - start_time
```

### 實驗設備及參數設置

| 系統     | 具體信息                    | 軟體    | 版本        |
| -------- | --------------------------- | ------- | ----------- |
| 系統環境 | Ubuntu 20.04.6 LTS          | Python  | 3.9.15      |
| CPU      | Intel(R) Xeon(R) Gold 6240R | PyTorch | 2.5.0.post0 |
| GPU      | V100S-PCIE-32GB             | CUDA    | 12.8        |

### 超參數設置

| 方法     | 保留集學習率 | 遺忘集學習率 |
| -------- | ------------ | ------------ |
| Original | 0.1          | -            |
| Retrain  | 0.1          | -            |
| SF       | 0.001        | 0.01         |
| TS-BGM   | 0.001        | 0.01         |
| TS-BGRM  | 0.001        | 0.01         |
| LB-HR    | 0.1          | -            |

**表7：核心方法的學習率。**

| Method   | Retain | Forget |
| -------- | ------ | ------ |
| Original | 0.1    | -      |
| Retrain  | 0.1    | -      |
| SF       | 0.001  | 0.01   |
| TS-BGM   | 0.001  | 0.01   |
| TS-BGRM  | 0.001  | 0.01   |
| LB-HR    | 0.1    | -      |

**表8：核心方法使用的epoch數。**

| Method   | CIFAR10 |      | CIFAR100 |      | Tiny-Imagenet |      |
| -------- | ------- | ---- | -------- | ---- | ------------- | ---- |
|          | 1       | 3    | 1        | 3    | 1             | 3    |
| Original | 150     | 150  | 150      | 150  | 150           | 10   |
| Retrain  | 150     | 150  | 150      | 150  | 5             | 5    |
| SF       | 8       | 8    | 20       | 20   | 10            | 10   |
| TS-BGM   | 1       | 1    | 2        | 12   | 1             | 4    |
| TS-BGRM  | 1       | 1    | 2        | 12   | 4             | 1    |
| LB-HR    | 5       | 6    | 10       | 10   | 5             | 5    |



至此，我們相信你能夠很順利地複現我們的實驗！

---

## 論文介紹

**Beyond the Shadow of Bias: From Classification Head Bias to Parameter Redistribution**

**摘要：** 隨著人工智能模型的發展，模型在海量數據的訓練下極易學習到一些低質的、過時的或者隱私數據，從而影響模型性能或者帶來一些安全隱患。本文聚焦於類別級遺忘，全面且係統地分析了基於梯度的遺忘更新對模型分類頭偏置的影響，並通過實驗證明了現有大部分遺忘方法過度依賴遺忘類對應分類頭的偏置項。基於此，我們將對遺忘發揮作用的參數局限於遺忘類的分類頭偏置，提出了在傳統評價指標上實現最高效遺忘的 BiasShift 方法，但該方法與現有大部分方法一樣高度依賴分類頭偏置，可能會存在一些隱私問題。因此，我們將對遺忘發揮作用的參數由分類頭偏置轉移到分類頭權重、特徵提取層參數，提出了 TS-BGRM、LB-HR 和 FGLU 等遺忘框架，並提出了 BSC、MBS、MBG 等評價指標來衡量遺忘對分類頭偏置的依賴程度。

---

## 一個潛在的研究點

在論文中，我們提出的 FGLU 遺忘框架並沒有進行實驗驗證。我們歡迎廣大研究者通過實驗去驗證我們的框架！

---

