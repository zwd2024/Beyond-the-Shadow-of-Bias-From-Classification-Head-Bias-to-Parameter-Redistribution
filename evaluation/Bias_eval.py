import json
import numpy as np
import os
from pathlib import Path

'''
使用方法见readme文件
Usage instructions can be found in the readme.
'''
# ==================== 参数配置(parameter configuration) ====================
class Config:
    # JSON 数据路径 (JSON data path)
    JSON_FILE = './bias_show/bias_data.json'

    # 数据集 (Dataset name)
    DATASET = 'CIFAR10'  # CIFAR10、CIFAR100、Tiny-Imagenet

    # 遗忘类别数量 (The number of forgotten categories)
    FORGET_CLASS_COUNT = 3

    # 遗忘类别索引 (Forgotten category index)
    FORGET_CLASS_INDICES = [3, 4, 5]

    # 输出目录 (Output path)
    OUTPUT_DIR = f'./{DATASET}_{FORGET_CLASS_INDICES}'

    # 输出文件名 (Output file name)
    METRICS_FILE = 'bias_metrics.json'


# ===================================================


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_json_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File does not exist: {file_path}")

    with open(file_path, 'r', encoding='utf-8-sig') as f:
        return json.load(f)


def compute_metrics_for_method(biases, forget_indices):
    """
    计算单个方法的 BSC / MBG / MBS
    Calculate the BSC/MBG/MBS of a single method
    """
    biases = np.asarray(biases)

    V = np.array(forget_indices)
    R = np.array([i for i in range(len(biases)) if i not in forget_indices])

    b_V = biases[V]
    b_R = biases[R]

    # BSC
    bsc = 1.0 / (1.0 + abs(np.mean(b_V) - np.mean(b_R)))

    # MBG
    mbg = sigmoid(np.median(b_V) - np.min(b_R))

    # MBS
    mbs = sigmoid(np.min(b_V) - np.min(b_R))

    return {
        "BSC": float(bsc),
        "MBG": float(mbg),
        "MBS": float(mbs)
    }


def compute_all_metrics(data, dataset, forget_count, forget_indices):
    key = f"{dataset}_{forget_count}"
    if key not in data:
        raise KeyError(f"Key '{key}' not found in data")

    results = {}
    for method, biases in data[key].items():
        results[method] = compute_metrics_for_method(biases, forget_indices)

    return {key: results}


def save_metrics_to_single_json(metrics, output_dir, filename):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    save_path = os.path.join(output_dir, filename)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print(f"The indicator has been saved to{save_path}")


def main():
    print("Data loading...")
    data = load_json_data(Config.JSON_FILE)

    print("Calculating BSC / MBG / MBS...")
    metrics = compute_all_metrics(
        data=data,
        dataset=Config.DATASET,
        forget_count=Config.FORGET_CLASS_COUNT,
        forget_indices=Config.FORGET_CLASS_INDICES
    )

    print("Save as a single JSON file...")
    save_metrics_to_single_json(
        metrics=metrics,
        output_dir=Config.OUTPUT_DIR,
        filename=Config.METRICS_FILE
    )

    print("\nComplete！")


if __name__ == "__main__":
    main()
