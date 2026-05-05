import json
import os

# ==================== 配置(parameter configuration) ====================
# 输出目录
# Output path
OUTPUT_DIR = './eval/recovery_time_ratio'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 数据准备(data preparation) ====================
# 参考例子(Reference example)
time_data = {
    "CIFAR10_1": {
        "Retrain": 1972.01,
        "FT": 204.40,
        "NegGrad+": 189.41,
        "Random-label": 91.92,
        "SalUn": 47.26,
        "SCRUB": 314.26,
        "UNSIR": 76.93,
        "SSD": 28.93,
        "BiasShift": 0.018,
        "SF": 69.40,
        "TS-BGM": 10.17,
        "TS-BGRM": 11.37,
        "LB-HR": 40.14
    },
    "CIFAR10_3": {
        "Retrain": 1890.25,
        "FT": 50.77,
        "NegGrad+": 85.45,
        "Random-label": 68.05,
        "SalUn": 49.37,
        "SCRUB": 267.79,
        "UNSIR": 82.09,
        "SSD": 30.16,
        "BiasShift": 0.016,
        "SF": 50.40,
        "TS-BGM": 9.51,
        "TS-BGRM": 9.57,
        "LB-HR": 37.22
    }
}

# ==================== 计算 RTR(calculating RTR)====================
rtr_results = {}

for dataset_key, methods in time_data.items():
    rtr_results[dataset_key] = {}
    retrain_time = methods["Retrain"]

    for method, unlearn_time in methods.items():
        if method == "Retrain":
            continue  # 基准方法不计算RTR(The benchmark method does not calculate RTR)

        if retrain_time > 0:
            rtr = (unlearn_time / retrain_time) * 100
            rtr_results[dataset_key][method] = round(rtr, 10)  # 保留4位小数(Keep to four decimal places)
        else:
            rtr_results[dataset_key][method] = None

# ==================== 保存结果(save results) ====================
output_file = os.path.join(OUTPUT_DIR, 'recovery_time_ratio.json')

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(rtr_results, f, indent=4, ensure_ascii=False)

print(f"RTR calculation completed, and the result has been saved to {output_file}")

# 打印前几个结果供预览(Print the first few results for preview)
print(json.dumps(rtr_results["CIFAR10_1"], indent=4))