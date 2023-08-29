# 导入必要的库
import numpy as np


def calculate_metrics(true_positives, false_positives, false_negatives):
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * precision * recall / (precision + recall)

    return precision, recall, f1_score


# 用于模拟的测试数据，0表示未检测到目标，1表示检测到目标
ground_truth = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 1])
predictions = np.array([1, 0, 0, 1, 1, 0, 1, 1, 0, 1])

# 统计真正例、假正例和假负例
true_positives = np.sum(np.logical_and(predictions == 1, ground_truth == 1))
false_positives = np.sum(np.logical_and(predictions == 1, ground_truth == 0))
false_negatives = np.sum(np.logical_and(predictions == 0, ground_truth == 1))

# 计算指标
precision, recall, f1_score = calculate_metrics(true_positives, false_positives, false_negatives)

# 打印结果
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
