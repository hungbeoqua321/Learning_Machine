import pandas as pd
import math

# Đọc dữ liệu từ DataFrame
data = pd.read_csv('D:\DATA for LM\DATA_Play_or_Not.csv')

# Hàm tính entropy
def calculate_entropy(data, target_column):
    total_samples = len(data)
    entropy = 0

    for target_value in data[target_column].unique():
        count = len(data[data[target_column] == target_value])
        probability = count / total_samples
        entropy -= probability * math.log2(probability)

    return entropy

# Hàm chọn thuộc tính tốt nhất dựa trên Information Gain
def select_best_attribute(data, target_column):
    initial_entropy = calculate_entropy(data, target_column)
    best_attribute = None
    best_information_gain = 0

    for attribute in data.columns:
        if attribute == target_column:
            continue

        attribute_entropy = 0
        total_samples = len(data)

        for attribute_value in data[attribute].unique():
            attribute_subset = data[data[attribute] == attribute_value]
            attribute_entropy += len(attribute_subset) / total_samples * calculate_entropy(attribute_subset, target_column)

        information_gain = initial_entropy - attribute_entropy

        if information_gain > best_information_gain:
            best_information_gain = information_gain
            best_attribute = attribute

    return best_attribute

# Chọn thuộc tính tốt nhất để chia
best_attribute = select_best_attribute(data, 'Label ')

print(f"Tên thuộc tính chia tốt nhất: {best_attribute}")
