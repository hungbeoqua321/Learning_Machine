import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


data, true_labels = make_blobs(n_samples=300, centers=3, cluster_std=2.0, random_state=42)


scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

def GS(data, k_max=15, n=10):
    gap_values = []
    for k in range(1, k_max + 1):
        gap_k = []
        for _ in range(n):
            random_data = np.random.rand(*data.shape)
            kmeans_random = KMeans(n_clusters=k)
            kmeans_random.fit(random_data)
            gap_k.append(kmeans_random.inertia_)
        
        gap_k = np.mean(np.log(gap_k))
        gap_values.append(gap_k)
    
    return gap_values

k_values = range(1, 16)
gap_values = GS(data_scaled)

plt.figure(figsize=(10, 6))
plt.plot(k_values, gap_values, marker='o', linestyle='-')
plt.xlabel('Số lượng cụm (k)')
plt.ylabel('Gap Statistic')
plt.title('Gap Statistic để chọn số cụm tối ưu')
plt.grid(True)
plt.show()
