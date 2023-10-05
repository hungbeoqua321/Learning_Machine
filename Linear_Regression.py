import numpy as np
import matplotlib.pyplot as plt

# Tạo dữ liệu mẫu
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.rand(100, 1)

# Visualize dữ liệu
plt.scatter(X, y, label="Dữ liệu mẫu")
plt.xlabel("Biến độc lập")
plt.ylabel("Biến phụ thuộc")
plt.legend()
plt.show()

# Thực hiện hồi quy tuyến tính
from sklearn.linear_model import LinearRegression

# Khởi tạo mô hình hồi quy tuyến tính
model = LinearRegression()

# Huấn luyện mô hình trên dữ liệu
model.fit(X, y)

# Lấy các tham số của mô hình
slope = model.coef_[0][0]
intercept = model.intercept_[0]

# Vẽ đường hồi quy tuyến tính
plt.scatter(X, y, label="Dữ liệu mẫu")
plt.plot(X, slope * X + intercept, color='red', label="Đường hồi quy tuyến tính")
plt.xlabel("Biến độc lập")
plt.ylabel("Biến phụ thuộc")
plt.legend()
plt.show()

# In ra các tham số của mô hình
print(f"Hệ số góc (slope): {slope}")
print(f"Hệ số góc (intercept): {intercept}")
