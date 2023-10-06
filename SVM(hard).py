import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import tkinter as tk
from tkinter import Entry, Button
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Dữ liệu huấn luyện
X = np.array([[-3, 1], [-2, 1], [-1, -1], [0, -1], [1, -1], [2, 1], [3, 1]])
y = np.array([1, 1, -1, -1, -1, 1, 1])

# Khởi tạo mô hình SVM với kernel tuyến tính (hard margin)
clf = svm.SVC(kernel='linear', C=1.0)

# Huấn luyện mô hình
clf.fit(X, y)

# Trích xuất các tham số w và b từ mô hình
w = clf.coef_[0]
b = clf.intercept_

# Tạo dữ liệu để vẽ đường thẳng ban đầu
slope = -w[0] / w[1]
intercept = -b / w[1]
xx = np.linspace(-3, 3)
yy = slope * xx + intercept

# Tạo cửa sổ GUI
root = tk.Tk()
root.title("SVM Classifier")

# Tạo canvas để vẽ biểu đồ trên GUI
canvas = plt.figure(figsize=(5, 4), dpi=100)
canvas_widget = FigureCanvasTkAgg(canvas, master=root)
canvas_widget.get_tk_widget().pack()

# Hàm để vẽ lại đường thẳng dựa trên tọa độ điểm mới
def draw_new_line():
    x_new = float(entry_x.get())
    y_new = float(entry_y.get())
    X_new = np.append(X, [[x_new, y_new]], axis=0)
    y_new = np.append(y, 1)  # Assume the new point is classified as '1'

    # Khởi tạo mô hình SVM với kernel tuyến tính (hard margin)
    clf_new = svm.SVC(kernel='linear', C=1.0)
    clf_new.fit(X_new, y_new)

    # Trích xuất các tham số w và b từ mô hình mới
    w_new = clf_new.coef_[0]
    b_new = clf_new.intercept_

    # Tạo dữ liệu để vẽ đường thẳng mới
    slope_new = -w_new[0] / w_new[1]
    intercept_new = -b_new / w_new[1]
    xx_new = np.linspace(-3, 3)
    yy_new = slope_new * xx_new + intercept_new

    # Xóa biểu đồ hiện tại và vẽ lại với đường thẳng mới
    plt.clf()
    plt.scatter(X_new[:, 0], X_new[:, 1], c=y_new, cmap=plt.cm.Paired)
    plt.plot(xx_new, yy_new, 'k-')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('SVM Classifier Thể Loại Hard Margin')
    canvas_widget.draw()  # Sử dụng canvas_widget.draw() thay cho canvas.draw()

# Tạo ô nhập tọa độ điểm mới
label_x = tk.Label(root, text="X1:")
label_x.pack()
entry_x = Entry(root)
entry_x.pack()
label_y = tk.Label(root, text="X2:")
label_y.pack()
entry_y = Entry(root)
entry_y.pack()

# Tạo nút để vẽ lại đường thẳng
draw_button = Button(root, text="Vẽ Lại Đường Thẳng", command=draw_new_line)
draw_button.pack()

root.mainloop()
