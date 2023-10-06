import cv2
import mediapipe as mp
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Khởi tạo Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Hàm đọc dữ liệu từ các tệp và thư mục con
def load_data_from_folders(data_folder):
    gestures = []
    labels = []

    for folder_name in os.listdir(data_folder):
        folder_path = os.path.join(data_folder, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.txt'):
                    file_path = os.path.join(folder_path, file_name)
                    with open(file_path, 'r') as file:
                        data = []
                        for line in file:
                            x, y, z = map(float, line.strip().split())
                            data.extend([x, y, z])
                        gestures.append(data)
                        labels.append(folder_name)

    return np.array(gestures), np.array(labels)

# Đường dẫn đến thư mục chứa dữ liệu
data_folder = 'hand_shape_data'

# Đọc dữ liệu từ thư mục và các thư mục con
gestures, labels = load_data_from_folders(data_folder)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(gestures, labels, test_size=0.2, random_state=42)

# Huấn luyện mô hình SVM
model = SVC(kernel='linear', C=1)
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Đánh giá độ chính xác
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Khởi tạo webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển đổi thành ảnh màu RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Nhận diện các điểm mốc trên tay
    results = hands.process(frame_rgb)

    # Vẽ các điểm mốc của các ngón tay
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Chuyển dữ liệu điểm mốc thành mảng numpy
            data = []
            for landmark in hand_landmarks.landmark:
                data.extend([landmark.x, landmark.y, landmark.z])
            data = np.array(data).reshape(1, -1)

            # Dự đoán cử chỉ tay
            predicted_gesture = model.predict(data)[0]

            # Hiển thị kết quả dự đoán lên khung hình
            cv2.putText(frame, f"Gesture: {predicted_gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
