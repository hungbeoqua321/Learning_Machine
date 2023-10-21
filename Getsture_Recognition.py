# Import các thư viện cần thiết
import cv2  # Thư viện OpenCV cho xử lý ảnh và video
import mediapipe as mp  # Thư viện MediaPipe cho việc nhận dạng điểm mốc trên tay
import os  # Thư viện hệ thống để làm việc với tệp và thư mục
import numpy as np  # Thư viện NumPy cho xử lý dữ liệu số
from sklearn.svm import SVC  # Thư viện scikit-learn cho học máy và phân loại
from sklearn.model_selection import train_test_split  # Để chia dữ liệu thành tập huấn luyện và tập kiểm tra
from sklearn.metrics import accuracy_score  # Để đo độ chính xác của mô hình

# Khởi tạo đối tượng Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
# static_image_mode: ché độ ảnh tĩnh
# max_num_hand: nhận diện tối đa bao nhiêu tay
# min_detection_confidence: Đây là ngưỡng tối thiểu cho độ tin cậy của việc nhận dạng các điểm mốc trên tay. 
# Chỉ những điểm mốc có độ tin cậy lớn hơn hoặc bằng ngưỡng này sẽ được xem xét.

# Hàm để đọc dữ liệu từ các tệp và thư mục con
def load_data_from_folders(data_folder):
    gestures = []  # Danh sách chứa dữ liệu tay
    labels = []    # Danh sách chứa nhãn tương ứng

    # Duyệt qua tất cả thư mục trong thư mục dữ liệu
    for folder_name in os.listdir(data_folder):
        folder_path = os.path.join(data_folder, folder_name)
        if os.path.isdir(folder_path):
            # Duyệt qua tất cả tệp tin văn bản trong thư mục con
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.txt'):
                    #Hàm os.path.join sẽ nối hai thành phần này lại với nhau 
                    # và tạo ra một đường dẫn đầy đủ đến tệp tin file_name 
                    # bên trong thư mục folder_path. 
                    # Kết quả sẽ là một chuỗi đại diện cho đường dẫn tới tệp tin đó.
                    file_path = os.path.join(folder_path, file_name)
                    with open(file_path, 'r') as file:
                        data = []
                        # Duyệt qua từng dòng trong tệp và chuyển thành mảng dữ liệu
                        for line in file:
                            x, y, z = map(float, line.strip().split())
                            data.extend([x, y, z])
                        gestures.append(data)  # Thêm dữ liệu vào danh sách tay
                        labels.append(folder_name)  # Thêm nhãn vào danh sách

    return np.array(gestures), np.array(labels)

# Đường dẫn đến thư mục chứa dữ liệu
data_folder = 'hand_shape_data'

# Đọc dữ liệu từ thư mục và các thư mục con
gestures, labels = load_data_from_folders(data_folder)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra Perceptron 
# test_size=0.2 có nghĩa là 20% của dữ liệu sẽ được sử dụng cho tập kiểm tra và 80% còn lại sẽ được sử dụng cho tập huấn luyện.
# 42 là Lucky number
X_train, X_test, y_train, y_test = train_test_split(gestures, labels, test_size=0.2, random_state=42)

# Huấn luyện mô hình SVM với kernel tuyến tính với C là tham số vùng an toàn
# Đây là biến thể 2
model = SVC(kernel='linear', C=1)
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Đánh giá độ chính xác của mô hình bằng cách sử dụng
# SVC (Support Vector Classifier) với kernel tuyến tính (linear kernel). 
# SVM với kernel tuyến tính thường được sử dụng trong các tác vụ phân loại tuyến tính, 
# nơi dữ liệu có thể được phân chia thành hai lớp bằng một đường thẳng (hoặc siêu phẳng) tốt nhất.
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Khởi tạo webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển đổi khung hình thành ảnh màu RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Nhận diện các điểm mốc trên tay
    results = hands.process(frame_rgb)

    # Vẽ các điểm mốc của các ngón tay và dự đoán cử chỉ tay
    if results.multi_hand_landmarks:
        # i là index, nếu xóa thì han_landmarks sẽ trở thành index 
        # hand_landmarks là từng điểm trên bàn tay
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Vẽ các điểm mốc trên tay
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
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
