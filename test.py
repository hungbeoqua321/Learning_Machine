import cv2
import mediapipe as mp

# Đường dẫn đến ảnh
image_path = 'data\\0\\side_img.jpg'

# Khởi tạo đối tượng Mediapipe
mp_pose = mp.solutions.pose

# Khởi tạo Pose model
pose = mp_pose.Pose()

# Đọc ảnh từ đường dẫn
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Xử lý ảnh để lấy kết quả
results = pose.process(image_rgb)

# Tăng độ sáng của ảnh
image_brightened = cv2.convertScaleAbs(image, alpha=1.2, beta=30)

# Làm rõ cạnh của ảnh và chuyển về ảnh grayscale
image_edges = cv2.Canny(cv2.cvtColor(image_brightened, cv2.COLOR_BGR2GRAY), 50, 150)

# Resize ảnh làm rõ cạnh để có cùng kích thước với ảnh gốc
image_edges = cv2.resize(image_edges, (image.shape[1], image.shape[0]))

# Kết hợp ảnh gốc với ảnh làm rõ cạnh
image_combined = cv2.addWeighted(image_brightened, 0.7, cv2.cvtColor(image_edges, cv2.COLOR_GRAY2BGR), 0.3, 0)

# Vẽ các điểm và số thứ tự lên ảnh
if results.pose_landmarks:
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = image_combined.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image_combined, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        cv2.putText(image_combined, f'{id}', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# Lấy kích thước cửa sổ hiển thị
screen_width, screen_height = 400, 800  # Thay đổi kích thước theo ý muốn

# Resize ảnh để vừa với cửa sổ
image_combined = cv2.resize(image_combined, (screen_width, screen_height))

# Hiển thị ảnh
cv2.imshow('Enhanced Pose Detection', image_combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
