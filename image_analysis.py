from PIL import Image
import mediapipe as mp
import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_image_with_mediapipe(image_path):
    # Khởi tạo công cụ Mediapipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Đọc ảnh từ đường dẫn
    image = Image.open(image_path)

    # Chuyển đổi ảnh thành numpy array
    image_np = np.array(image)

    # Chuyển đổi ảnh sang định dạng RGB
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    # Nhận diện các điểm trên cơ thể
    results = pose.process(image_rgb)

    # Trích xuất thông tin về các điểm trên cơ thể
    keypoints = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            keypoints.append((landmark.x, landmark.y, landmark.z if landmark.HasField('z') else 0))

    return keypoints