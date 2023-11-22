import math
import joblib
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mediapipe as mp
from image_analysis import process_image_with_mediapipe
from encode_keypoints import get_index_by_position

data_folder = 'data'

def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2 + (point2[2] - point1[2])**2)

def load_data_from_folder(folder_path):
    
    
    for i in range(21):
        folder_path = os.path.join(data_folder, str(i))
        
        # Pull data from file
        json_file_path = os.path.join(folder_path,'measurements.json')
        with open(json_file_path, 'r') as j_file:
            data = json.load(j_file)
        
        # Front image analysis
        front_img_path = os.path.join(folder_path, 'front_img.jpg')
        keypoints = process_image_with_mediapipe(front_img_path)
        
        # calculate ratio
        Height_pixel = calculate_distance(keypoints[get_index_by_position('Left Eye')],
                                             keypoints[get_index_by_position('Right Heel')])
        Height = data['Height']
        ratio = Height / Height_pixel
        
load_data_from_folder(data_folder)




# Lưu model
# joblib.dump('model.pkl')
        
        
        
        