import math
import joblib
import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from image_analysis import *
from encode_keypoints import get_index_by_name

# Init variable
model = LinearRegression()
    
data_folder = 'data'

model_path = 'model.joblib'

def preprocess_data(data):
    
    for name in data:
        data[name] = data[name].replace("_tbr", '')
        
    df = pd.DataFrame(data, index=[0])
    return df

def update_and_retrain(predict_data,real_data , model_path):
    # Load existing model
    existing_model = joblib.load(model_path)

    X_train, X_test, y_train, y_test = train_test_split(predict_data,real_data,test_size=.2,random_state=42)
    # Retrain the model with updated data
    existing_model.fit(X_train, y_train)
    
    # Save the updated model
    joblib.dump(existing_model, model_path)

def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2 + (point2[2] - point1[2])**2)

def load_data_from_folder(folder_path):
    coordinates = []
    
    for i in range(21):
                
        folder_path = os.path.join(data_folder, str(i))
        
        # Pull data from file
        json_file_path = os.path.join(folder_path,'measurements.json')
        with open(json_file_path, 'r') as j_file:
            real_data = json.load(j_file)
            preprocess_data(real_data)
        # Front image analysis
        front_img_path = os.path.join(folder_path, 'front_img.jpg')
        keypoints = process_image_with_mediapipe(front_img_path)
        
        coordinates.append(keypoints)

        
        # calculate ratio
        Height_pixel = calculate_distance(keypoints[get_index_by_name('Left Eye')],
                                             keypoints[get_index_by_name('Left Heel')])
        Height = float(real_data['height'])
        ratio = Height / Height_pixel
        
        predict_data = {
            'arm_length_cm' : ratio * (calculate_distance(keypoints[get_index_by_name('Left Shoulder')],
                                                        keypoints[get_index_by_name('Left Elbow')]) + 
                                       calculate_distance(keypoints[get_index_by_name('Left Elbow')],
                                                        keypoints[get_index_by_name('Left Wrist')]))
        }
        
        print(predict_data)
        
        
  
load_data_from_folder(data_folder)

# Lưu model
# joblib.dump('model.pkl')
        
        
        
        