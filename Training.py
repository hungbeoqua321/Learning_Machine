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
from encode_keypoints import get_index_by_position

def preprocess_data(data):
    df = pd.DataFrame(data, index=[0])
    for name in df:
        df[name] = df[name].replace("_tbr", '').astype(float)

    label_encoder = LabelEncoder()
    df['gender'] = label_encoder.fit_transform(df['gender'])
    df['race'] = label_encoder.fit_transform(df['race'])
    df['profession'] = label_encoder.fit_transform(df['profession'])
    
    return df

def update_and_retrain(new_data, model_path):
    # Load existing model
    existing_model = joblib.load(model_path)
    
    
    
    X_train, X_test, y_train, y_test = train_test_split()
    
    # Retrain the model with updated data
    existing_model.fit(X_updated, y_updated)
    
    # Save the updated model
    joblib.dump(existing_model, model_path)
data_folder = 'data'

def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2 + (point2[2] - point1[2])**2)

def load_data_from_folder(folder_path):
    coordinates = []
    datas = []
    
    for i in range(21):
                
        folder_path = os.path.join(data_folder, str(i))
        
        # Pull data from file
        json_file_path = os.path.join(folder_path,'measurements.json')
        with open(json_file_path, 'r') as j_file:
            data = json.load(j_file)
            preprocess_data(data)
            datas.append(data)
        
        # Front image analysis
        front_img_path = os.path.join(folder_path, 'front_img.jpg')
        keypoints = process_image_with_mediapipe(front_img_path)
        
        coordinates.append(keypoints)
        
        # calculate ratio
        # Height_pixel = calculate_distance(keypoints[get_index_by_position('Left Eye')],
        #                                      keypoints[get_index_by_position('Right Heel')])
        # Height = float(data['height'])
        # ratio = Height / Height_pixel
    
    
load_data_from_folder(data_folder)


    
        




# Lưu model
# joblib.dump('model.pkl')
        
        
        
        