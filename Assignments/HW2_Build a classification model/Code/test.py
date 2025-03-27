# Standard python libraries
import os
import time
import requests

# Essential DS libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
import joblib
from sklearn.preprocessing import LabelEncoder

# load LightAutoML Model
automl = joblib.load('lightautoml_model.pkl')
print("Successful")

# Load the dataset to examine the structure and features
file_path1 = 'train_X.csv'
file_path2 = 'train_y.csv'
file_path3 = 'test_X.csv'
file_path4 = 'sample_submission.csv'
train_X = pd.read_csv(file_path1)
test_data = pd.read_csv(file_path3)
sub = pd.read_csv(file_path4)

# 1. 初步選擇類別型特徵和數值型特徵
categorical_features = list(train_X.select_dtypes(include='object').columns)
numeric_features = list(train_X.select_dtypes(include=['int64', 'float64']).columns)

# 2. 基於唯一值數量篩選出 0 到 6 的特徵，並將其視為類別型特徵
other_categorical_features = [col for col in numeric_features if train_X[col].nunique() <= 6]

# 3. 更新類別型和數值型特徵列表
categorical_features.extend(other_categorical_features)  # 將誤認為數值型特徵加入類別型特徵
numeric_features = [col for col in numeric_features if col not in other_categorical_features]  # 剔除屬於類別的特徵

# ------------------ LabelEncoder() -------------------

for col in categorical_features:
    le = LabelEncoder()
    test_data[col] = le.fit_transform(test_data[col].apply(str))


test_pred = automl.predict(test_data)
print(f'Prediction for test_data:\n{test_pred}\nShape = {test_pred.shape}')
plt.hist(test_pred.data, bins=20)
plt.show()

# Fast feature importances calculation
fast_fi = automl.get_feature_scores('fast')

# fast_fi saved as 'fast_feature_importances.csv'
fast_fi.to_csv('fast_feature_importances.csv', index=False)
print('to csv is Successful')
fast_fi.set_index('Feature')['Importance'].plot.bar(figsize = (30, 20), grid = True)
plt.savefig('fast_feature_importances.png')
plt.show()

# 將 test_pred 機率值轉換為 0 或 1, 當機率值 >= threshold, 則轉換為 1
sub['pred'] = (test_pred.data[:, 0] >= 0.16).astype(int)
# submission saved as 'testing_result.csv'
sub.to_csv('testing_result.csv', index=False)