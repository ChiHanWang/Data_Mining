# Standard python libraries
import os
import time
import requests

# Essential DS libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from scipy import stats 


# Load the dataset to examine the structure and features
file_path1 = 'train_X.csv'
file_path2 = 'train_y.csv'
file_path3 = 'test_X.csv'
file_path4 = 'sample_submission.csv'
train_X = pd.read_csv(file_path1)
train_y = pd.read_csv(file_path2)

# 1. 初步選擇類別型特徵和數值型特徵
categorical_features = list(train_X.select_dtypes(include='object').columns)
numeric_features = list(train_X.select_dtypes(include=['int64', 'float64']).columns)

# 2. 基於唯一值數量篩選出 0 到 6 的特徵，並將其視為類別型特徵
other_categorical_features = [col for col in numeric_features if train_X[col].nunique() <= 6]

# 3. 更新類別型和數值型特徵列表
categorical_features.extend(other_categorical_features)  # 將誤認為數值型特徵加入類別型特徵
numeric_features = [col for col in numeric_features if col not in other_categorical_features]  # 剔除屬於類別的特徵

# 4. print out
print("\nCategorical Features:", len(categorical_features))
print("\nNumeric Features:", len(numeric_features))

# ------------------ Data Analysis ----------------

# ------------------ 數據摘要統計 ------------------
# 查看數據基本信息
print("Dataset Information:")
print(train_X.info())

# 查看統計摘要
print("\nStatistical Summary (Numerical Features):")
print(train_X.describe())

# 類別型特徵頻率分布
print("\nValue Counts for Categorical Features:")
for col in train_X.select_dtypes(include='object').columns:
    print(f"\nFeature: {col}")
    print(train_X[col].value_counts())

# ------------------ Data Visualization ------------------

# Histogram and boxplot of numerical features
for feature in numeric_features:
    # Histogram
    plt.figure(figsize=(8, 6))
    train_X[feature].hist(bins=20, color='steelblue', edgecolor='black')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.grid(False)
    plt.savefig(f'{feature}_histogram.png')
    # plt.show()

    # Boxplot
    plt.figure(figsize=(8, 6))
    train_X.boxplot(column=[feature], vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    plt.title(f'Boxplot of {feature}')
    plt.xlabel(feature)
    plt.savefig(f'{feature}_boxplot.png')
    # plt.show()
    plt.close('all')

# Bar plot of categorical features
for feature in categorical_features:
    plt.figure(figsize=(12, 16))
    train_X[feature].value_counts().plot(kind='bar', color='lightcoral', edgecolor='black')
    plt.title(f'Bar Plot of {feature}')
    plt.xlabel('Category')
    plt.ylabel('Frequency')
    plt.savefig(f'{feature}_barplot.png')
    # plt.show()
    plt.close('all')

# calculate the distribution of has_died
value_counts = train_y['has_died'].value_counts()

# draw Pie Chart
plt.figure(figsize=(6, 6))
plt.pie(
    value_counts, 
    labels=value_counts.index, 
    autopct='%1.1f%%',  # 顯示比例到小數點一位
    startangle=90,      # 起始角度
    colors=['skyblue', 'orange'],  # 自定義的顏色
    explode=(0.05, 0)   # 分離 0 的部分，使其更醒目
)

# add title
plt.title('Proportion of has_died Values')
plt.show()

# ------------------ Data Cleaning -------------------

# ------------------ 統計每行的缺失值 ------------------

# 1. calculate missing value counts and ratio of each row
row_missing_counts = train_X.isnull().sum(axis=1)  # missing value counts of each row
row_missing_ratio = train_X.isnull().mean(axis=1)  # missing value ratio of each row

# add 2 new columns to train_X
train_X['Missing Count'] = row_missing_counts
train_X['Missing Ratio'] = row_missing_ratio

# ------------------ 查看缺失值統計 ------------------

# print 前幾行的統計結果
print("Row-wise Missing Values:")
print(train_X[['Missing Count', 'Missing Ratio']].head())

# ------------------ 篩選缺失值過多的行 ------------------

# set missing ratio threshold
threshold = 0.5
rows_to_drop = train_X[train_X['Missing Ratio'] > threshold]

print(f"\nRows with > {threshold*100}% missing values:")
print(rows_to_drop)

# drop the rows of missing value ratio > 50%
train_X_cleaned = train_X[train_X['Missing Ratio'] <= threshold]
train_y_cleaned = train_y[train_X['Missing Ratio'] <= threshold]

print(f"\nRemaining rows after cleaning: {len(train_X_cleaned)}")
print(f"\nRemaining rows after cleaning: {len(train_y_cleaned)}")

train_X_cleaned = train_X_cleaned.drop(columns=['Missing Count', 'Missing Ratio'])
# print(train_data.head())

# # ------------------ Data Imputation -------------------

# 1. numerical features imputation
imputer_numeric = SimpleImputer(strategy='median')  # 使用中位數填補
train_X_imputed = train_X_cleaned
train_X_imputed[numeric_features] = imputer_numeric.fit_transform(train_X_imputed[numeric_features])

# 2. categorical features imputation
train_X_imputed[categorical_features] = train_X_imputed[categorical_features].fillna("Missing")

# ------------------ 確認處理後的數據 ------------------
print("\nRemaining missing values (should be 0):")
print(train_X_imputed.isnull().sum())

# ------------------ Data Transformation -------------------
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    train_X_imputed[col] = le.fit_transform(train_X_imputed[col].apply(str))
    label_encoders[col] = le

# ------------------ Data Imbalance -------------------

# use SMOTE for oversampling to minority class
smote = SMOTE(sampling_strategy=0.2)
X_resampled, y_resampled = smote.fit_resample(train_X_imputed, train_y_cleaned)

# 檢查重新採樣後的數據分布
print("Original dataset class distribution:")
print(train_y_cleaned['has_died'].value_counts())
print("\nResampled dataset class distribution:")
print(pd.DataFrame(y_resampled, columns=['has_died'])['has_died'].value_counts())


X_resampled.to_csv('train_X_process.csv', index=False)
y_resampled.to_csv('train_y_process.csv', index=False)

