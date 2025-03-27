# Standard python libraries
import os
import time
import requests

# Essential DS libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import KFold
import torch

# LightAutoML presets, task and report generation
from lightautoml.automl.presets.tabular_presets import TabularUtilizedAutoML
from lightautoml.tasks import Task
import joblib

# Load the dataset to examine the structure and features
file_path1 = 'train_X_process.csv'
file_path2 = 'train_y_process.csv'
file_path3 = 'test_X.csv'
file_path4 = 'sample_submission.csv'
train_X = pd.read_csv(file_path1)
train_y = pd.read_csv(file_path2)
train_data = pd.concat([train_X, train_y], axis=1)
# print(train_data.head())
# print(train_data.shape)
# print(train_data.head(15))
# print(train_data.tail(15))
# print(train_data.info())

test_data = pd.read_csv(file_path3)
# print(test_data.shape)
# print(test_data.head())

sub = pd.read_csv(file_path4)
# print(sub.shape)
# print(sub.head())

N_THREADS = 12   # number of vCPUs for LightAutoML model creation
N_FOLDS = 5     # number of folds in LightAutoML inner CV
TEST_SIZE = 0.2     # houldout data part size
TIMEOUT = 2 * 3600   # limit in seconds for model to train
TARGET_NAME = 'has_died'    # target column name in dataset(train_y.csv)


torch.set_num_threads(N_THREADS)

# 確保 train data 沒有洩漏
# train_ids = set(train_data['patient_id'].values)
# test_ids = set(test_data['patient_id'].values)
# print(len(train_ids.intersection(test_ids)))
#
# train_ids = set(train_data['encounter_id'].values)
# test_ids = set(test_data['encounter_id'].values)
# print(len(train_ids.intersection(test_ids)))

task = Task('binary')

roles = {
    'target': TARGET_NAME,
    'drop': ['patient_id', 'encounter_id', 'fold_id']
}

automl = TabularUtilizedAutoML(
    task = task, 
    timeout = TIMEOUT,
    cpu_limit = N_THREADS,
    tuning_params = {'max_tuning_time': 900},
    reader_params = {'n_jobs': N_THREADS, 'cv': N_FOLDS}
)

# 使用相同的 KFold 設置作為 LightAutoML
kf = KFold(n_splits=N_FOLDS, shuffle=True)

# 初始化 fold_id 資訊
train_data['fold_id'] = -1

# 分配 fold_id
for fold, (train_idx, val_idx) in enumerate(kf.split(train_data)):
    train_data.loc[val_idx, 'fold_id'] = fold

# 檢查是否成功分配
print(train_data['fold_id'].value_counts())  # 每個 fold 的樣本數量

oof_pred = automl.fit_predict(train_data, roles = roles, verbose = 2)

# automl is my LightAutoML Model
joblib.dump(automl, 'lightautoml_model.pkl')
print("Model has saved")

# use train data to find best threshold
train_pred = automl.predict(train_X)
predicted_probs = train_pred.data[:, 0]
y_true = train_y['has_died'].values
best_f1 = 0
best_threshold = 0
thresholds = np.arange(0.0, 1.01, 0.001)
for threshold in thresholds:
    # 根據閾值將機率轉為二元分類結果
    y_pred = (predicted_probs >= threshold).astype(int)
    
    # 計算 F1 Score
    f1 = f1_score(y_true, y_pred)
    
    # 如果當前 F1 分數更高，則更新最佳分數和閾值
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
print(f"Best Threshold: {best_threshold:.2f}, Best F1 Score: {best_f1:.4f}")

# 計算每個 fold 的 AUROC 和 Macro F1-Score
auroc_scores_fold = []
f1_scores_fold = []

for fold in range(N_FOLDS):
    # 取得該 fold 的資料
    val_idx = train_data[train_data['fold_id'] == fold].index
    y_true_fold = train_data.loc[val_idx, TARGET_NAME].values
    y_pred_fold = oof_pred.data[val_idx, 0]  # 預測概率

    # 計算 AUROC
    auroc = roc_auc_score(y_true_fold, y_pred_fold)
    auroc_scores_fold.append(auroc)

    # 計算 Macro F1-Score
    y_pred_class = (y_pred_fold >= best_threshold).astype(int)  # 最佳閾值
    f1_fold = f1_score(y_true_fold, y_pred_class, average='macro')
    f1_scores_fold.append(f1_fold)

    print(f"Fold {fold}: AUROC = {auroc:.4f}, Macro F1-Score = {f1_fold:.4f}")

# 計算平均分數
print(f"Average AUROC: {np.mean(auroc_scores_fold):.4f}")
print(f"Average Macro F1-Score: {np.mean(f1_scores_fold):.4f}")

print(automl.create_model_str_desc())



