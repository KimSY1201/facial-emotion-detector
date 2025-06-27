import os
import cv2
import numpy as np
import pandas as pd
import utils

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import matplotlib.pyplot as plt
import seaborn as sns

#  CSV 로드
df = pd.read_csv(r"C:\Users\KDT-37\Desktop\KDT_7\09_ML_CV\Project\보라_crop_std_output.csv")

df['H_avg'] = pd.to_numeric(df['H_avg'], errors='coerce')

# 2. 결측치 제거
df = df.dropna(subset=['H_avg'])

#  색상 분류 라벨 생성
df['color_group'] = df['H_avg'].apply(utils.classify_color_group_from_h)

#  피처 / 타겟 정의
features = ['R_avg', 'G_avg', 'B_avg', 'H_avg', 'S_avg', 'V_avg',
            'R_std', 'G_std', 'B_std', 'H_std', 'S_std', 'V_std']
X = df[features]
y = df['color_group']

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(y)

#  train/test 분할
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

#  SVM용 정규화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#  여러 모델 정의

models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "LightGBM": LGBMClassifier(random_state=42)
}

#  Confusion Matrix 시각화 함수
def plot_confusion(model_name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

#  모델 훈련 및 평가
for name, model in models.items():
    print(f"\n Model: {name}")
    if name == "SVM":
        model.fit(X_train_scaled, y_train)
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

    print("Train Classification Report")
    print(classification_report(y_train, y_pred_train))
    print("Test Classification Report")
    print(classification_report(y_test, y_pred_test))
    
    plot_confusion(name, y_test, y_pred_test)
    