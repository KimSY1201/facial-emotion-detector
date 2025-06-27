from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import joblib
import pandas as pd
import utils
from sklearn.preprocessing import LabelEncoder
import cv2

#  CSV 로드
df = pd.read_csv(r"C:\Users\KDT-37\Desktop\KDT_7\09_ML_CV\Project\보라_crop_std_output.csv")

#  색상 분류 라벨 생성
df['color_group'] = df['H_avg'].apply(utils.classify_color_group_from_h)

#  피처 / 타겟 정의
features = ['R_avg', 'G_avg', 'B_avg', 'H_avg', 'S_avg', 'V_avg',
            'R_std', 'G_std', 'B_std', 'H_std', 'S_std', 'V_std']
X = df[features]
y = df['color_group']

le = LabelEncoder()
y_encoded = le.fit_transform(y)


# 1-1. 의사결정트리
dt_model = DecisionTreeClassifier(random_state=42)
dt_scores = cross_val_score(dt_model, X, y, cv=5)
print(f" 의사결정트리 평균 정확도 (5-fold): {dt_scores.mean():.2f}")

dt_model.fit(X, y)
# joblib.dump({
#     'model': dt_model,
#     'features': features  # 정확한 컬럼 순서 리스트
# }, 'decision_tree_model.pkl')

print("학습 시 사용된 feature 컬럼:")
print(X.columns.tolist())

# 1-2. XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_scores = cross_val_score(xgb_model, X, y_encoded, cv=5)
print(f" XGBoost 평균 정확도 (5-fold): {xgb_scores.mean():.2f}")

xgb_model.fit(X, y_encoded)
# joblib.dump({
#     'model': xgb_model,
#     'features': features  # 정확한 컬럼 순서 리스트
# }, 'xgboost_model.pkl')

print("학습 시 사용된 feature 컬럼:")
print(X.columns.tolist())

print("모델 저장 완료 (decision_tree_model.pkl, xgboost_model.pkl)")
