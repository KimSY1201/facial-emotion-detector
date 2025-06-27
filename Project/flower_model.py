from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import utils

from sklearn.metrics import accuracy_score

# 데이터 로드
df = pd.read_csv(r"C:\Users\KDT-37\Desktop\KDT_7\09_ML_CV\Project\보라_crop_std_output.csv")

df['label'] = df['filename'].str.split("_").str[0]

print(df[['filename', 'label']].head())

# 평균 H를 기준으로 색상 계열 분류
df['color_group'] = df['H_avg'].apply(utils.classify_color_group_from_h)


# 특성과 타겟 분리
X = df[['R_avg', 'G_avg', 'B_avg', 'H_avg', 'S_avg', 'V_avg', 'R_std', 'G_std', 'B_std', 'H_std', 'S_std', 'V_std']]
y = df['color_group']

# 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.fit_transform(y_test)

# SVM을 위한 정규화 (다른 모델은 원본 그대로 사용)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 사용할 모델들 정의
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(random_state=42)
}

# 모델별 훈련 및 평가
for name, model in models.items():
    if name == "SVM":
        model.fit(X_train_scaled, y_train_encoded)
        train_score = model.score(X_train_scaled, y_train_encoded)
        test_score = model.score(X_test_scaled, y_test_encoded)
    else:
        model.fit(X_train, y_train_encoded)
        train_score = model.score(X_train, y_train_encoded)
        test_score = model.score(X_test, y_test_encoded)

    print(f"{name}")
    print(f"  Train Score: {train_score:.4f}")
    print(f"  Test Score:  {test_score:.4f}\n")
