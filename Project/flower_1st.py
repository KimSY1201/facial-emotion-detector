'''
[ 꽃 분류 ]

“봄의 감성 팔레트: 꽃의 색으로 감정을 피우다”

꽃 사진 속 색상으로 감정을 예측하고, 감성 콘텐츠를 추천하는 AI 큐레이터

✅ 프로젝트 목적
	•	봄꽃 사진을 분석하여 대표 색상 정보를 추출하고,
	•	그 색감이 주는 감정(설렘, 따뜻함 등)을 머신러닝으로 예측하여,
	•	감정에 어울리는 감성 문장, 음악, 스타일 등을 자동으로 추천하는 프로젝트입니다.

< 전체 흐름도 >

[꽃 이미지 입력]
    ↓
[KMeans로 대표 색상 추출 (공통)]
    ↓
[지배적인 색상 계열 분류 (공통)]
    ↓
[해당 색상 담당 팀원의 ML 모델 호출 (개별)]
    ↓
[감정 예측 → 감성 콘텐츠 생성 및 출력 (개별)]

✅ 통합 구조
	•	main.py: 꽃 이미지 입력 → 색상 추출 → 색상 분류 → 해당 팀원 모델 호출 → 결과 출력
	•	개인별 .py 파일을 import하여 if문 분기 처리
	•	최종적으로는 사용자 1명이 사진 1장을 넣었을 때,
    
감정 예측 + 감성 콘텐츠가 자동 생성되는 시스템

'''
import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ---------- 설정 ----------
IMAGE_PATH = "your_flower_image.jpg"  # 분석할 꽃 이미지 경로
NUM_COLORS = 3                        # 추출할 대표 색상 수
RESIZE_SHAPE = (100, 100)            # 전처리 크기 (작을수록 연산 빠름)

# ---------- 1. 이미지 불러오기 & 전처리 ----------
img = Image.open(IMAGE_PATH).convert("RGB")  # PIL로 열고 RGB 변환
img = img.resize(RESIZE_SHAPE)               # 리사이즈
img_np = np.array(img)

# ---------- 2. (H, W, 3) → (N, 3)으로 reshape ----------
pixels = img_np.reshape(-1, 3)

# ---------- 3. KMeans로 대표 색상 추출 ----------
kmeans = KMeans(n_clusters=NUM_COLORS, random_state=42)
kmeans.fit(pixels)
colors = kmeans.cluster_centers_.astype(int)

# ---------- 4. 대표 색상 시각화 ----------
def plot_colors(colors):
    plt.figure(figsize=(6, 2))
    for i, color in enumerate(colors):
        plt.subplot(1, NUM_COLORS, i+1)
        plt.imshow(np.ones((50, 50, 3), dtype=np.uint8) * color)
        plt.axis('off')
        plt.title(f'RGB: {tuple(color)}')
    plt.tight_layout()
    plt.show()

print("🎨 추출된 대표 색상 RGB값:")
for i, color in enumerate(colors):
    print(f"{i+1}. {tuple(color)}")

plot_colors(colors)