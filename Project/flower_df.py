import os
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# 이미지가 직접 들어있는 폴더 경로
base_dir = r"C:\Users\KDT-37\Desktop\KDT_7\09_ML_CV\Project\data"
output_path = r"C:\Users\KDT-37\Desktop\KDT_7\09_ML_CV\Project\보라_output.csv"

data_rows = []

# data 폴더 내의 이미지 파일만 직접 순회
for img_name in os.listdir(base_dir):
    if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    img_path = os.path.join(base_dir, img_name)
    try:
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (100, 100))

        pixels = img.reshape(-1, 3)
        kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
        kmeans.fit(pixels)

        colors = kmeans.cluster_centers_.astype(int)

        row = [img_name]
        for color in colors:
            row += list(color)  # R, G, B

        data_rows.append(row)

    except Exception as e:
        print(f"Error processing {img_path}: {e}")

# 컬럼 정의
columns = ['filename']
for i in range(1, 6):
    columns += [f'R{i}', f'G{i}', f'B{i}']

# 저장
df = pd.DataFrame(data_rows, columns=columns)
df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"\n저장 완료 (ratio 제외): {output_path}")

