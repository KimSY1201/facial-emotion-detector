import os
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# 폴더 경로
base_dir = r"C:\Users\KDT-37\Desktop\KDT_7\09_ML_CV\Project\data"
output_path = r"C:\Users\KDT-37\Desktop\KDT_7\09_ML_CV\Project\보라_HSV_mean_output.csv"

data_rows = []

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

        rgb_list = []
        hsv_list = []

        row = [img_name]
        for color in colors:
            r, g, b = color
            rgb_list.append([r, g, b])

            # RGB → HSV
            rgb_pixel = np.uint8([[[r, g, b]]])
            hsv_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2HSV)[0][0]
            h, s, v = hsv_pixel
            hsv_list.append([h, s, v])

            # RGB + HSV 값 추가
            row += [r, g, b, h, s, v]

        # 평균 계산 및 추가
        rgb_mean = np.mean(rgb_list, axis=0).astype(int)
        hsv_mean = np.mean(hsv_list, axis=0).astype(int)
        row += rgb_mean.tolist() + hsv_mean.tolist()  # [R_avg, G_avg, B_avg, H_avg, S_avg, V_avg]

        data_rows.append(row)

    except Exception as e:
        print(f"Error processing {img_path}: {e}")

# 컬럼 정의
columns = ['filename']
for i in range(1, 6):
    columns += [f'R{i}', f'G{i}', f'B{i}', f'H{i}', f'S{i}', f'V{i}']

# 평균 컬럼 추가
columns += ['R_avg', 'G_avg', 'B_avg', 'H_avg', 'S_avg', 'V_avg']

# 저장
df = pd.DataFrame(data_rows, columns=columns)
df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"\n저장 완료 (HSV 포함 + 평균 컬럼 추가): {output_path}")
