import os
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# 폴더 경로
base_dir = r"C:\Users\KDT-37\Desktop\KDT_7\09_ML_CV\Project\data"
output_path = r"C:\Users\KDT-37\Desktop\KDT_7\09_ML_CV\Project\보라_crop_std_output.csv"

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

        # 가운데 50x50 자르기
        h, w, _ = img.shape
        cx, cy = w // 2, h // 2
        crop_size = 50
        x1, y1 = cx - crop_size // 2, cy - crop_size // 2
        x2, y2 = cx + crop_size // 2, cy + crop_size // 2
        img_crop = img[y1:y2, x1:x2]

        pixels = img_crop.reshape(-1, 3)
        kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_.astype(int)

        rgb_list = []
        hsv_list = []
        row = [img_name]

        for color in colors:
            r, g, b = color
            rgb_list.append([r, g, b])

            rgb_pixel = np.uint8([[[r, g, b]]])
            hsv_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2HSV)[0][0]
            h, s, v = hsv_pixel
            hsv_list.append([h, s, v])

            row += [r, g, b, h, s, v]

        rgb_mean = np.mean(rgb_list, axis=0).astype(int)
        hsv_mean = np.mean(hsv_list, axis=0).astype(int)

        rgb_std = np.std(rgb_list, axis=0).astype(int)
        hsv_std = np.std(hsv_list, axis=0).astype(int)

        row += rgb_mean.tolist() + hsv_mean.tolist()
        row += rgb_std.tolist() + hsv_std.tolist()

        data_rows.append(row)

    except Exception as e:
        print(f"Error processing {img_path}: {e}")

# 컬럼 정의
columns = ['filename']
for i in range(1, 6):
    columns += [f'R{i}', f'G{i}', f'B{i}', f'H{i}', f'S{i}', f'V{i}']
columns += ['R_avg', 'G_avg', 'B_avg', 'H_avg', 'S_avg', 'V_avg']
columns += ['R_std', 'G_std', 'B_std', 'H_std', 'S_std', 'V_std']

# 저장
df = pd.DataFrame(data_rows, columns=columns)
df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"\n 저장 완료 (중앙 자른 이미지 기준 + 평균 + 표준편차): {output_path}")

