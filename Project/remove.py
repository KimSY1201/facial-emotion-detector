import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def remove_background(image):
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    height, width = image.shape[:2]
    rect = (10, 10, width - 20, height - 20)
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    result = image * mask2[:, :, np.newaxis]
    return result

def extract_dominant_colors(img, k=5):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (100, 100))
    pixels = img.reshape(-1, 3)
    pixels = pixels[np.any(pixels != [0, 0, 0], axis=1)]  # Remove black background

    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(pixels)

    colors = kmeans.cluster_centers_.astype(int)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    proportions = counts / counts.sum()

    return list(zip(colors, proportions))

def classify_color_group(rgb_color):
    color = np.uint8([[rgb_color]])
    hsv = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)[0][0]
    h = hsv[0]
    if h < 15 or h >= 165:
        return 'Red'
    elif 15 <= h < 35:
        return 'Orange/Yellow'
    elif 35 <= h < 85:
        return 'Green'
    elif 85 <= h < 135:
        return 'Blue'
    elif 135 <= h < 165:
        return 'Purple'
    else:
        return 'Other'

def plot_colors(colors, proportions, filename):
    plt.figure(figsize=(6, 1))
    start = 0
    for color, prop in zip(colors, proportions):
        plt.fill_between([start, start + prop], 0, 1, color=np.array(color / 255))
        start += prop
    plt.xlim(0, 1)
    plt.axis('off')
    plt.title(filename, fontsize=10)
    plt.tight_layout()
    plt.show()

# 실행 부분
folder = 'data/Tulip'
output_csv = 'tulip_colors.csv'

image_files = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.png'))]
rows = []

for file in image_files:
    path = os.path.join(folder, file)
    img = cv2.imread(path)
    if img is None:
        continue

    fg_img = remove_background(img)
    dominant_colors = extract_dominant_colors(fg_img, k=5)

    #print(f"\n{file} 지배 색상 분석:")
    row = [file]
    for idx, (color, ratio) in enumerate(dominant_colors):
        hue_group = classify_color_group(color)
        #print(f"  Color {idx+1}: RGB={color}, 비율={ratio:.2f}, 계열={hue_group}")
        row += list(color) + [ratio]

    rows.append(row)

    # 대표 색상 시각화
    plot_colors([c for c, _ in dominant_colors],
                [r for _, r in dominant_colors],
                file)

# CSV 저장
columns = ['filename']
for i in range(1, 6):
    columns += [f'R{i}', f'G{i}', f'B{i}', f'Ratio{i}']

df = pd.DataFrame(rows, columns=columns)
df.to_csv(output_csv, index=False)
print(f"\n CSV 저장 완료: {output_csv}")
