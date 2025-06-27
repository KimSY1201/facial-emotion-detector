import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def extract_dominant_colors(img, k=5, show_chart=True, img_name="Image"):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (100, 100))
    pixel_data = img.reshape((-1, 3))

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixel_data)

    colors = kmeans.cluster_centers_.astype(int)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    proportions = counts / counts.sum()

    if show_chart:
        plt.figure(figsize=(6, 3))
        for i in range(k):
            plt.bar(i, proportions[i], color=colors[i]/255, width=1)
        plt.title(f'Dominant Colors - {img_name}')
        plt.xticks([])
        plt.yticks([])
        plt.show()

    return list(zip(colors, proportions))

# Tulip 폴더 내 모든 이미지 처리
folder_path = 'data/Tulip'  # Tulip 폴더 이름
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    img = cv2.imread(image_path)

    print(f"\n{image_file}의 대표 색상:")
    dominant_colors = extract_dominant_colors(img, k=5, show_chart=True, img_name=image_file)

    for idx, (color, ratio) in enumerate(dominant_colors):
        print(f"  Color {idx + 1}: RGB={color}, 비율={ratio:.2f}")
