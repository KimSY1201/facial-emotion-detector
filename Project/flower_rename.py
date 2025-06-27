import os

def rename_images(folder_path, flower_name):
    if not os.path.exists(folder_path):
        print(f"경로 없음: {folder_path}")
        return
    
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    files.sort()  # 정렬 (선택)

    for idx, filename in enumerate(files, start=1):
        ext = os.path.splitext(filename)[1]
        new_name = f"{flower_name}_{idx:03d}{ext}"
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
        print(f"{filename} → {new_name}")
        


rename_images("data\Jin", "Jin")


