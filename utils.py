import random
from torch.utils.data import Subset 
from collections import Counter
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
from PIL import Image
from model_funcion import *

def relabel_dataset(dataset):
    for i, (path, label) in enumerate(dataset.samples):
        if label == 0:  # 'angry' 라벨 번호가 0로 지정된 경우
            new_label = 1  # angry -> 1
        else:
            new_label = 0  # 나머지 모두 -> 0
        dataset.samples[i] = (path, new_label)


# 감정별로 이미지 선택

def get_custom_subset(dataset, num_images_per_class, classes_to_include_all_images):
    # 클래스별 인덱스 추출
    class_to_idx = dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # 모든 이미지를 사용할 클래스들
    selected_indices = []

    # angry, disgust, fear, surprise 클래스는 모든 이미지 사용
    for class_name in classes_to_include_all_images:
        class_idx = class_to_idx[class_name]
        # 해당 클래스의 이미지 인덱스 가져오기
        class_indices = [i for i, label in enumerate(dataset.targets) if label == class_idx]
        selected_indices.extend(class_indices)  # 모든 이미지 인덱스를 추가

    # 나머지 감정 클래스는 num_images_per_class만큼 랜덤으로 샘플링
    for class_idx in range(len(dataset.classes)):
        class_name = idx_to_class[class_idx]
        if class_name in classes_to_include_all_images:  # 이미 모든 이미지를 포함했으면 건너뛰기
            continue
        
        # 해당 클래스의 이미지 인덱스 추출
        class_indices = [i for i, label in enumerate(dataset.targets) if label == class_idx]
        
        # 9923개 샘플 랜덤 선택 (만약 해당 클래스의 이미지 수가 9923보다 적으면, 모든 이미지 선택)
        selected_class_indices = random.sample(class_indices, min(num_images_per_class, len(class_indices)))
        
        # 선택된 인덱스 추가
        selected_indices.extend(selected_class_indices)

    # Subset을 사용하여 선택된 인덱스만 포함된 데이터셋 생성
    return Subset(dataset, selected_indices)

# 각 감정 클래스별 이미지 수를 세기 위해 train_subset의 인덱스를 사용
def count_images_in_subset(subset, dataset):
    # 각 이미지 인덱스가 속한 클래스의 레이블을 찾아서 개수를 셈
    targets = [dataset.targets[i] for i in subset.indices]
    
    # Counter로 각 클래스별 이미지 수 계산
    class_counts = Counter(targets)
    
    # 각 클래스의 이름을 출력
    class_to_idx = dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # 클래스별 이미지 수 출력
    for class_idx, count in class_counts.items():
        class_name = idx_to_class[class_idx]
        print(f"{class_name}: {count} images")
        
# Focal Loss 함수
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
        
def predict_image(model, image_path):
    # 이미지 전처리
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.Grayscale(num_output_channels=1),  # 흑백 이미지로 변환
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),  # 정규화
    ])
    
    img = Image.open(image_path)
    img = transform(img).unsqueeze(0)  # 배치 차원 추가
    
    # 예측
    with torch.no_grad():
        output = model(img)
    prediction = torch.sigmoid(output).item()  # sigmoid를 적용하여 확률로 변환
    
    # "angry"인지 "not angry"인지 예측
    if prediction > 0.5:
        return "angry", prediction  # "angry"일 때
    else:
        return "not angry", 1 - prediction  # "not angry"일 때

def load_model(model_path):
    model = EmotionDNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 평가 모드로 전환
    return model