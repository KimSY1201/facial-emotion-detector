import torch.nn as nn
import torch.nn.functional as F

class EmotionDNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 입력 크기가 48x48인 이미지를 처리하는 레이어로, 48 * 48 = 2304
        self.in_layer   = nn.Flatten()  # 3D (BS, H, W) -> 2D (BS, H*W)
        
        # 첫 번째 Hidden Layer + 배치 정규화 (Batch Normalization)
        self.hd_layer1  = nn.Linear(2304, 512)  # 2304 -> 512
        self.batch_norm1 = nn.BatchNorm1d(512)  # 배치 정규화
        
        # 두 번째 Hidden Layer + 배치 정규화
        self.hd_layer2  = nn.Linear(512, 256)   # 512 -> 256
        self.batch_norm2 = nn.BatchNorm1d(256)  # 배치 정규화
        
        # 세 번째 Hidden Layer
        self.hd_layer3  = nn.Linear(256, 130)
        self.batch_norm3 = nn.BatchNorm1d(130)
        
        self.out_layer  = nn.Linear(130, 1)
        
        self.drop_layer = nn.Dropout(0.5)


    def forward(self, data):
        # 데이터의 크기 출력 (배치 크기, 높이, 너비)
        # print(f'Input data shape: {data.shape}')
        
        # Flatten (3D -> 2D)
        out = self.in_layer(data)
        # print(f'After Flatten: {out.shape}')

        # 첫 번째 Hidden Layer + 배치 정규화 + Dropout
        out = F.relu(self.hd_layer1(out))
        out = self.batch_norm1(out)  # 배치 정규화
        out = self.drop_layer(out)

        # 두 번째 Hidden Layer + 배치 정규화 + Dropout
        out = F.relu(self.hd_layer2(out))
        out = self.batch_norm2(out)  # 배치 정규화
        out = self.drop_layer(out)

        # 세 번째 Hidden Layer
        out = F.relu(self.hd_layer3(out))
        out = self.batch_norm3(out)
        out = self.drop_layer(out)



        # Output Layer
        out = self.out_layer(out)
        
        # print(f'Output shape: {out.shape}')
        return out