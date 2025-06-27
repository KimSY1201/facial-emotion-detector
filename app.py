import streamlit as st
from utils import load_model, predict_image
from PIL import Image
import os
from torchvision.transforms import transforms
import torch
from DNNmodel import EmotionDNN  # 모델 클래스 정의된 파일에서 임포트

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nanum+Myeongjo&family=Gowun+Dodum&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Gowun Dodum', sans-serif;
        background-image: url('https://search.pstatic.net/common/?src=http%3A%2F%2Fblogfiles.naver.net%2FMjAyMjA3MDdfMTEx%2FMDAxNjU3MjAzNjE5NTU0.zsBbBRX2VNlIE_f2t-DS3WJp3yvcLAyDjTnypy_6Tkgg.57259km8Ta8GimSNTfKtkOFmVBPLCCVrsrSjGeveeb0g.PNG.sismay%2F5.PNG&type=sc960_832');
        background-position: center;
        background-attachment: fixed;
        background-size: cover;
        color: #3e3e3e;
    }
    .stApp {
        background-color: rgba(255, 255, 255, 0.6);
        border-radius: 12px;
        padding: 1.5rem;
    }
    h1, h2, h3 {
        color: #4e2a84;
        text-shadow: 1px 1px 2px #e7d8f5;
        font-family: 'Nanum Myeongjo', serif;
    }
    .card {
        background: rgba(255, 255, 255, 0.8);
        border-left: 8px solid #a47cc2;
        padding: 1.2rem;
        margin-top: 1rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #e3c5ff;
        color: #4e2a84;
        border-radius: 10px;
        font-weight: bold;
        padding: 0.5em 1.5em;
        border: none;
    }
    a {
        color: #8a4fff;
        text-decoration: none;
        font-weight: bold;
    }
    a:hover {
        color: #d46ffb;
    }
    </style>
""", unsafe_allow_html=True)


# 모델 로딩 (이미 임포트한 EmotionDNN 클래스를 사용하여 모델을 초기화)c
MODEL_DIR = './models/'

# 모델 정의 및 가중치 로드
WEIGHTS_FILE = MODEL_DIR+'emotion_weights_epoch11_0.943.pt'
model = EmotionDNN()
states=torch.load(WEIGHTS_FILE, weights_only=True)
model.load_state_dict(states)
model.eval()  # 추론 모드로 전환



# Streamlit 앱
st.title("인사이드아웃 ver. 버럭이")

st.write("이 곳은 업로드된 이미지를 분석하여 감정을 예측해주는 사이트입니다.")

# 이미지 업로드
uploaded_file = st.file_uploader("이미지를 업로드하세요.", type=["jpg", "png", "jpeg"])

# 이미지 업로드 후 예측
if uploaded_file is not None:
    # 이미지 표시
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # 예측 버튼
    if st.button("예측하기"):
        # 이미지 전처리
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Grayscale 변환
            transforms.Resize((48, 48)),                  # 크기 조정
            transforms.ToTensor(),                        # 텐서로 변환
            transforms.Normalize((0.5,), (0.5,))          # 정규화
        ])

        # 이미지 전처리
        img = transform(image).unsqueeze(0)  # 배치 차원 추가

        # 예측 수행
        with torch.no_grad():  # 기울기 계산을 비활성화하여 예측만 수행
            output = model(img)  # 모델 출력 (로짓 값)
        
        prediction = torch.sigmoid(output).item()  # sigmoid를 적용하여 확률로 변환

        # 예측 결과 표시
        if prediction > 0.5:
            st.write(f"예측된 감정: 긍정적인 감정, 확률: {prediction:.2f}")
        else:
            st.write(f"예측된 감정: 부정적인 감정, 확률: {1 - prediction:.2f}")
            
            # 부정적인 감정이 예측되었을 때, 이미지를 띄우기
            negative_image = Image.open('angry2.png')  # 부정적인 감정에 해당하는 이미지 경로
            st.image(negative_image, caption="부정적인 감정 이미지", use_column_width=True)