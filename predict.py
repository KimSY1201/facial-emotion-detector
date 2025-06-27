from utils import predict_image
from model_function import load_model, predict_image

image_path = r'C:\Users\KDT-37\Desktop\KDT_7\10_DL\project\2.png'

MODEL_PATH = 'Emotion_DNNMODEL.pt'  # 모델 경로
model = load_model(MODEL_PATH)

result, confidence = predict_image(model, image_path)
print(f"예측된 감정: {result}, 확률: {confidence:.2f}")