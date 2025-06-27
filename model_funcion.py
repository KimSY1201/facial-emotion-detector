## 학습 관련 함수
## --------------------------------------------------------------
## - 검증 함수 : 테스트 또는 검증용 데이터셋 사용하는 함수 
##              W,b 업데이트 안함
## --------------------------------------------------------------
def evaluate(model, testDL, loss_fn, score_fn, n_iter):
    # 에포크 단위로 검증 => 검증 모드
    model.eval()

    # W, b가 업데이트 해제
    with torch.no_grad():
        T_LOSS, T_ACC = 0, 0
        for feature, target in testDL:
            # 학습 진행
            pre_y = model(feature)

            # 손실 계산
            target = target.float().unsqueeze(1)  # target을 [batch_size, 1] 형태로 변환
            loss = loss_fn(pre_y, target)
        
            # 정확도 계산: 시그모이드 적용 후 0.5 기준으로 임계값 처리
            pred = torch.sigmoid(pre_y)  # 로짓을 확률로 변환
            pred = (pred >= 0.5).float()  # 확률을 0 또는 1로 변환 (0.5 기준)

            acc = (pred == target).float().mean()  # 정확도 계산

            T_LOSS += loss.item()
            T_ACC  += acc.item()

    return T_LOSS/n_iter, T_ACC/n_iter



from torchmetrics.classification import BinaryF1Score, BinaryAccuracy

def training(model, trainDL, optimizer, loss_fn, acc_fn, n_iter):
    model.train()

    E_LOSS, E_ACC, E_SCORE = 0, 0, 0
    for feature, target in trainDL:
        feature, target = feature.to(DEVICE), target.to(DEVICE).float().unsqueeze(1)
        
        # 가중치 기울기 초기화
        optimizer.zero_grad()

        # 예측값 계산 (로짓 값)
        pre_y = model(feature)

        # 손실 계산
        loss = loss_fn(pre_y, target)

        # 모델 출력값을 Sigmoid 함수로 확률로 변환
        prob = torch.sigmoid(pre_y)

        # 예측 값(0 또는 1)으로 변환 (명시적으로 threshold=0.5 적용)
        pred = (prob > 0.5).float()

        # 정확도와 F1 Score 계산
        #score = score_fn(pred, target.int())
        
        # 정확도 계산 (acc_fn 사용)
        acc = acc_fn(pred, target.int())  # BinaryAccuracy 인스턴스 사용
        
        # 역전파 진행
        loss.backward()
        
        # 가중치 업데이트
        optimizer.step()

        E_LOSS += loss.item()
        # E_SCORE += score.item()
        E_ACC  += acc.item()

    return E_LOSS / n_iter, E_ACC / n_iter