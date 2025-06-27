import streamlit as st
import joblib
from PIL import Image
import numpy as np
import pandas as pd
import utils
import cv2
import streamlit as st
import plotly.express as px
import random

color_content_map = {
    'Red': {
        'emotion': '열정, 사랑, 생기',
        'places': [
            "프랑스 그라스 장미 정원", "불가리아 로즈밸리", "제주 동백꽃 군락지",
            "이탈리아 베로나 사랑의 거리", "부산 태종대 장미 터널", "경주 보문단지 장미원",
            "서울 올림픽공원 장미광장", "토스카나 포피 들판", "진해 장미공원", "일본 키타큐슈 장미 정원"
        ],
        'quotes': [
            "당신은 봄보다 뜨거운 사람이에요.",
            "사랑은 붉게 피어나, 당신을 닮았어요.",
            "한 걸음마다 열정이 피어나는 계절이에요.",
            "햇살보다 강렬한 마음을 가졌군요.",
            "오늘 하루, 장미처럼 빛날 거예요.",
            "당신은 어디에 있든 중심이에요.",
            "그대는 봄의 시작이자 끝이에요.",
            "감정도 꽃처럼 피어나는 오늘.",
            "당신의 웃음은 세상을 물들여요.",
            "따뜻함이란, 당신을 말하는 말."
        ],
        'music': [
            "https://www.youtube.com/watch?v=fLexgOxsZu0",
            "https://www.youtube.com/watch?v=XKfgdkcIUxw",
            "https://www.youtube.com/watch?v=d-diB65scQU",
            "https://www.youtube.com/watch?v=lY2yjAdbvdQ",
            "https://www.youtube.com/watch?v=HgzGwKwLmgM",
            "https://www.youtube.com/watch?v=eVTXPUF4Oz4",
            "https://www.youtube.com/watch?v=w4cTYnOPdNk",
            "https://www.youtube.com/watch?v=Pkh8UtuejGw",
            "https://www.youtube.com/watch?v=09R8_2nJtjg",
            "https://www.youtube.com/watch?v=i62Zjga8JOM"
        ]
    },

    'Orange/Yellow': {
        'emotion': '밝음, 희망, 에너지',
        'places': [
            "미국 캔자스 해바라기 밭", "제주 유채꽃밭", "강릉 노란 코스모스 정원",
            "경주 황금빛 들판", "스페인 세비야 오렌지 나무 거리", "서울 양재천 노란 꽃길",
            "태국 치앙마이 해바라기 언덕", "충남 서천 유채꽃밭", "고양 호수공원 해바라기 정원", "함평 나비축제 해바라기밭"
        ],
        'quotes': [
            "희망은 오늘도 노랗게 피어납니다.",
            "당신의 하루에 햇살을 선물할게요.",
            "에너지가 넘치는 당신이 좋아요.",
            "따뜻한 빛은 늘 당신 편이에요.",
            "빛나는 그 마음, 참 예뻐요.",
            "오늘도 환한 웃음으로 가득하길!",
            "주황빛 노을처럼 따뜻한 사람이에요.",
            "희망은 멀지 않아요, 바로 당신이에요.",
            "햇살이 당신을 닮았네요.",
            "기분 좋은 일들이 노랗게 피어나길."
        ],
        'music': [
            "https://www.youtube.com/watch?v=ApXoWvfEYVU",  
            "https://www.youtube.com/watch?v=KQetemT1sWc",
            "https://www.youtube.com/watch?v=ZbZSe6N_BXs",
            "https://www.youtube.com/watch?v=L3wKzyIN1yk",
            "https://www.youtube.com/watch?v=hT_nvWreIhg",
            "https://www.youtube.com/watch?v=vNoKguSdy4Y",
            "https://www.youtube.com/watch?v=LsoLEjrDogU",
            "https://www.youtube.com/watch?v=CevxZvSJLk8",
            "https://www.youtube.com/watch?v=jZhQOvvV45w",
            "https://www.youtube.com/watch?v=nfWlot6h_JM"
        ]
    },
'Purple': {
    'emotion': '신비, 우아함, 몽환적',
    'places': [
        "일본 후라노 라벤더 언덕", "제주 보랏빛 수국길", "영국 히친 라벤더 농장",
        "경북 의성 라벤더밭", "프랑스 프로방스 라벤더 들판", "서울 보라빛 수국정원",
        "스페인 안달루시아 라벤더 마을", "강릉 경포 라벤더 정원", "평창 보랏빛 산책길", "충남 태안 보라빛 해안도로"
    ],
    'quotes': [
        "당신은 보랏빛 꿈처럼 신비로워요.",
        "고요함 속의 우아함, 당신을 닮았네요.",
        "마음이 몽환적으로 일렁이는 순간이에요.",
        "차분하지만 강렬한, 그런 사람.",
        "보랏빛은 단순히 색이 아닌 감정이에요.",
        "당신의 깊이는 하루만에 알 수 없어요.",
        "은은한 매력이 이 계절을 가득 채워요.",
        "몽환적인 그 느낌, 당신이 주는 선물이죠.",
        "오늘은 당신이 세상을 물들일 차례예요.",
        "신비로운 에너지가 당신을 감싸고 있어요."
    ],
    'music': [
        "https://www.youtube.com/watch?v=asU8O_R5V6w",
        "https://www.youtube.com/watch?v=o7_CH7MpizI",
        "https://www.youtube.com/watch?v=5-iKqhkWIxM",
        "https://www.youtube.com/watch?v=1gO_WHlFcZA",
        "https://www.youtube.com/watch?v=d2EtvXnm-0U"
    ]
},
'Green': {
    'emotion': '평화, 생명력, 자연',
    'places': [
        "스위스 루체른의 초원", "제주 삼다수 숲길", "경기도 양평 두물머리 숲",
        "뉴질랜드 테카포 호숫가", "속리산 숲길", "서울 북한산 자락길",
        "강원도 치악산 둘레길", "전북 남원 숲속 캠핑장", "오스트리아 잘츠카머구트 초원", "캐나다 밴프 국립공원"
    ],
    'quotes': [
        "숨 쉬듯 편안한 당신의 하루를 응원해요.",
        "자연이 주는 평화가 당신에게 닿길 바라요.",
        "초록은 마음의 안식처에요.",
        "조용한 위로가 되는 존재, 바로 당신이에요.",
        "숨결 하나에도 생명이 깃든 듯한 하루네요.",
        "당신과 걷는 길이 숲이라면 어디든 좋아요.",
        "초록빛 감성, 잊지 말아요.",
        "당신의 하루에 나무 그늘 같은 쉼이 있길.",
        "자연처럼 깊고 잔잔한 사람, 당신이에요.",
        "바람결마저 푸르른 오늘이 당신 곁에 머물길."
    ],
    'music': [
        "https://www.youtube.com/watch?v=UfcAVejslrU",
        "https://www.youtube.com/watch?v=pRpeEdMmmQ0",
        "https://www.youtube.com/watch?v=wRRsXxE1KVY",
        "https://www.youtube.com/watch?v=FOjdXSrtUxA",
        "https://www.youtube.com/watch?v=7PCkvCPvDXk",
        "https://www.youtube.com/watch?v=SlPhMPnQ58k",
        "https://www.youtube.com/watch?v=AbPED9bisSc",
        "https://www.youtube.com/watch?v=5NV6Rdv1a3I",
        "https://www.youtube.com/watch?v=EpCamGQliIs",
        "https://www.youtube.com/watch?v=5LkLqF2LDrg"
    ]
},
'Blue': {
    'emotion': '차분함, 신뢰, 고요함',
    'places': [
        "그리스 산토리니 바닷가", "제주 하늘오름길", "스위스 인터라켄 호수",
        "부산 해운대 바다 산책로", "시드니 본다이 비치", "괌 투몬비치",
        "태국 피피섬", "몰디브 바닷길", "강릉 안목 해변", "울산 대왕암 해안길"
    ],
    'quotes': [
        "당신은 바다처럼 깊고 잔잔해요.",
        "말 없는 위로가 되는 존재, 그게 당신이에요.",
        "고요함은 힘이 있다는 걸 느껴요.",
        "푸른 감정이 오늘을 감싸 안아주네요.",
        "바람보다 잔잔한 마음을 가졌군요.",
        "믿음을 주는 눈빛, 파랑처럼 선명해요.",
        "당신은 언제나 차분한 힘을 가진 사람이에요.",
        "파란 하늘 아래, 평온한 기운이 당신을 감싸요.",
        "고요한 순간에 가장 진한 감정이 피어나요.",
        "흔들리지 않는 당신의 중심이 좋아요."
    ],
    'music': [
        "https://www.youtube.com/watch?v=BiQIc7fG9pA",
        "https://www.youtube.com/watch?v=kJQP7kiw5Fk",
        "https://www.youtube.com/watch?v=8xg3vE8Ie_E",
        "https://www.youtube.com/watch?v=t5Sd5c4o9UM",
        "https://www.youtube.com/watch?v=AJtDXIazrMo",
        "https://www.youtube.com/watch?v=RgKAFK5djSk",
        "https://www.youtube.com/watch?v=7wtfhZwyrcc",
        "https://www.youtube.com/watch?v=ScNNfyq3d_w",
        "https://www.youtube.com/watch?v=hLQl3WQQoQ0",
        "https://www.youtube.com/watch?v=OPf0YbXqDm0"
    ]
},
'Others': {
    'emotion': '창의, 자유, 다양성',
    'places': [
        "서울 DDP 전시관", "뉴욕 소호 거리", "프랑스 파리 몽마르트 언덕",
        "홍콩 침사추이 야경", "도쿄 시부야", "런던 코벤트가든",
        "베를린 거리예술 골목", "서울 성수동 갤러리길", "방콕 아트마켓", "LA 아보트 키니 거리"
    ],
    'quotes': [
        "당신은 하나뿐인 색이에요.",
        "정답 없는 자유로움이 오늘을 채워요.",
        "창의는 당신처럼 다양해요.",
        "세상은 당신의 시선으로 다시 태어나요.",
        "독특함은 아름다움이에요.",
        "지금 이 순간도 당신의 색으로 물들고 있어요.",
        "틀에 얽매이지 않는 감성이 멋져요.",
        "오늘 하루도 다채롭게 빛나요.",
        "다름은 곧 가능성이라는 걸 잊지 마요.",
        "당신의 자유로운 색채가 세상을 환하게 해요."
    ],
    'music': [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  
        "https://www.youtube.com/watch?v=JGwWNGJdvx8",
        "https://www.youtube.com/watch?v=fRh_vgS2dFE",
        "https://www.youtube.com/watch?v=KQ6zr6kCPj8",
        "https://www.youtube.com/watch?v=QK8mJJJvaes",
        "https://www.youtube.com/watch?v=hT_nvWreIhg",
        "https://www.youtube.com/watch?v=PkIsu2gaG4I",
        "https://www.youtube.com/watch?v=SXiSVQZLje8",
        "https://www.youtube.com/watch?v=0KSOMA3QBU0",
        "https://www.youtube.com/watch?v=3AtDnEC4zak"
    ]
}
}

# Streamlit 시작
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nanum+Myeongjo&family=Gowun+Dodum&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Gowun Dodum', sans-serif;
        background-image: url('https://search.pstatic.net/common/?src=http%3A%2F%2Fblogfiles.naver.net%2FMjAxOTAyMjRfNjEg%2FMDAxNTUwOTg1OTM0OTQz.XIiYU3spQvH8pFd_sMzo1RTIi_3uCvjH9Ek4DhMBgx0g.g4cWugzXsn62oYsCfebN_KIP3y80ZT7PPnad1SZy-f4g.JPEG.yzzzii%2Foutput_1625799828.jpg&type=sc960_832');
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

# Streamlit 시작
st.title("Flower Emotion")

uploaded_file = st.file_uploader("이미지를 업로드하세요", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='업로드된 이미지', use_container_width=True)

    temp_path = "temp_uploaded.jpg"
    image.save(temp_path)

    bundle = joblib.load('decision_tree_model.pkl')
    dt_model = bundle['model']
    features = bundle['features']

    df_input = utils.extract_avg_std_features(temp_path)
    df_input = df_input[features]

    with st.spinner(' 꽃을 살펴보는 중이에요... 조금만 기다려주세요!'):
        pred = dt_model.predict(df_input)[0]
        proba = dt_model.predict_proba(df_input)[0]
        class_names = dt_model.classes_

        color_group = pred
        content = color_content_map.get(color_group, {
            'emotion': '감정을 찾을 수 없음',
            'places': ['어딘가의 꽃길'],
            'quotes': ['당신만의 색으로 세상을 물들여요.'],
            'music': [None]
        })

        place = random.choice(content['places'])
        quote = random.choice(content['quotes'])
        music_url = random.choice(content['music'])
        emotion = content['emotion']

        # 콘텐츠 추천 카드 먼저 보여주기
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='font-size: 28px;'>꽃에 어울리는 장소: {place}</h3>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='font-size: 28px;'>꽃이 주는 감정: {emotion}</h3>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='font-size: 28px;'>감성 글귀: “{quote}”</h3>", unsafe_allow_html=True)
        if music_url:
            st.video(music_url)
        else:
            st.info("어울리는 음악이 아직 준비되지 않았어요!")
        st.markdown("</div>", unsafe_allow_html=True)

        # 예측 결과 시각화 (예쁜 Plotly 차트)
        df_proba = pd.DataFrame({"색상": class_names, "확률": proba})
        fig = px.bar(
            df_proba,
            x="색상",
            y="확률",
            color="색상",
            color_discrete_sequence=px.colors.sequential.Purples_r,
            title=" 꽃 색상별 예측 확률",
        )
        fig.update_layout(
            plot_bgcolor='rgba(255,255,255,0.0)',
            paper_bgcolor='rgba(255,255,255,0.0)',
            font=dict(family='Gowun Dodum', size=16, color='#4e2a84'),
            title_font=dict(size=24, family='Nanum Myeongjo', color='#4e2a84'),
            margin=dict(t=60, b=40, l=10, r=10),
            height=400
        )
        fig.update_traces(marker_line_color='white', marker_line_width=2)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader(f" 꽃 색상: {pred}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.plotly_chart(fig, use_container_width=True)
