# 필요한 라이브러리 불러오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ------------------------------------------------------------
# 1단계: 데이터 로딩
# ------------------------------------------------------------
df = pd.read_csv('3_AB.csv', encoding='utf-8-sig')

# ------------------------------------------------------------
# 2단계: 불필요한 컬럼 제거
# last_review, host_id 컬럼 삭제
# ------------------------------------------------------------
df = df.drop(columns=['last_review', 'host_id'])

# ------------------------------------------------------------
# 3단계: 결측치 처리
# 문자형 데이터(object)는 'Unknown'으로, 숫자형 데이터는 평균(mean)으로 채움
# ------------------------------------------------------------
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna('Unknown')
    else:
        df[col] = df[col].fillna(df[col].mean())
