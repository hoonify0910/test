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
