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


# ------------------------------------------------------------
# 4단계: 이상치 탐지 및 제거 (price)
# 가격 1000 이하만 남기기
# ------------------------------------------------------------
df = df[(df['price'] <= 1000)]

# ------------------------------------------------------------
# 5단계: 범주형 변수 Label Encoding
# 문자형(object) 컬럼을 숫자로 변환
# ------------------------------------------------------------
cat_cols = df.select_dtypes(include='object').columns.tolist()

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# ------------------------------------------------------------
# 6단계: 수치형 변수 정규화 (price만)
# 나이(price)만 0~1 사이로 정규화
# ------------------------------------------------------------
scaler = MinMaxScaler()
df['price_scaled'] = scaler.fit_transform(df[['price']])

# ------------------------------------------------------------
# 7단계: 파생변수 생성 - 나이대 그룹 생성
# 10대/20대/30대/... 구간별로 묶기
# ------------------------------------------------------------
def availability_365_group(availability_365):
    if availability_365 > 0:
        return '가능'
    elif availability_365 == 365:
        return '매일 가능'
    else:
        return '불가능'

df['availability_365_group'] = df['availability_365'].apply(availability_365_group)

# ------------------------------------------------------------
# 8단계: 예약 가능성별 가격비교 시각화 (matplotlib 사용)
# ------------------------------------------------------------
availability_price_rate = df.groupby('availability_365_group')['price'].mean().sort_index()

plt.figure(figsize=(8, 5))
plt.bar(availability_price_rate.index, availability_price_rate.values)
plt.title('Average price Rate by availability_365 Group')
plt.xlabel('availability Group')
plt.ylabel('price Rate')
plt.ylim(0, 1)
plt.grid(axis='y')
plt.show()

# ------------------------------------------------------------
# 9단계: 최종 데이터 저장
# ------------------------------------------------------------
df.to_csv('3_AB_cleaned.csv', index=False)

print("최종 전처리 및 파일 저장 완료: '3_AB_cleaned.csv'")
