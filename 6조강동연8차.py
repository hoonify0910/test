import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy import stats

input_file = "2_Card.csv"
#전처리 함수
def preprocess_credit_data(input_file):
  
    df = pd.read_csv(input_file)
    df = df.drop(columns=['ID'])

    y = df['default.payment.next.month']
    X = df.drop(columns=['default.payment.next.month'])

    numeric_cols = ['LIMIT_BAL', 'AGE',
                    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
                    'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
                    'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

    categorical_cols = ['SEX', 'EDUCATION', 'MARRIAGE',
                        'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    #KNNImputer로 결측치 보간
    imputer = KNNImputer(n_neighbors=2)
    X[numeric_cols] = imputer.fit_transform(X[numeric_cols])

    #Z스코어 2를 기준으로 이상치 판단
    z = np.abs(stats.zscore(X[numeric_cols]))
    X = X[(z < 2).all(axis=1)]
    y = y.loc[X.index]

    #원핫 인코더 
    encoder = OneHotEncoder(sparse=False)
    encoded = encoder.fit_transform(X[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))

    #Standard Scaler를 사용하여 수치 정규화
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X[numeric_cols])
    scaled_df = pd.DataFrame(scaled, columns=numeric_cols)

    # GPT가 짜준 하나의 X 데이터로 합치는 코드
    X_processed = pd.concat([scaled_df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

    return X_processed, y.reset_index(drop=True), X.reset_index(drop=True)  # X(원본 컬럼 유지)도 같이 리턴

# 연체 예측 함수    
def predict(X_raw):
    #연체 판단 점수를 risk score로 limit score는 기존 연체로 판단한 점수 education score는 학업 수준에 따른 점수수
    limit_score = (X_raw['LIMIT_BAL'].max() - X_raw['LIMIT_BAL']) / X_raw['LIMIT_BAL'].max()
    education_score = (4 - X_raw['EDUCATION']) / 3

    risk_score = (limit_score + education_score) / 2

    X_raw['predicted_default'] = (risk_score >= 0.5).astype(int)

    X_raw.to_csv('predicted_result.csv')

   

# 전처리 함수 수행
X_processed, y, X_raw = preprocess_credit_data(input_file)

predict(X_raw)

