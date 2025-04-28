import pandas as pd
import numpy as np
df = pd.read_csv('/content/4_MED_NS.csv', encoding='cp949')

print(df.isnull().sum())  # 전체 결측치 수

#중복된 값 확인하기
print(df.duplicated().sum())

#불필요한 컬럼 제거,모델 예측에 직접 필요없음
df = df.drop(["PatientId", "AppointmentID"], axis=1)


#날짜 데이터를 날짜형으로 변환
df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"])
df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"])

df["Gender"] = df["Gender"].map({"M": 0, "F": 1})

# noshow를 숫자로 변환 no:0 yes:1
df["No-show"] = df["No-show"].map({"No": 0, "Yes": 1})
df = df[(df["Age"] >= 0) & (df["Age"] <= 100)]


# 문자 수신 여부한 사람의 평균 노쇼율
noshow_sms = df.groupby("SMS_received")["No-show"].mean()
print(noshow_sms)

# 파생변수
df["예약기간"] = (df["AppointmentDay"] - df["ScheduledDay"]).dt.days
df["성별"] = df["Gender"].map({"M": 0, "F": 1})
df["건강위험"] = df[["Hipertension", "Diabetes", "Alcoholism", "Handcap"]].sum(axis=1)


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def evaluate_team(df, label_col='target'):
    X = df.drop(columns=[label_col])
    y = df[label_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    
    model = RandomForestClassifier(random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return accuracy_score(y_test, y_pred)

