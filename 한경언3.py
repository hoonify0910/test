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
df["기다린_기간"] = (df["AppointmentDay"] - df["ScheduledDay"]).dt.days
df["성별"] = df["Gender"].map({"M": 0, "F": 1})
df["건강위험"] = df[["Hipertension", "Diabetes", "Alcoholism", "Handcap"]].sum(axis=1)

print(df[["ScheduledDay", "AppointmentDay", "기다린_기간"]].head())

import matplotlib.pyplot as plt
import seaborn as sns

# 한글 깨짐 방지
plt.rc('font', family='NanumGothic')

# 1. 나이 분포 (Age) vs No-show 여부
plt.figure(figsize=(10,6))
sns.histplot(data=df, x="Age", hue="No-show", multiple="stack", bins=30)
plt.title("나이에 따른 노쇼 여부")
plt.show()

# 기다린 기간 분포 vs No-show 여부
plt.figure(figsize=(10,6))
sns.histplot(data=df, x="기다린_기간", hue="No-show", multiple="stack", bins=30)
plt.title("예약까지 대기일에 따른 노쇼 여부")
plt.show()

# 3. SMS 수신 여부에 따른 노쇼 비율
plt.figure(figsize=(6,4))
sns.barplot(data=df, x="SMS_received", y=df["No-show"].apply(lambda x: 1 if x == "Yes" else 0))
plt.title("SMS 수신 여부별 노쇼 비율")
plt.xlabel("SMS 수신 여부 (0=안받음, 1=받음)")
plt.ylabel("노쇼 비율")
plt.show()

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# def evaluate_team(df, label_col='target'):
#     X = df.drop(columns=[label_col])
#     y = df[label_col]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    
#     model = RandomForestClassifier(random_state=0)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
    
#     return accuracy_score(y_test, y_pred)

