import a
import pandas as pd # 데이터프레임 조작
import numpy as np # 결측치 처리 및 수치 계산
from sklearn.preprocessing import StandardScaler # Z-score 정규화

input_file = pd.read_csv('1_adults.csv')

def some_function(input_file):
    df=input_file.copy()
    obj_cols = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country', 'income']
    for col in obj_cols:
        df[col] = df[col].replace('?', '') # '?'를 공백으로 대체 해 나중에 결측치 처리
    df=a.first_step(df) # 결측치 처리 및 중복 제거  
    df=a.bins2(df,'education.num',[0,0.25,0.5,0.75,1],['low','mid','high','very high'],'edu_level') # education.num을 bins로 나누기
    remove_cols = ['education.num', 'capital.gain', 'capital.loss'] # 제거할 컬럼
    df=a.remove_cols(df, remove_cols) 
    num_cols = ['age', 'fnlwgt', 'hours.per.week']
    df=a.remove_out(df,num_cols)
    return df

output_file = some_function(input_file)