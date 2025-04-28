import a
import pandas as pd # 데이터프레임 조작
import numpy as np # 결측치 처리 및 수치 계산
from sklearn.preprocessing import StandardScaler # Z-score 정규화

input_file = pd.read_csv('1_adults.csv')
def some_function(input_file):
    df=input_file.copy() # df로 복사
    obj_cols = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country', 'income']
    # 범주형 데이터 한 리스트로 묶기-> 이 과정은 info를 통해 확인해 구현했습니다.

    # unique한 값이 확인결과 기존 데이터셋에서는 결측치가 '?'로 되어있었습니다. 공백으로 변경해 후에 결측치 처리
    for col in obj_cols:
        df[col] = df[col].replace('?', '') # '?'를 공백으로 대체 해 나중에 결측치 처리

    df=a.first_step(df) # 결측치 처리 및 중복 제거  

    df=a.bins2(df,'education.num',[0,0.25,0.5,0.75,1],['low','mid','high','very high'],'edu_level') 
    # education.num을 사분위수로 나누어로 나누기-> 파생변수 생성

    remove_target = ['education.num', 'capital.gain'] # 제거할 컬럼 
    # education.num은 사분위수로 나누어져서 필요없고, capital.gain은 모든 엘리먼트가 0이라 제거
    df=a.remove_col(df, remove_target) 

    num_cols = ['age', 'fnlwgt', 'hours.per.week','capital.loss'] # 수치형 데이터
    df=a.remove_out(df,num_cols) # 4분위수 근거한 이상치 제거

    obj_cols=obj_cols + ['edu_level'] # 범주형 데이터에 파생변수 추가
    df=a.cat_encode(df,obj_cols) # income를 One-hot Encoding

    df=a.num_encode(df,num_cols) # 수치형 데이터 z-score 정규화
    return df

output_file = some_function(input_file)
output_file.to_csv('output.csv', index=False) # 결과를 CSV 파일로 저장