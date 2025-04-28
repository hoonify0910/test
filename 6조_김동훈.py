import a
import pandas as pd # 데이터프레임 조작
import numpy as np # 결측치 처리 및 수치 계산
from sklearn.preprocessing import StandardScaler # Z-score 정규화

# 1번데이터에 대해 진행
# 진행과정은 주피터 노트북으로 각 결과값들을 확인해가며 진행했습니다. 
# a.py는 각 함수들을 직관적으로 네이밍 해서 시험 전에 만든 모듈 파일입니다. 
# a.py와 같은 폴더에 있어야 합니다.

input_file = pd.read_csv('1_adults.csv')
def some_function(input_file):
    df=input_file.copy() # df로 복사
    obj_cols = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country', 'income']
    # 범주형 데이터 한 리스트로 묶기-> 이 과정은 info를 통해 확인해 구현했습니다.
    # 타자가 번거로워 이 부분에 대해선 지피티에게 리스트 타입 형태로 만들어 달라 하였습니다.

    # unique값 확인결과 기존 데이터셋에서는 결측치가 '?'로 표현 되어있었습니다. 공백으로 변경해 후에 결측치 처리
    for col in obj_cols:
        df[col] = df[col].replace('?', '') # '?'를 공백으로 대체 해 나중에 결측치 처리

    df=a.first_step(df) # 결측치 처리 및 중복 제거  

    df=a.bins2(df,'education.num',[0,0.25,0.5,0.75,1],['low','mid','high','very high'],'edu_level') 
    # education.num을 사분위수로 나누어로 나누기-> 파생변수 생성

    remove_target = ['education.num', 'capital.gain','capital.loss'] # 제거할 컬럼 
    # education.num은 사분위수로 나누어져서 필요없음 
    # capital.gain은 모든 요소가 0이라 제거
    # capital.loss는 대부분 값이 0이라 제거
    df=a.remove_col(df, remove_target) 

    num_cols = ['age', 'fnlwgt', 'hours.per.week'] # 수치형 데이터
    df=a.remove_out(df,num_cols) # 4분위수 근거한 이상치 제거

    obj_cols=obj_cols + ['edu_level'] # 범주형 데이터 리스트에 생성한 파생변수 추가
    obj_cols.remove('income') # income은 타겟변수로 제거-시험지시 사항
    df=a.cat_encode(df,obj_cols) # 범주형 데이터 One-hot Encoding

    df=a.num_encode(df,num_cols) # 수치형 데이터 z-score 정규화
    return df

output_file = some_function(input_file)
#output_file.to_csv('output.csv', index=False) # 결과를 CSV 파일로 저장
