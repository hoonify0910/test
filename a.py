import pandas as pd               # 데이터프레임 조작
import numpy as np                # 결측치 처리 및 수치 계산
from sklearn.preprocessing import StandardScaler  # Z-score 정규화

def remove_col(df, cols):
    """
    지정한 열들을 데이터프레임에서 삭제하는 함수
    Parameters:
        df (pd.DataFrame): 원본 데이터프레임
        cols (list): 삭제할 열 이름 리스트
    """
    return df.drop(columns=cols, errors='ignore')

def first_step(df):
    """
    결측치 처리 및 중복 제거 함수
    처리 절차:
    1. 공백은 결측치로 변환
    2. 결측치 비율이 50% 이상인 컬럼은 제거
    3. 결측치 비율이 5% 이하인 컬럼은 해당 결측치 행 자체를 삭제s
    4. 남아 있는 결측치는:
    - 범주형(object) → 'unknown'으로 채움
    - 숫자형(float, int) → 평균값으로 채움
    5. 마지막으로 중복된 행 제거
    Parameters: df (pd.DataFrame): 원본 데이터프레임
    """
    # 1. 공백 문자열("")도 결측치로 간주
    df = df.replace(r'^\s*$', np.nan, regex=True)

    # 2. 컬럼별 결측치 비율 계산
    missing_ratio = df.isna().mean()

    # 3. 결측치 비율이 50% 이상인 컬럼 제거
    df = df.loc[:, missing_ratio < 0.5]

    # 4. 결측치 비율이 5% 이하인 컬럼의 경우 → 그 행 제거
    for col in df.columns:
        if df[col].isna().mean() <= 0.05:
            df = df[df[col].notna()]

    # 5. 나머지 결측치 처리
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('unknown')  # 범주형: 'unknown'
        else:
            df[col] = df[col].fillna(df[col].mean())  # 숫자형: 평균값 대체

    # 6. 중복된 행 제거
    df = df.drop_duplicates()
    return df

def remove_out(df, num_cols):
    """
    수치형 컬럼에서 4분위수를 기준으로 이상치를 제거하는 함수
    Parameters:
        df (pd.DataFrame): 원본 데이터프레임
        num_cols (list): 이상치 탐지할 수치형 컬럼 리스트
    """
    for col in num_cols:
        if col in df.columns:
            # 1사분위수(Q1), 3사분위수(Q3) 계산
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            # 이상치 경계 설정
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # IQR 범위를 벗어나는 값 제거
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

def cat_encode(df, cat_cols):
    """
    지정된 범주형 컬럼들을 One-Hot Encoding 방식으로 변환
    Parameters:
        df (pd.DataFrame): 원본 데이터프레임
        cat_cols (list): 인코딩할 범주형 컬럼 리스트
    """
    # get_dummies는 One-Hot Encoding 함수
    # drop_first=True: 첫 번째 범주는 제거 (기준 변수)
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    return df

def num_encode(df, num_cols):
    """
    지정된 수치형 컬럼들을 Z-score로 정규화 (평균 0, 표준편차 1)
    Parameters:
        df (pd.DataFrame): 원본 데이터프레임
        num_cols (list): 정규화할 수치형 컬럼 리스트
    """
    scaler = StandardScaler()

    # num_cols에 해당하는 컬럼만 정규화
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

def bins1(df, col, bins, labels=None, new_col=None):
    """
    지정한 수치형 컬럼을 구간(bins)으로 나누어 파생변수(범주형 컬럼)를 생성

    Parameters:
        df (pd.DataFrame): 원본 데이터프레임
        col (str): 구간화할 수치형 컬럼 이름
        bins (list): 경계값 리스트 (예: [0, 1000, 2000, 3000])
        labels (list, optional): 각 구간에 대응할 라벨 리스트 (예: ['낮음', '중간', '높음'])
        new_col (str, optional): 생성할 새로운 컬럼 이름. 지정하지 않으면 기본 이름 사용
    """
    if new_col is None:
        new_col = f"{col}_binned"

    # pd.cut을 이용해 구간 나누기 (labels는 생략 가능)
    df[new_col] = pd.cut(df[col], bins=bins, labels=labels)
    return df

def bins2(df, col, quantiles, labels=None, new_col=None):
    """
    분위수(quantile) 기반으로 수치형 컬럼을 구간 나눠 파생변수 생성

    Parameters:
        df (pd.DataFrame): 원본 데이터프레임
        col (str): 구간화할 수치형 컬럼 이름
        quantiles (list): 분위수 리스트 (0~1 사이 값들, 예: [0.0, 0.3, 0.6, 0.9, 1.0])
        labels (list, optional): 각 구간에 붙일 이름
        new_col (str, optional): 생성할 컬럼 이름 (기본: 'col_binned')
    """
    if new_col is None:
        new_col = f"{col}_binned"

    # 분위수 기반으로 구간 나누기
    df[new_col] = pd.qcut(
        df[col],
        q=quantiles,
        labels=labels,
        duplicates='drop'  # 경계값이 중복될 경우 자동으로 제거
    )
    return df


def split_col(df, col, sep, new_cols):
    """
    한 열을 지정한 구분자(sep)를 기준으로 분리하여 두 개의 새로운 열로 나누는 함수
    Parameters:
        df (pd.DataFrame): 원본 데이터프레임
        col (str): 분리할 원본 열 이름
        sep (str): 구분자 (예: '-', ',', ':', ' ')
        new_cols (list): 분리된 열의 새 이름 리스트 (예: ['first', 'second'])
    """
    df[new_cols] = df[col].str.split(sep, expand=True)
    return df



# 특정 열에서 지정한 값(condition_val)을 가진 행들의 다른 열(target_col)의 숫자 값을 변환
def condition_replace(df, condition_col, condition_val, target_col, factor):
    """
    특정 열에서 지정한 값(condition_val)을 가진 행들의 다른 열(target_col)의 숫자 값을 변환
    Parameters:
        df (pd.DataFrame): 원본 데이터프레임
        condition_col (str): 조건을 검사할 컬럼명
        condition_val (any): 해당 값과 일치하는 행을 찾기 위한 기준 값
        target_col (str): 값을 변환할 대상 컬럼명
        factor (float): 곱하거나 나눌 배수 (예: 시급을 월급으로 환산하려면 8*20 = 160)
    """
    condition = df[condition_col] == condition_val
    df.loc[condition, target_col] = df.loc[condition, target_col] * factor
    return df

def create_max_category_feature(df, cols, new_col, remove_text=""):
    """
    주어진 여러 열 중에서 가장 큰 값을 가진 열의 이름을 파생변수로 생성하는 일반화된 함수
    Parameters:
        df (pd.DataFrame): 입력 데이터프레임
        cols (list): 비교할 수치형 컬럼 리스트
        new_col (str): 생성할 파생변수 컬럼 이름
        remove_text (str): 열 이름에서 제거할 문자열 (선택)
    """
    df[new_col] = df[cols].idxmax(axis=1).str.replace(remove_text, '', regex=False)
    return df