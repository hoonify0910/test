
병원 예약 환자의 노쇼 가능성을 예측하기 위해 데이터를 전처리했습니다. 
ScheduledDay랑 AppointmentDay는 문자열이기 때문에 날짜로 변환하였다. 
gender,No-show 값이 문자열이라 0(no),1(yes)로 변환 필요
PatientId, AppointmentID은 불필요한 칼럼이라 생각해 삭제


SMS 여부에 따른 노쇼율 비교가 의미있을지? 
0    0.175
1    0.238
Name: No-show, dtype: float64
문자 수신 여부별 평균 노쇼율을 비교함. 
유의미한 차이가 나지 않음을 확인.

refactoring
해당 데이터에 불필요한 함수를 그대로 둠.
예약일과 진료일 간의 기다린 기간이 음수, 1000일 이상인 이상치를 확인만하고 처리하지 않음.
순서와 흐름에 관계없이 작성함.
