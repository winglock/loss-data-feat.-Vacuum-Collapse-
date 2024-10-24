import numpy as np
import matplotlib.pyplot as plt

# 실제 값 (정육면체의 중심점, 즉 최종적으로 공기가 채워지는 지점)
actual_value = np.array([0, 0, 0])

# 모델이 예측하는 값(공기가 채워지는 과정에서의 좌표)
def predict_position(t, total_time, cube_size):
    # 공기가 채워지는 비율 계산 (시간에 따라 예측 좌표 변화)
    fill_fraction = t / total_time
    return np.array([fill_fraction * cube_size / 2, fill_fraction * cube_size / 2, fill_fraction * cube_size / 2])

# 시뮬레이션 설정
cube_size = 10  # 정육면체 한 변의 길이
time_steps = 10  # 시뮬레이션 시간 단계
x_diffs, y_diffs, z_diffs = [], [], []  # 각 좌표별 차이 저장

# 시뮬레이션 진행
for t in range(time_steps):
    # 예측 값 계산
    predicted_value = predict_position(t, time_steps, cube_size)
    
    # 각 좌표축별 실제 값과 예측 값의 차이 계산
    diff = actual_value - predicted_value
    x_diffs.append(diff[0])
    y_diffs.append(diff[1])
    z_diffs.append(diff[2])

# 막대 그래프 그리기
time_range = range(time_steps)

plt.figure(figsize=(10, 6))

# x축 차이
plt.bar(time_range, x_diffs, alpha=0.6, label='x-axis diff', color='b')

# y축 차이
plt.bar(time_range, y_diffs, alpha=0.6, label='y-axis diff', color='g')

# z축 차이
plt.bar(time_range, z_diffs, alpha=0.6, label='z-axis diff', color='r')

# 그래프 설정
plt.title('Difference between Actual and Predicted Values for Each Axis')
plt.xlabel('Time Steps')
plt.ylabel('Difference')
plt.legend()
plt.grid(True)

plt.show()
