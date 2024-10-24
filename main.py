import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 실제 값 (정육면체의 중심점, 즉 최종적으로 공기가 채워지는 지점)
actual_value = np.array([0, 0, 0])

# 모델이 예측하는 값(진공 상태가 풀리면서 공기가 정중앙으로 향하는 과정에서의 좌표)
def predict_position(t, total_time, cube_size):
    # 공기가 채워지는 비율 계산 (시간에 따라 공기가 이동하는 좌표를 예측)
    fill_fraction = t / total_time
    return np.array([cube_size/2 * (1 - fill_fraction), cube_size/2 * (1 - fill_fraction), cube_size/2 * (1 - fill_fraction)])

# 시뮬레이션 설정
cube_size = 10  # 정육면체 한 변의 길이
time_steps = 20  # 시뮬레이션 시간 단계
max_time = 10  # 공기가 완전히 채워지는 시간

# 차이 계산용 데이터 저장
x_diffs, y_diffs, z_diffs = [], [], []

# 그래프 설정
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(2, 1, 1)

# 3D 시각화 - 공기가 채워지는 과정 애니메이션
ax2 = fig.add_subplot(2, 1, 2, projection='3d')
ax2.set_xlim([-cube_size / 2, cube_size / 2])
ax2.set_ylim([-cube_size / 2, cube_size / 2])
ax2.set_zlim([-cube_size / 2, cube_size / 2])
ax2.set_title('Filling Simulation in 3D')

point, = ax2.plot([], [], [], 'ro', markersize=10)  # 공기 채워지는 점

# 애니메이션 초기화
def init():
    point.set_data([], [])
    point.set_3d_properties([])
    return point,

# 막대 그래프 그리기 함수
def update(frame):
    predicted_value = predict_position(frame, max_time, cube_size)
    diff = actual_value - predicted_value
    x_diffs.append(abs(diff[0]))
    y_diffs.append(abs(diff[1]))
    z_diffs.append(abs(diff[2]))
    
    # 막대 그래프 업데이트
    ax1.clear()
    ax1.bar(range(frame + 1), x_diffs, alpha=0.6, label='x-axis diff', color='b')
    ax1.bar(range(frame + 1), y_diffs, alpha=0.6, label='y-axis diff', color='g')
    ax1.bar(range(frame + 1), z_diffs, alpha=0.6, label='z-axis diff', color='r')
    ax1.set_title('Difference between Actual and Predicted Values for Each Axis')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Difference')
    ax1.legend()
    ax1.grid(True)
    
    # 3D 공기 채워지는 과정 업데이트
    point.set_data_3d([predicted_value[0]], [predicted_value[1]], [predicted_value[2]])  # 좌표를 리스트로 전달
    return point,

# 애니메이션 생성
ani = FuncAnimation(fig, update, frames=range(time_steps), init_func=init, blit=False, repeat=False)

plt.tight_layout()
plt.show()
