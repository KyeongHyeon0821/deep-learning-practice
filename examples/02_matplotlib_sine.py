import numpy as np
import matplotlib.pyplot as plt

# x 값 생성 (0부터 2π까지 0.01 간격)
x = np.arange(0, 2 * np.pi, 0.01)

# y 값 계산 (sin 함수)
y = np.sin(x)

# 그래프 그리기
plt.plot(x, y, label='sin(x)')  # sin(x) 그래프
plt.title('Sine Function')      # 제목
plt.xlabel('x')                 # x축 라벨
plt.ylabel('sin(x)')            # y축 라벨
plt.legend()                    # 범례 표시
plt.grid(True)                  # 격자 표시
plt.show()                      # 그래프 출력