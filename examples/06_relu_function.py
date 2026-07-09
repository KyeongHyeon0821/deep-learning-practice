import numpy as np
import matplotlib.pyplot as plt

# x 범위
x = np.linspace(-10, 10, 100)

# ReLU 함수
def relu(x):
    return np.maximum(0, x)

# y 값 계산
y = relu(x)

# 그래프 그리기
plt.plot(x, y)
plt.title('ReLU Function')
plt.xlabel('x')
plt.ylabel('ReLU(x)')
plt.grid(True)
plt.show()