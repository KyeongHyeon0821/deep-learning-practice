import numpy as np
import matplotlib.pyplot as plt

# x 범위 설정
x = np.linspace(-10, 10, 100)

# 시그모이드 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 계단 함수 (Heaviside)
def step_function(x):
    return np.where(x >= 0, 1, 0)

# 그래프 그리기
plt.figure()

plt.plot(x, sigmoid(x), label='Sigmoid Function')
plt.plot(x, step_function(x), label='Step Function')

plt.title('Sigmoid vs Step Function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()

plt.show()