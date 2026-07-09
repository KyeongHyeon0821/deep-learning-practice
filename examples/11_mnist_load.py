import sys, os
sys.path.append(os.pardir)

from dataset.mnist import load_mnist

# MNIST 데이터 로드
(x_train, t_train), (x_test, t_test) = load_mnist(
    normalize=True,      # 0~255 → 0.0~1.0
    flatten=True,        # 28x28 → 784
    one_hot_label=False  # 정답을 숫자로
)

# 확인 출력
print("x_train shape:", x_train.shape)
print("t_train shape:", t_train.shape)
print("x_test shape:", x_test.shape)
print("t_test shape:", t_test.shape)