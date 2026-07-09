import numpy as np
import matplotlib.pyplot as plt


# =========================
# Optimizer 구현
# =========================

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params:
            params[key] -= self.lr * grads[key]


class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = {}

    def update(self, params, grads):

        if not self.v:
            for key in params:
                self.v[key] = np.zeros_like(params[key])

        for key in params:
            self.v[key] = (
                self.momentum * self.v[key]
                - self.lr * grads[key]
            )

            params[key] += self.v[key]


class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = {}

    def update(self, params, grads):

        if not self.h:
            for key in params:
                self.h[key] = np.zeros_like(params[key])

        for key in params:
            self.h[key] += grads[key] ** 2

            params[key] -= (
                self.lr * grads[key]
                / (np.sqrt(self.h[key]) + 1e-7)
            )


class Adam:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

        self.iter = 0
        self.m = {}
        self.v = {}

    def update(self, params, grads):

        if not self.m:
            for key in params:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

        self.iter += 1

        lr_t = (
            self.lr
            * np.sqrt(1.0 - self.beta2 ** self.iter)
            / (1.0 - self.beta1 ** self.iter)
        )

        for key in params:

            self.m[key] += (
                1 - self.beta1
            ) * (grads[key] - self.m[key])

            self.v[key] += (
                1 - self.beta2
            ) * (grads[key] ** 2 - self.v[key])

            params[key] -= (
                lr_t * self.m[key]
                / (np.sqrt(self.v[key]) + 1e-7)
            )


# =========================
# 최적화할 함수
# =========================

def f(x, y):
    return x**2 / 20.0 + y**2


def df(x, y):
    return x / 10.0, 2.0 * y


# =========================
# Optimizer 목록
# =========================

optimizers = {
    "SGD": SGD(lr=0.95),
    "Momentum": Momentum(lr=0.1),
    "AdaGrad": AdaGrad(lr=1.5),
    "Adam": Adam(lr=0.3),
}


# =========================
# 그래프 그리기
# =========================

x = np.arange(-10, 10, 0.1)
y = np.arange(-5, 5, 0.1)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)

init_pos = (-7.0, 2.0)
steps = 30


fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes = axes.flatten()

for idx, (name, optimizer) in enumerate(optimizers.items()):

    params = {
        "x": np.array(init_pos[0]),
        "y": np.array(init_pos[1]),
    }

    x_history = []
    y_history = []

    for i in range(steps):

        x_history.append(params["x"].copy())
        y_history.append(params["y"].copy())

        dx, dy = df(params["x"], params["y"])

        grads = {
            "x": np.array(dx),
            "y": np.array(dy),
        }

        optimizer.update(params, grads)

    ax = axes[idx]

    ax.contour(X, Y, Z, levels=30)
    ax.plot(
        x_history,
        y_history,
        marker='o'
    )

    ax.set_title(name)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid()


plt.tight_layout()
plt.show()