import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, tanh, diff, lambdify

# 定义函数
x = symbols('x')
f = 1 + 0.5 * tanh(2 * x)

# 计算解析解
f_prime = diff(f, x)
f_prime_lambdified = lambdify(x, f_prime, 'numpy')

# 中心差分法实现
def central_difference(f, a, b, h):
    n = int((b - a) / h) + 1
    x = np.linspace(a, b, n)
    df = np.zeros_like(x)
    for i in range(1, n-1):
        df[i] = (f(x[i+1]) - f(x[i-1])) / (2 * h)
    return x, df

# Richardson外推法实现
def richardson(f, x, n, h):
    d = np.zeros((n+1, n+1), float)
    for i in range(n+1):
        d[i, 0] = (f(x+h) - f(x-h)) / (2 * h)
    for j in range(1, n+1):
        for i in range(n-j+1):
            d[i, j] = d[i, j-1] + (d[i, j-1] - d[i-1, j-1]) / (4**j - 1)
    return d

# 计算中心差分法和Richardson外推法的导数
a, b = -2, 2
h_values = [0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6]
central_diff_errors = []
richardson_errors = []

for h in h_values:
    x = a + h / 2
    x_values, central_diff = central_difference(lambda z: f_prime_lambdified(z), a, b, h)
    central_diff_errors.append(np.max(np.abs(central_diff - f_prime_lambdified(x_values))))
    
    x_richardson = np.arange(a, b + h, h)
    d = richardson(lambda z: f_prime_lambdified(z), x_richardson, 5, h)
    richardson_errors.append(np.max(np.abs(d[-1, :] - f_prime_lambdified(x_richardson))))

# 绘制误差与步长的关系图
plt.figure(figsize=(10, 6))
plt.loglog(h_values, central_diff_errors, label='Central Difference', marker='o')
plt.loglog(h_values, richardson_errors, label='Richardson Extrapolation', marker='o')
plt.xlabel('Step Size (h)')
plt.ylabel('Error')
plt.title('Error vs Step Size')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()

# 绘制导数图
x_values = np.linspace(a, b, 400)
plt.figure(figsize=(10, 6))
plt.plot(x_values, f_prime_lambdified(x_values), label='Analytical Derivative', color='black')
plt.plot(x_values[1:-1], central_diff[1], label='Central Difference', linestyle='--', color='blue')
plt.plot(x_values, richardson_diff[-1, :], label='Richardson Extrapolation', linestyle='--', color='red')
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.title('Comparison of Derivative Approximations')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()
