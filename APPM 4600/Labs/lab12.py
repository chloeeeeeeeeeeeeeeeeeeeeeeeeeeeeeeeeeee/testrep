
"""
파일 이름: lab8.py
작성자: 클로이 정 (Chloe Chung)
작성일: 2025년 4월 8일
"""

import numpy as np
from numpy.polynomial.legendre import leggauss
import math


def composite_trapezoid(f, a, b, N):
    x = np.linspace(a, b, N)
    y = f(x)
    h = (b - a) / (N - 1)
    return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])

def composite_simpson(f, a, b, N):
    if N % 2 == 0:
        N += 1  # make N odd
    x = np.linspace(a, b, N)
    y = f(x)
    h = (b - a) / (N - 1)
    return (h / 3) * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])

def adaptive_quad(f, a, b, tol, method, N=5):
    def recursive_quad(f, a, b, tol, whole, method, N):
        mid = (a + b) / 2
        left = method(f, a, mid, N)
        right = method(f, mid, b, N)
        if np.abs(left + right - whole) < 3 * tol:
            return left + right
        else:
            return (recursive_quad(f, a, mid, tol / 2, left, method, N) +
                    recursive_quad(f, mid, b, tol / 2, right, method, N))

    whole = method(f, a, b, N)
    return recursive_quad(f, a, b, tol, whole, method, N)

def gaussian_quadrature(f, a, b, N=5):
    [x, w] = leggauss(N)
    t = 0.5 * (x + 1) * (b - a) + a
    return 0.5 * (b - a) * np.sum(w * f(t))


def test_func(x):
    return np.sin(1 / x)

a, b = 0.1, 2
tol = 1e-3
N = 5

trap_result = adaptive_quad(test_func, a, b, tol, composite_trapezoid, N)
simp_result = adaptive_quad(test_func, a, b, tol, composite_simpson, N)
gauss_result = adaptive_quad(test_func, a, b, tol, gaussian_quadrature, N)

print(f"Adaptive Trapezoid Result: {trap_result:.6f}")
print(f"Adaptive Simpson Result:   {simp_result:.6f}")
print(f"Adaptive Gaussian Result:  {gauss_result:.6f}")


