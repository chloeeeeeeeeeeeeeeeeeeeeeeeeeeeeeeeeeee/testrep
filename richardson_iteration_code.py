"""
파일 이름: richardson_iteration.py
작성자: 글로이정 (Chloe Chung)
작성일: 2025년 3월 23일
설명:
    이 스크립트는 Richardson 반복법(Richardson Iteration)을 사용하여 선형 시스템 Ax = b를 푸는 방법을 구현합니다.
    Richardson 방법은 간단한 반복법이며, A가 대칭 양의 정부호(SPD) 행렬일 때 수렴 보장이 있습니다.

    스크립트는 서로 다른 조건 수(condition number)를 가진 SPD 행렬을 생성하고,
    각 경우에 대한 수렴 속도를 시각화합니다.

사용 방법:
    - 이 스크립트를 실행하면 3가지 조건 수에 대해 Richardson 알고리즘이 자동으로 실행됩니다.
    - 각 테스트는 남은 잔차(norm of residuals)를 로그 스케일 그래프로 보여줍니다.

의존성:
    - Python 3.x
    - NumPy
    - Matplotlib

함수 설명:
    - richardson_iteration(A, b, alpha, ...): Richardson 알고리즘을 사용하여 Ax = b를 반복적으로 풉니다.
    - generate_test_matrix(n, condition_number): 주어진 조건 수를 가지는 SPD 행렬을 생성합니다.
    - run_richardson_tests(): 다양한 조건 수와 alpha 값에 대해 Richardson 반복을 실행하고 결과를 시각화합니다.

버전:
    1.0
    
Description:
    This script implements the Richardson Iteration method to solve linear systems of the form Ax = b.
    It's a simple iterative solver that works best when A is SPD (symmetric positive definite).

    The script generates SPD matrices with different condition numbers and
    visualizes how quickly Richardson converges in each case.

Use:
    - Just run the script. It'll automatically test the method with 3 different condition numbers.
    - Each test shows how the residuals shrink over time using a log-scale plot.

Dependencies:
    - Python 3.x
    - NumPy
    - Matplotlib

Function Descriptions:
    - richardson_iteration(A, b, alpha, ...): runs Richardson Iteration to solve Ax = b
    - generate_test_matrix(n, condition_number): builds an SPD matrix with a given condition number
    - run_richardson_tests(): runs tests for various matrices and plots convergence behavior
"""


import numpy as np
import matplotlib.pyplot as mpl

def richardson_iteration(A, b, x0=None, tol=1e-10, max_iter=1000):
    n = A.shape[0]
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    residuals = []
    for _ in range(max_iter):
        r = b - A @ x
        residual = np.linalg.norm(r)
        residuals.append(residual)
        if residual < tol:
            break

        alpha = (r @ r) / (r @ (A @ r))
        x = x + alpha * r

    return x, residuals

def gen_test_mat(n, con_num=10):
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    eigs = np.linspace(1, con_num, n)
    A = Q.T @ np.diag(eigs) @ Q
    return A

def run_tests():
    n = 100
    con_nums = [5, 50, 500]

    for cond in con_nums:
        A = gen_test_mat(n, con_num=cond)
        b = np.random.randn(n)
        x_true = np.linalg.solve(A, b)
        x_approx, residuals = richardson_iteration(A, b)

        print(f"Condition number: {cond}, Final residual: {residuals[-1]:.2e}")
        mpl.semilogy(residuals, label=f"cond={cond}")

    mpl.xlabel("Iteration")
    mpl.ylabel("Residual Norm")
    mpl.title("Modified Richardson Iteration Convergence")
    mpl.legend()
    mpl.grid(True)
    mpl.show()

if __name__ == "__main__":
    run_tests()
