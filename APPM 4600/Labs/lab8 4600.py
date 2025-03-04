#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
파일 이름: lab8.py
작성자: 클로이 정 (Chloe Chung)
작성일: 2025년 3월 4일
설명:
    이 스크립트는 선형 및 3차 스플라인 보간법을 구현하고 비교합니다.
    포함된 보간 방법:
    - 선형 스플라인 보간 (Linear Spline Interpolation)
    - 자연 스플라인 보간 (Natural Cubic Spline Interpolation)
    
    스크립트는 각 방법의 정확도를 평가하고, 시각화를 통해 보간 성능을 분석합니다.

사용 방법:
    - 스크립트를 실행하여 선형 및 3차 스플라인 보간 결과를 확인합니다.
    - 시각화에는 다음이 포함됩니다:
        - 원래 함수와 보간 결과 비교
        - 절대 오차 그래프

의존성:
    - Python 3.x
    - NumPy
    - Matplotlib

함수 설명:
    - eval_line(x0, f0, x1, f1, alpha): 두 점을 지나는 직선을 평가합니다.
    - eval_lin_spline(xeval, Neval, a, b, f, Nint): 선형 스플라인을 평가합니다.
    - create_natural_spline(yint, xint, N): 자연 스플라인 계수를 계산합니다.
    - eval_local_spline(x, xi, xip, Mi, Mip, Ci, Di): 개별 구간에서 3차 스플라인을 평가합니다.
    - eval_cubic_spline(xeval, Neval, xint, Nint, M, C, D): 전체 구간에서 3차 스플라인을 평가합니다.
    - driver_linear(): 선형 스플라인 보간을 실행하고 결과를 시각화합니다.
    - driver_cubic(): 자연 스플라인 보간을 실행하고 결과를 시각화합니다.

버전:
    1.0
"""





import numpy as np
import matplotlib.pyplot as mpl

def eval_line(x0, f0, x1, f1, alpha):
    return f0 + (f1 - f0) * (alpha - x0) / (x1 - x0)

def eval_lin_spline(xeval, Neval, a, b, f, Nint):
    xint = np.linspace(a, b, Nint + 1)
    yeval = np.zeros(Neval)
    
    for j in range(Nint):
        atmp, btmp = xint[j], xint[j + 1]
        fa, fb = f(atmp), f(btmp)
        ind = np.where((xeval >= atmp) & (xeval <= btmp))
        xloc = xeval[ind]
        
        yloc = np.array([eval_line(atmp, fa, btmp, fb, x) for x in xloc])
        yeval[ind] = yloc
    
    return yeval

def driver_linear():
    f = lambda x: np.exp(x)
    a, b = 0, 1
    Neval = 100
    xeval = np.linspace(a, b, Neval)
    Nint = 10
    
    yeval = eval_lin_spline(xeval, Neval, a, b, f, Nint)
    fex = f(xeval)
    
    mpl.figure()
    mpl.plot(xeval, fex, 'ro-', label='Exact Function')
    mpl.plot(xeval, yeval, 'bs-', label='Linear Spline')
    mpl.legend()
    mpl.show()

def create_natural_spline(yint, xint, N):
    h = np.diff(xint)
    b = np.zeros(N - 1)
    A = np.zeros((N - 1, N - 1))
    
    for i in range(1, N):
        if i > 1:
            A[i - 1, i - 2] = h[i - 2]
        A[i - 1, i - 1] = 2 * (h[i - 2] + h[i - 1])
        if i < N - 1:
            A[i - 1, i] = h[i - 1]
        b[i - 1] = 6 * ((yint[i + 1] - yint[i]) / h[i] - (yint[i] - yint[i - 1]) / h[i - 1])
    
    M = np.zeros(N + 1)
    M[1:N] = np.linalg.solve(A, b)
    
    C = (yint[1:] - yint[:-1]) / h - h * (M[1:] + 2 * M[:-1]) / 6
    D = yint[:-1]
    
    return M, C, D

def eval_local_spline(x, xi, xip, Mi, Mip, Ci, Di):
    hi = xip - xi
    return ((Mi * (xip - x)**3 + Mip * (x - xi)**3) / (6 * hi) +
            (Ci * (x - xi) + Di))

def eval_cubic_spline(xeval, Neval, xint, Nint, M, C, D):
    yeval = np.zeros(Neval)
    
    for j in range(Nint):
        atmp, btmp = xint[j], xint[j + 1]
        ind = np.where((xeval >= atmp) & (xeval <= btmp))
        xloc = xeval[ind]
        
        yloc = eval_local_spline(xloc, atmp, btmp, M[j], M[j + 1], C[j], D[j])
        yeval[ind] = yloc
    
    return yeval

def driver_cubic():
    f = lambda x: np.exp(x)
    a, b = 0, 1
    Nint = 3
    xint = np.linspace(a, b, Nint + 1)
    yint = f(xint)
    Neval = 100
    xeval = np.linspace(a, b, Neval)
    
    M, C, D = create_natural_spline(yint, xint, Nint)
    yeval = eval_cubic_spline(xeval, Neval, xint, Nint, M, C, D)
    fex = f(xeval)
    
    mpl.figure()
    mpl.plot(xeval, fex, 'ro-', label='Exact Function')
    mpl.plot(xeval, yeval, 'bs--', label='Cubic Spline')
    mpl.legend()
    mpl.show()

driver_linear()
driver_cubic()


# In[ ]:




