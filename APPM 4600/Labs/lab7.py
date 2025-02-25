#!/usr/bin/env python
# coding: utf-8

# In[7]:


"""
파일 이름: lab7.py
작성자: 글로이정 (Chloe Chung)
작성일: 2025년 2월 25일 
원래 제출 시간: 오전 9시 25분 
설명:
    이 스크립트는 다양한 보간(interpolation) 방법을 구현하고 비교합니다.
    포함된 보간 방법:
    - Vandermonde 행렬을 사용한 단항 보간(Monomial Interpolation)
    - 나눗셈 차분(Divided Differences)을 이용한 뉴턴 보간(Newton Interpolation)
    - BarycentricInterpolator를 이용한 라그랑주 보간(Lagrange Interpolation)
    
    스크립트는 각 방법의 정확도를 평가하고,
    서로 다른 노드 유형(일정 노드 및 Chebyshev 노드)을 사용한 결과와 오류 분석을 시각화합니다.

사용 방법:
    - 스크립트를 실행하여 다양한 N 값에 대한 보간 결과를 확인합니다.
    - 시각화에는 다음이 포함됩니다:
        - 보간 방법 비교
        - 각 방법의 절대 오차 그래프

의존성:
    - Python 3.x
    - NumPy
    - SciPy
    - Matplotlib

함수 설명:
    - f(x): 보간에 사용할 함수 1 / (1 + (10 * x) ** 2)를 정의합니다.
    - uni_nodes(N): [-1, 1] 구간에서 N개의 일정하게 분포된 노드를 생성합니다.
    - mon_interp(xnodes, ynodes, xev): Vandermonde 행렬을 사용하여 단항 보간을 수행합니다.
    - new_eval(xnodes, coef, xev): 주어진 x 값에서 뉴턴 다항식을 평가합니다.
    - cheshe_nodes(N): Chebyshev 노드를 생성합니다.
    - new_div_diff(xnodes, ynodes): 나눗셈 차분을 사용하여 뉴턴 보간 계수를 계산합니다.
    - lag_interp(xnodes, ynodes, xev): BarycentricInterpolator를 사용하여 라그랑주 보간을 수행합니다.
    - long_ass_function(Nvals, nt='uniform'): 보간 방법을 비교하고 결과를 시각화합니다.


버전:
    1.0
"""

import numpy as np 
import matplotlib.pyplot as mpl
from scipy.interpolate import BarycentricInterpolator as bci


def f(x):
    return 1 / (1 + (10 * x) ** 2) 

def uni_nodes(N):
    return np.linspace(-1, 1, N)


def mon_interp(xnodes, ynodes, xev):
    V = np.vander(xnodes, increasing=True)
    coeffs = np.linalg.solve(V, ynodes)
    return np.polyval(coeffs[::-1], xev)

def new_eval(xnodes, coef, xev):
    n = len(coef)
    result = np.zeros_like(xev, dtype=float)  
    for i in range(n - 1, -1, -1):
        result = result * (xev - xnodes[i]) + coef[i]  
    return result 
    
    
def new_div_diff(xnodes, ynodes):
    N = len(xnodes)  
    coef = np.copy(ynodes)
    for i in range(1, N):
        for j in range(N-1, i - 1, -1):
            coef[j] = (coef[j] - coef[j - 1]) / (xnodes[j] - xnodes[j - i])  
    return coef

def lag_interp(xnodes, ynodes, xev): 
    interpolator = bci(xnodes, ynodes)
    return interpolator(xev)

def cheshe_nodes(N):
    return np.cos((2 * np.arange(1, N + 1) - 1) * np.pi / (2 * N))


def long_ass_function(Nvals, nt='uniform'): #nt=node type 
    xev = np.linspace(-1, 1, 1000)
    tvals = f(xev) #tvals=true values
    
    for N in Nvals:
        xnodes = uni_nodes(N) if nt == 'uniform' else cheshe_nodes(N)
        ynodes = f(xnodes)
        
        mon_res = mon_interp(xnodes, ynodes, xev) 
        lag_res = lag_interp(xnodes, ynodes, xev) 
        
        new_coefs = new_div_diff(xnodes, ynodes)  
        new_res = new_eval(xnodes, new_coefs, xev) 
        
        mpl.plot(xev, tvals, label='True Function', linewidth=2, linestyle='dashed')  
        mpl.plot(xev, lag_res, label=f'Lagrange (N={N})')
        mpl.plot(xev, new_res, label=f'Newton (N={N})')
        mpl.scatter(xnodes, ynodes, color='pink', zorder=5, label='Interpolation Nodes')
        mpl.legend()
        mpl.title(f'Interpolation Methods Comparison (N={N})')
        mpl.xlabel('x')
        mpl.ylabel('f(x)')
        mpl.show()
        
        mpl.figure(figsize=(10, 6))
        mpl.semilogy(xev, np.abs(mon_res - tvals), label='Monomial Error')
        mpl.semilogy(xev, np.abs(lag_res - tvals), label='Lagrange Error')
        mpl.semilogy(xev, np.abs(new_res - tvals), label='Newton Error')
        mpl.legend()
        mpl.title(f'Absolute Error of Interpolation Methods (N={N})')
        mpl.xlabel('x')
        mpl.ylabel('Error')
        mpl.show()

        
    
long_ass_function(range(2, 11), nt='uniform')

long_ass_function(range(11, 21), nt='uniform')

long_ass_function(range(2, 21), nt='chebyshev')








# In[6]:





# In[ ]:




