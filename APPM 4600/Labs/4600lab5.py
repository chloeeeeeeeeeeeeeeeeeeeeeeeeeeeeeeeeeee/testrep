#!/usr/bin/env python
# coding: utf-8

# In[2]:


#author: Chloe Chung

import numpy as np


def f(x):
    return np.exp(x**2 + 7*x - 30) - 1

def df(x):
    return (2*x + 7) * np.exp(x**2 + 7*x - 30)


def newton_convergence(x):
    return abs(1 - (df(x) * f(x)) / (df(x)**2))


def bisection(f, a, b, tol=1e-6):
    if f(a) * f(b) > 0:
        raise ValueError("function must have opposite signs at endpoint")
    
    while (b - a) / 2 > tol:
        midpoint = (a + b) / 2
        if f(midpoint) == 0:
            return midpoint
        elif f(a) * f(midpoint) < 0:
            b = midpoint
        else:
            a = midpoint
        #basin check!!!!!!
        if newton_convergence(midpoint) < 1:
            return midpoint
    return (a + b) / 2

def hybridrf(f, df, a, b, tol=1e-6, max_iter=1000):
    mid = bisection(f, a, b, tol)
    x = mid
    
    for _ in range(max_iter):
        if abs(f(x)) < tol:
            return x
        
        if df(x) != 0:
            x_new = x - f(x) / df(x)
            if abs(x_new - x) < tol:
                return x_new
            x = x_new
        else:
            break  #division by zero
        
    return x

# 6. applying Methods to f(x)
a, b = 2, 4.5
x0 = 4.5

bisection_root = bisection(f, a, b)
print(f"Bisection Root: {bisection_root}")

#newton's method
def newton(f, df, x0, tol=1e-6, max_iter=1000):
    x = x0
    for _ in range(max_iter):
        if abs(f(x)) < tol:
            return x
        x_new = x - f(x) / df(x)
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return x

newton_root = newton(f, df, x0)
print(f"Newton Root: {newton_root}")

#hybrid method
hybrid_root = hybridrf(f, df, a, b)
print(f"Hybrid Root: {hybrid_root}")


# In[ ]:




