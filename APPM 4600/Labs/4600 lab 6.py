#!/usr/bin/env python
# coding: utf-8

# In[1]:


#작성자: 클로이 정 (Chloe Chung)
#원래 제출 시간: 2025년 2월 18일 오전 9시 37분 

# sol h=0.0100000000 | Forward: -0.9999833334 | Centered: -0.9999833334
# h=0.0050000000 | Forward: -0.9999958333 | Centered: -0.9999958333
# h=0.0025000000 | Forward: -0.9999989583 | Centered: -0.9999989583
# h=0.0012500000 | Forward: -0.9999997396 | Centered: -0.9999997396
# h=0.0006250000 | Forward: -0.9999999349 | Centered: -0.9999999349
# h=0.0003125000 | Forward: -0.9999999837 | Centered: -0.9999999837
# h=0.0001562500 | Forward: -0.9999999959 | Centered: -0.9999999959
# h=0.0000781250 | Forward: -0.9999999990 | Centered: -0.9999999990
# h=0.0000390625 | Forward: -0.9999999997 | Centered: -0.9999999997
# h=0.0000195313 | Forward: -0.9999999999 | Centered: -0.9999999999
# Slacker Newton solution: [ 0.99860694 -0.10553049]
# Newton with Approximate Jacobian solution: [ 0.99860694 -0.10553049]
# Hybrid Newton Solution: [ 0.99860694 -0.10553049]

import numpy as np 
from scipy.linalg import inv

def f(x):
    return np.cos(x)
    
def fdiff(f, x, h):
    return (f(x+h)-f(x))/h

def cdoff(f, x, h):
    return (f(x+h)-f(x-h))/(2*h)

h_vals = 0.01 * 2.0 ** (-np.arange(0,10))
x = np.pi/2

fapprx = [fdiff(f, x, h) for h in h_vals]
apprx = [cdoff(f, x, h) for h in h_vals]

for h, fwd, ctr in zip(h_vals, fapprx, apprx):
    print(f"h={h:.10f} | Forward: {fwd:.10f} | Centered: {ctr:.10f}")
    
def f_vec(x):
    return np.array([4*x[0]**2 + x[1]**2 -4, x[0] + x[1] - np.sin(x[0] - x[1])])

def jacobian(x):
    J = np.array([[8*x[0], 2*x[1]], [1 - np.cos(x[0] - x[1]), 1 + np.cos(x[0] - x[1])]])
    return J

def slacker_newton(f, jacobian, x0, tol=1e-10, max_iter=50):
    x = np.array(x0, dtype=float)
    J_inv = inv(jacobian(x))
    
    for _ in range(max_iter):
        delta_x = -J_inv @ f(x)
        x_new = x + delta_x
        
        if np.linalg.norm(delta_x) < tol:
            return x_new
        
        if np.linalg.norm(delta_x) > 0.1:
            J_inv = inv(jacobian(x_new))
        
        x = x_new
    
    return x

x0 = np.array([1.0, 0.0])
sol = slacker_newton(f_vec, jacobian, x0)
print("Slacker Newton solution:", sol)

def apprx_jac(f, x, h=1e-7):
    n = len(x)
    J_approx = np.zeros((n, n))
    
    for j in range(n):
        h_vec = np.zeros(n)
        h_vec[j] = h * abs(x[j]) if x[j] != 0 else h
        J_approx[:, j] = (f(x + h_vec) - f(x - h_vec)) / (2 * h_vec[j])
    
    return J_approx

def n_apprx_jac(f, x0, tol=1e-10, max_iter=50, h=1e-7):
    x = np.array(x0, dtype=float)
    
    for _ in range(max_iter):
        J_approx = apprx_jac(f, x, h)
        delta_x = -np.linalg.solve(J_approx, f(x))
        x_new = x + delta_x
        
        if np.linalg.norm(delta_x) < tol:
            return x_new
        x = x_new
    return x

sol_apprx = n_apprx_jac(f_vec, x0)
print("Newton with Approximate Jacobian solution:", sol_apprx)

def hybrid_newton(f, x0, tol=1e-10, maxi=50, h0=1e-13):
    x = np.array(x0, dtype=float)
    h = h0
    
    for _ in range(maxi):
        J_apprx = apprx_jac(f, x, h)
        delta_x = -np.linalg.solve(J_apprx, f(x))
        xn = x + delta_x
        
        if np.linalg.norm(delta_x) < tol:
            return xn
        
        h /= 2
    
        x = xn
    return x

sol_hybrid = hybrid_newton(f_vec, x0)
print("Hybrid Newton Solution:", sol_hybrid)


# In[ ]:




