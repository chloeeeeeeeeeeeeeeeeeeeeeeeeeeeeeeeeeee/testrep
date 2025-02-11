#!/usr/bin/env python
# coding: utf-8


#작성자: 클로이 정 (Chloe Chung)
#원래 제출 시간: 2025년 2월 11일 오전 9시 30분 



import numpy as np

#function
def f(x):
    return np.exp(x**2 + 7*x - 30) - 1

#d/dx of f(x)
def df(x):
    return (2*x + 7) * np.exp(x**2 + 7*x - 30)

#bisection method w/iteration tracking
def bisection(f, a, b, tol=1e-6):
    if f(a) * f(b) > 0:
        raise ValueError("Function must have opposite signs at endpoints.")
    
    bi_count = 0
    while (b - a) / 2 > tol:
        bi_count += 1
        midpoint = (a + b) / 2
        if f(midpoint) == 0 or abs(f(midpoint)) < tol:
            return midpoint, bi
        elif f(a) * f(midpoint) < 0:
            b = midpoint
        else:
            a = midpoint
        #basin check!!!! (I had to google what this meant and tbh idk if I did it right)
        if newton_convergence_condition(midpoint) < 1:
            return midpoint, bi_count
    return (a + b) / 2, bi_count

#function to check condition for newton's method
def newton_convergence(x):
    return abs(1 - (df(x) * f(x)) / (df(x)**2))

#newton's method w/iteration tracking
def newton(f, df, x0, tol=1e-6, max_iter=100):
    x = x0
    ni_count = 0
    for _ in range(max_iter):
        ni_count += 1
        if abs(f(x)) < tol:
            return x, ni_count
        if df(x) == 0:
            break  #avoid div by 0
        x_new = x - f(x) / df(x)
        if abs(x_new - x) < tol:
            return x_new, ni
        x = x_new
    return x, ni

#hybrid method
def hybrid(f, df, a, b, tol=1e-6, max_iter=100):
    ti_count = 0  #total iterations
    
    #bisection until midpoint is in Newton's basin
    mid, bi_it = bisection_method(f, a, b, tol)
    ti_count += bi_it  #add bisection iterations to total

    #once in midpoint apply Newton's
    x = mid
    ne_it = 0
    for _ in range(max_iter):
        ti_count += 1  #track newton's iterations as part of total
        ne_it += 1
        if abs(f(x)) < tol:
            return x, ti_count
        if df(x) == 0:
            break  #avoiding division by 0
        x_new = x - f(x) / df(x)
        if abs(x_new - x) < tol:
            return x_new, ti_count
        x = x_new
    
    return x, ti_count

#testing methods on given f(x) in part 6
a, b = 2, 4.5
x0 = 4.5

#bisection roots and iterations
bisection_rt, bisection_is = bisection_method(f, a, b)
print(f"Bisection Root: {bisection_rt}, Iterations: {bisection_is}")

#newton's method root and iterations
newton_rt, newton_is = newton_method(f, df, x0)
print(f"\nNewton Root: {newton_rt}, Iterations: {newton_is}")

#hybrid root and iterations
hybrid_rt, hybrid_is = hybrid_root_finder(f, df, a, b)
print(f"\nHybrid Root: {hybrid_rt}, Total Iterations: {hybrid_is}")





