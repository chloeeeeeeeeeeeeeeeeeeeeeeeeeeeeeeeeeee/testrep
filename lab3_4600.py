#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
import math
import matplotlib.pyplot as plt

def driver(f=None, a=None, b=None, tol=None):
    #f = NaN
    #a = NaN
    #b = NaN
    #tol = NaN

    astar, ier, errors = bisection(f, a, b, tol)

    print('The approximate root is', astar)
    print('The error message reads:', 'Success' if ier == 0 else 'Failed')
    print('f(astar) =', f(astar))

    plt.figure(figsize=(8, 6))
    plt.plot(errors, marker='o')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Error (log scale)')
    plt.title('Convergence of Bisection Method')
    plt.grid(True)
    plt.show()

def bisection(f, a, b, tol):
    fa = f(a)
    fb = f(b)
    errors = []  

    if fa * fb > 0:
        ier = 1
        astar = a
        return astar, ier, errors

    if fa == 0:
        return a, 0, errors
    if fb == 0:
        return b, 0, errors

    ier = 0
    while abs(b - a) > tol:
        d = 0.5 * (a + b)  
        fd = f(d)

        errors.append(abs(b - a))

        if fd == 0:
            return d, ier, errors
        elif fa * fd < 0:
            b = d  
        else:
            a = d  
            fa = fd

    astar = 0.5 * (a + b)
    errors.append(abs(b - a))  

    return astar, ier, errors

#1a: approximate root 0.9999, root found, 
#driver(lambda x: x**2 * (x-1), 0.5, 2, 1e-6)

#1b: root was not found because both f(a) and f(b) have the same sign
#driver(lambda x: x**2 * (x-1), -1, 0.5, 1e-6)

#1c: root was found, 0.9999
#driver(lambda x: x**2 * (x-1), -1, 2, 1e-6)

#2a: root was found, roughly 2.2087
#driver(lambda x: x-1 * (x-3) * (x - 5), 0, 2.4, 1e-5)

#2b: no root was found
#driver(lambda x: (x-1)**2 * (x-3), 0, 2, 1e-5)

#2c: a root was found at 0
driver(lambda x: math.sin(x), 0.5, math.pi(), 1e-5)


# In[16]:





# In[ ]:




