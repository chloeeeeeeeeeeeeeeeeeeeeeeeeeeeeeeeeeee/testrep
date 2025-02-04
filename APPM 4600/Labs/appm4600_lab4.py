#!/usr/bin/env python
# coding: utf-8

# ### Prelab code 

# In[20]:


import numpy as np
import math 

def order_of_conv(approximations, fp=1.5):
    errors = np.abs(np.array(approximations) - fp)

    if len(errors) < 3:
        raise ValueError("Need at least 3 approximations to estimate the order of convergence.")
    
    orders = []
    for n in range(2, len(errors)):
        if errors[n-1] == 0 or errors[n-2] == 0:
            continue  # handle division by zero
        
        alpha = np.log(errors[n] / errors[n-1]) / np.log(errors[n-1] / errors[n-2])
        orders.append(alpha)
    
    # computing orders for stability
    average_order = np.mean(orders)
    
    return average_order,orders



# 

# In[23]:


def fpi(g, p0, tol=1e-10, max_iter=100):
    approximations = [p0]
    
    for _ in range(max_iter):
        p1 = g(p0)
        approximations.append(p1)
        
        if abs(p1 - p0) < tol:
            break
        
        p0 = p1
    
    return approximations



def g(x):
    return math.sqrt(10/(x+4))

fpi(g, 1.5)


# ### 3.1 - 3.2 

# In[22]:


def aitken(approximations):
    aitken_seq = []
    for n in range(len(approximations) - 2):
        p_n, p_n1, p_n2 = approximations[n], approximations[n+1], approximations[n+2]
        num = (p_n1 - p_n)**2
        denom = p_n2 - 2*p_n1 + p_n
        
        if denom == 0:
            continue  # handle division by zero
        
        p_hat = p_n - num / denom
        aitken_seq.append(p_hat)
    
    return aitken_seq



# ### 3.3 method 

# In[27]:


def steffensen(g, p0, tol=1e-5, max_iter=100):
    approximations = [p0]
    
    for _ in range(max_iter):
        a = p0
        b = g(a)
        c = g(b)
        
        denom = c - 2*b + a
        if denom == 0:
            break  # division by 0
        
        p1 = a - ((b - a)**2) / denom
        approximations.append(p1)
        
        if abs(p1 - p0) < tol:
            break
        
        p0 = p1
    
    return approximations

### 3.4.3
def g(x):
    return math.sqrt(10/(x+4))

steffensen(g, 1.5)

apps = [1.5, 1.3652652239572602, 1.3652300134165856, 1.3652300134140969]

order_of_conv(apps, 1.3652300134140976)


# ### Excercises 3.4
# #### 1
# inputs:  g(function), p0(initial guess), tolerance, max_iterations
# outputs: list of approximations 
# 
# for val in range of max iterations: 
#     a=p0
#     b=g(a)
#     c=g(b)
#     
#     denom = c - 2b + a
#      if denom = 0 break (div by 0)
#     
#     p1 = (a - b)^2 / denom
#     approx.append(p1)
#     
#     if |p1 - p0| < tol: 
#         break (found approx) 
#     set p0 to p1 
#  return approximations
#  
# #### 2
# see code above 
#  
# #### 3
# 1.5, 1.3652652239572602, 1.3652300134165856, 1.3652300134140969
# 
# #### 4
# It looks like the order of convergence was initially quadratic but gradually decreased as iterations increased. 

# In[ ]:



    

