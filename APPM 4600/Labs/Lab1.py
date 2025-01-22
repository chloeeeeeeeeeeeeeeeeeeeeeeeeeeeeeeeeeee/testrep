#!/usr/bin/env python
# coding: utf-8

# # Lab 1
# **Name your notebook** `Lab1.ipynb`. At the end of class, you will convert the notebook to `.py` format and upload to Gradescope for autograding.
# 
# ---

# *Be sure to* `import math` *before beginning these exercises.  
# Answers shown are rounded to 6 decimal digits. It is not necessary for you to round your answers.*

# In[2]:


import math


# ### Basic Operations
# 
# The fraction $\frac{22}{7}$ can be used as an approximation for $\pi$. What is the absolute difference of the two values? Use the `abs` function and the `math.pi` constant.
# 
#  Answer: `0.001264`

# In[4]:


abs(math.pi-(22/7))


# $(3,4,5)$ is a Pythagorean triple: $3^2 + 4^2 = 5^2$. Another triple is $(19, b, 181)$. Find the value of $b$.
# 
#  Answer: `180.0`

# In[5]:


math.sqrt(181**2 - 19**2)


# ### Quotients and Remainders
# 
# Find the quotient when $1234$ is divided by $19$. Use the `//` operator.
# 
#  Answer: `64`

# In[6]:


1234//19


# Find the remainder when $1234$ is divided by $19$. Use the `%` operator.
# 
#  Answer: `18`
#  

# In[7]:


1234%19


# Use the `%` operator to calculate the units digit of $17^{19}$.
# 
#  Answer: `3`
#  

# In[18]:


(17**19)%10


# Use the `%` operator to calculate the last three digits of $17^{19}$.
# 
#  Answer: `153`

# In[20]:


(17**19)%1000


# ### Exponentials and Logarithms
# 

# Find the value of $\pi^e$.
# 
#  Answer: `22.459158`

# In[21]:


math.pi**math.e


# The expression $\ln \left(16\cdot 7^3 \right)$ simplifies to $4\ln(2) + 3\ln(7)$. Check that they are equal.

# In[22]:


math.log(16*(7**3))


# In[24]:


4*math.log(2) + 3*math.log(7)


# ### Trigonometry
# 
# Calculate the degree measure of $1$ radian. Use the `math.degrees` function.
# 
#  Answer: `57.295780`

# In[25]:


math.degrees(1)


# Calculate the value of $\sec(75^\circ)$. Use the `math.cos` and `math.radians` functions. (The `math` module does not include functions for $\sec x, \csc x,$ or $\cot x$.)
# 
#  Answer: `3.863703`

# In[29]:


1/math.cos(math.radians(75))


# The value of $\tan(\pi/2)$ is undefined. What happens if the angle is close to $\pi/2$? Evaluate the `math.tan` function at the angles $\pi/2 \pm 10^{-6}$. Use scientific notation for $10^{-6}$.
# 
# Answers: `-1000000.000143` and `1000000.000021`

# In[30]:


math.tan((math.pi)/2 + 10**-6)


# In[31]:


math.tan((math.pi)/2 - 10**-6)


# ### Printing
# Predict the output:
# ```
# a, b = 7, 11
# print(a, b)
# print('a', 'b')
# print('a', a, 'b', b)
# ```

# In[39]:


a, b = 7, 11
print(a, b)
print('a', 'b')
print('a', a, 'b', b)


# ### Geometric Mean
# 
# The *geometric mean* of two numbers $a$ and $b$ is defined to be $\sqrt{ab}$. Write a function **`geom_mean(a, b)`** that **returns** the geometric mean of two numbers. Assume that the arguments are positive numbers.
# 
# Examples:  
# `geom_mean(2, 1/8)` returns `0.5`.<br>
# `5 * geom_mean(2, 1/8)` returns `2.5`.  

# In[34]:


def geom_mean(a, b):
    return math.sqrt(a*b)


# In[35]:


geom_mean(2, 1/8)


# In[37]:


5 * geom_mean(2, 1/8)


# Write a function **`geom_mean_print(a, b)`** that **prints** the geometric mean of two numbers. 
# 
# Examples:  
# `geom_mean_print(2, 1/8)` displays `0.5`.<br>
# `5 * geom_mean_print(2, 1/8)` produces an error. Why?

# In[38]:


def geom_mean_print(a, b):
    print(geom_mean(a, b))


# In[40]:


geom_mean_print(2, 1/8)


# In[ ]:


# 5 * geom_mean_print(2, 1/8)
# there's an error bc we printed in the function; bc it was printed, the value wasn't stored so 
# so we can't perform mathematical operations on the output 


# *Important:* Before uploading to the autograder, you must **comment out** the previous line along with any other code that generates an error.

# ### Sum of Squares
# Write a function **`sum_squares(n)`** that returns the sum of the perfect squares $1^2 + 2^2 + \cdots + n^2$ using the formula
# 
# $$ S = \frac{n(n+1)(2n+1)}{6}. $$
# 
# Assume that `n` is a positive integer. The return value should be an `int`. (Note that the formula always evaluates to an integer. Why is that?)
# 
# Example:  
# `sum_squares(200)` returns `2686700` (not `2686700.0`).

# In[41]:


def sum_squares(n):
    return int((n*(n+1)*(2*n+1))/6)


# In[42]:


sum_squares(200)


# ### Fahrenheit and Celsius Conversion
# Write a function **`fah_to_cel(fah)`** that converts a Fahrenheit temperature to Celsius. Figure out the formula without looking it up. (*Hint:* Use the boiling point and freezing point temperatures.)
# 
# Example:  
# `fah_to_cel(-10)` returns `-23.333333`.

# In[44]:


def fah_to_cel(fah):
    return (fah-32) * (100/180)


# In[45]:


fah_to_cel(-10)


# In[ ]:





# In[ ]:





# Write a function **`cel_to_fah(cel)`** that converts a Celsius temperature to Fahrenheit.
#  
# Example:  
# `cel_to_fah(50)` returns `122.0`.

# In[48]:


def cel_to_fah(cel):
    return (cel * 180/100) + 32


# In[49]:


cel_to_fah(50)


# In[ ]:





# In[ ]:





# ### Upload Your Work to Gradescope for Autograding
# 
# **You must submit your Lab solutions to the autograder at the end of class even if your work is incomplete.**
# * In Jupyter, select `Run` $\to$ `Run All Cells` and make sure there are no errors in your code. Comment out any non-working code.
# * Select `File` $\to$ `Export Notebook as` $\to$ `Executable Script`. This will create an executable `.py` version of your notebook. (On some computers, you will `Download` to `.py` format.)
# * **The file must be named `Lab1.py`**.
# * In Gradescope, select the `Lab1.py` assignment and upload your `Lab1.py` file.
# * Your solutions will be autograded. Some test cases may be hidden. Correct answers will be shown in green. Incorrect answers will be shown in red.
# * If you do not complete the lab during class, you may resubmit your file up to a week late. Multiple submissions are allowed.
# * **Important: Your file must be in the correct format**. If the autograder is unable to grade your work, a score of zero will be given to the assignment.
# 

# ---
# 
# ## Extra Problems
# If you complete all of the assigned exercises before the end of class, please work on these extra problems.

# ### Jupyter Shortcuts
# 
# You can use the `_` symbol to reference a previous result. For example, if you execute `3 + 5` in a cell to produce 8, and then `_ * 10` in any other cell, the result will be 80. You also can reference a result by cell number. For example, `_12` gives the result from cell 12.
# 
# *Caution:* This feature should be used rarely and only for quick testing. This is a Jupyter feature, not a Python feature. **Do not use `_` in homework, quiz, or exam solutions.**
# 
# *Important:* **Comment out** any uses of `_` before uploading code to the autograder.

# In[ ]:





# ### Complex Numbers
# To represent complex numbers in Python use the letter `j` for $i = \sqrt{-1}$. For example the complex numbers $i$, $7i$ and $3+2i$ are represented as `1j`, `7j` and `3+2j`.
# 
# * Evaluate $i(3+2i)(4-5i)$.
#  
#  Answer: `(7+22j)` 

# In[ ]:





# ### Euler's Identity
# The equation $e^{i\pi} + 1 = 0$ is considered to be one of the most beautiful equations in mathematics. Verify the identity. 

# In[ ]:





# ### Lines and Angles
# Write a function **`x_axis_angle(x1, y1, x2, y2)`** that finds the angle $\theta$ (in radians) that a line makes with the $x$-axis given two points on the line $(x1, y1)$ and $(x2, y2)$. Assume $-\frac{\pi}{2}< \theta < \frac{\pi}{2}$ and that the line intersects the $x$-axis.
# 
# Example:  
# `x_axis_angle(0, 0, -5, 5)` returns `-0.785398`.<br />
# `x_axis_angle(2, 0, 7, 8.66)` returns `1.047185`.

# In[ ]:





# ### Hypotenuse
# Write a function **`hypot_trig(leg1, leg2)`** that finds the length of the hypotenuse of a right triangle given the lengths of the legs. The function should use **trigonometric functions** instead of the Pythagorean Theorem or the distance formula.
# 
# Example:  
# `hypot_trig(24, 7)` returns `25.0`.

# In[ ]:




