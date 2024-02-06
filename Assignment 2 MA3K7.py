#!/usr/bin/env python
# coding: utf-8

# In[99]:


import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.linalg as LA
import sympy as sp
import math
from time import perf_counter as timer
get_ipython().run_line_magic('matplotlib', '')


# In[88]:


#Game for 2x2 matrix
rng = np.random.default_rng()

samples = 100000
Det = []
dim = 2 # to give a 2 by 2 matrix

for i in np.arange(samples):
    A = rng.integers(0,2, (dim,dim))
    sumA = A.sum()
    #print(A) - Testing
    #print(sumA) - Testing
     #below am assumining "I" the player with 0's is going first
    if sumA == int((dim*dim)/2): # ensures out of the matrices generated only ones representative of the game are counted
        detA = LA.det(A)
        Det.append(detA)
#print(Det) - Testing

Max = max(Det)
Min = min(Det)
bins = np.arange(Min,Max+2)-0.5

fig, ax = plt.subplots(1,1)

ax.hist(Det, bins, density = True, rwidth = 1, color = 'teal')
ax.set_xlim(Min-1, Max+1)
ax.set_xticks(np.arange(Min, Max+1))
ax.set_xlabel('Det A')
ax.set_ylabel('probability mass function')
#plt.hist(Det)

plt.grid('on')
plt.show()


# In[90]:


#Game for 3x3 matrix
rng = np.random.default_rng()

samples = 100000
Det = []
dim = 3 # to give a 3 by 3 matrix

for i in np.arange(samples):
    A = rng.integers(0,2, (dim,dim))
    sumA = A.sum()
    #print(A) - Testing
    #print(sumA) - Testing
    if sumA == int((dim*dim)/2): # ensures out of the matrices generated only ones representative of the game are counted
        detA = LA.det(A)
        Det.append(detA)
#print(Det) - Testing

Max = max(Det)
Min = min(Det)
bins = np.arange(Min,Max+2)-0.5

fig, ax = plt.subplots(1,1)

ax.hist(Det, bins, density = True, rwidth = 1, color = 'green')
ax.set_xlim(Min-1, Max+1)
ax.set_xticks(np.arange(Min, Max+1))
ax.set_xlabel('Det A')
ax.set_ylabel('probability mass function')
#plt.hist(Det)

plt.grid('on')
plt.show()


# In[91]:


#Game for 8x8 matrix
rng = np.random.default_rng()

samples = 100000
Det = []
dim = 8 # to give a 8 by 8 matrix

for i in np.arange(samples):
    A = rng.integers(0,2, (dim,dim))
    sumA = A.sum()
    #print(A) - Testing
    #print(sumA) - Testing
    if sumA == int((dim*dim)/2): # ensures out of the matrices generated only ones representative of the game are counted
        detA = LA.det(A)
        Det.append(detA)
#print(Det) - Testing

Max = max(Det)
Min = min(Det)
bins = np.arange(Min,Max+2)-0.5

fig, ax = plt.subplots(1,1)

ax.hist(Det, bins, density = True, rwidth = 1, color = 'purple')
ax.set_xlim(Min-1, Max+1)
ax.set_xticks(np.arange(Min, Max+1))
ax.set_xlabel('Det A')
ax.set_ylabel('probability mass function')
#plt.hist(Det)

plt.grid('on')
plt.show()


# In[93]:


#Game for 7x7 matrix
rng = np.random.default_rng()

samples = 1000000
Det = []
dim = 7 # to give a 7 by 7 matrix

for i in np.arange(samples):
    A = rng.integers(0,2, (dim,dim))
    sumA = A.sum()
    #print(A) - Testing
    #print(sumA) - Testing
    if sumA == int((dim*dim)/2): # ensures out of the matrices generated only ones representative of the game are counted
        detA = LA.det(A)
        Det.append(detA)
#print(Det) - Testing

Max = max(Det)
Min = min(Det)
bins = np.arange(Min,Max+2)-0.5

fig, ax = plt.subplots(1,1)

ax.hist(Det, bins, density = True, rwidth = 1, color = 'blue')
ax.set_xlim(Min-1, Max+1)
ax.set_xticks(np.arange(Min, Max+1))
ax.set_xlabel('Det A')
ax.set_ylabel('probability mass function')
#plt.hist(Det)

plt.grid('on')
plt.show()


# In[94]:


#Game for 6x6 matrix
rng = np.random.default_rng()

samples = 1000000
Det = []
dim = 6 # to give a 6 by 6 matrix

for i in np.arange(samples):
    A = rng.integers(0,2, (dim,dim))
    sumA = A.sum()
    #print(A) - Testing
    #print(sumA) - Testing
    if sumA == int((dim*dim)/2): # ensures out of the matrices generated only ones representative of the game are counted
        detA = LA.det(A)
        Det.append(detA)
#print(Det) - Testing

Max = max(Det)
Min = min(Det)
bins = np.arange(Min,Max+2)-0.5

fig, ax = plt.subplots(1,1)

ax.hist(Det, bins, density = True, rwidth = 1, color = 'red')
ax.set_xlim(Min-1, Max+1)
ax.set_xticks(np.arange(Min, Max+1))
ax.set_xlabel('Det A')
ax.set_ylabel('probability mass function')
#plt.hist(Det)

plt.grid('on')
plt.show()


# In[95]:


# new code so can generate a more representative sample size without needing extremely long running times
dim = 2 # dimension of our matrix starting low to test code is accurate
Det = [] # array/list of our determinants to determine who wins the game, wanted to keep variable name same so could copy and paste histogram code over
samples = 1000 # the size of our sample set

for i in range(samples):  #I am construction code that generates appropriate matrices for the game
    
    matrix = np.empty(shape=(dim,dim)) # starting with an empty matrix, here have called matrix as we are making them 
    matrix.fill(2) # we fill it with a value that is not 1 or 0 here I have decided to use 2

    iteration = 0 # a way to count how many values have been filled in
    zeroTurn = True # this ensures I (player 1) goes first

    for i in range(dim**2): # this function goes through our matrix of 2's ramdomly finds an elements that has not been defined and sets it to 0 or 1 depending on who's turn it is
        randPosition = random.randrange(1, (dim**2) - iteration + 1)
        iteration = iteration + 1 # there is now one less space that needs to be filled 
        numberUnfilled = 0
        for i in range(dim**2):
            if (matrix[int(i/dim),i % dim] == 2): # goes to the generated position and if unclassed sets it to the appropriate value
                numberUnfilled = numberUnfilled + 1
                if (numberUnfilled == randPosition):
                    if zeroTurn:
                        matrix[int(i/dim),i % dim]= 0
                        zeroTurn = False # ends my turn so that the next random space to be filled is defined to be a 1
                    else:
                        matrix[int(i/dim),i % dim]= 1
                        zeroTurn = True
    Det.append(LA.det(matrix)) # then adds the determinant of our random matrix to the list 
                    
#print(Det) - For testing
#time to plot a hisogram with our results and hopefully will still confirm my previous results:
Max = max(Det)
Min = min(Det)
bins = np.arange(Min,Max+2)-0.5

fig, ax = plt.subplots(1,1)

ax.hist(Det, bins, density = True, rwidth = 1, color = 'teal')
ax.set_xlim(Min-1, Max+1)
ax.set_xticks(np.arange(Min, Max+1))
ax.set_xlabel('Det A')
ax.set_ylabel('probability mass function')
#plt.hist(Det)

plt.grid('on')
plt.show()


# In[96]:


# new code so can generate a more representative sample size without needing extremely long running times
dim = 7 # dimension of our matrix re-testing my previous boundary dimension
Det = [] # array/list of our determinants to determine who wins the game, wanted to keep variable name same so could copy and paste histogram code over
samples = 1000 # the size of our sample set

for i in range(samples):  #I am construction code that generates appropriate matrices for the game
    
    matrix = np.empty(shape=(dim,dim)) # starting with an empty matrix, here have called matrix as we are making them 
    matrix.fill(2) # we fill it with a value that is not 1 or 0 here I have decided to use 2

    iteration = 0 # a way to count how many values have been filled in
    zeroTurn = True # this ensures I (player 1) goes first

    for i in range(dim**2): # this function goes through our matrix of 2's ramdomly finds an elements that has not been defined and sets it to 0 or 1 depending on who's turn it is
        randPosition = random.randrange(1, (dim**2) - iteration + 1)
        iteration = iteration + 1 # there is now one less space that needs to be filled 
        numberUnfilled = 0
        for i in range(dim**2):
            if (matrix[int(i/dim),i % dim] == 2): # goes to the generated position and if unclassed sets it to the appropriate value
                numberUnfilled = numberUnfilled + 1
                if (numberUnfilled == randPosition):
                    if zeroTurn:
                        matrix[int(i/dim),i % dim]= 0
                        zeroTurn = False # ends my turn so that the next random space to be filled is defined to be a 1
                    else:
                        matrix[int(i/dim),i % dim]= 1
                        zeroTurn = True
    Det.append(LA.det(matrix)) # then adds the determinant of our random matrix to the list 
                    
#print(Det) - For testing
#time to plot a hisogram with our results and hopefully will still confirm my previous results:
Max = max(Det)
Min = min(Det)
bins = np.arange(Min,Max+2)-0.5

fig, ax = plt.subplots(1,1)

ax.hist(Det, bins, density = True, rwidth = 1, color = 'blue')
ax.set_xlim(Min-1, Max+1)
ax.set_xticks(np.arange(Min, Max+1))
ax.set_xlabel('Det A')
ax.set_ylabel('probability mass function')
#plt.hist(Det)

plt.grid('on')
plt.show()


# In[ ]:


# new code so can generate a more representative sample size without needing extremely long running times
dim = 7 # dimension of our matrix re-testing my previous boundary dimension
Det = [] # array/list of our determinants to determine who wins the game, wanted to keep variable name same so could copy and paste histogram code over
samples = 100000 # the size of our sample set

for i in range(samples):  #I am construction code that generates appropriate matrices for the game
    
    matrix = np.empty(shape=(dim,dim)) # starting with an empty matrix, here have called matrix as we are making them 
    matrix.fill(2) # we fill it with a value that is not 1 or 0 here I have decided to use 2

    iteration = 0 # a way to count how many values have been filled in
    zeroTurn = True # this ensures I (player 1) goes first

    for i in range(dim**2): # this function goes through our matrix of 2's ramdomly finds an elements that has not been defined and sets it to 0 or 1 depending on who's turn it is
        randPosition = random.randrange(1, (dim**2) - iteration + 1)
        iteration = iteration + 1 # there is now one less space that needs to be filled 
        numberUnfilled = 0
        for i in range(dim**2):
            if (matrix[int(i/dim),i % dim] == 2): # goes to the generated position and if unclassed sets it to the appropriate value
                numberUnfilled = numberUnfilled + 1
                if (numberUnfilled == randPosition):
                    if zeroTurn:
                        matrix[int(i/dim),i % dim]= 0
                        zeroTurn = False # ends my turn so that the next random space to be filled is defined to be a 1
                    else:
                        matrix[int(i/dim),i % dim]= 1
                        zeroTurn = True
    Det.append(LA.det(matrix)) # then adds the determinant of our random matrix to the list 
                    
#print(Det) - For testing
#time to plot a hisogram with our results and hopefully will still confirm my previous results:
Max = max(Det)
Min = min(Det)
bins = np.arange(Min,Max+2)-0.5

fig, ax = plt.subplots(1,1)

ax.hist(Det, bins, density = True, rwidth = 1, color = 'blue')
ax.set_xlim(Min-1, Max+1)
ax.set_xticks(np.arange(Min, Max+1))
ax.set_xlabel('Det A')
ax.set_ylabel('probability mass function')
#plt.hist(Det)

plt.grid('on')
plt.show()


# In[100]:


# new code so can generate a more representative sample size without needing extremely long running times
dim = 6 # dimension of our matrix re-testing my previous boundary dimension
Det = [] # array/list of our determinants to determine who wins the game, wanted to keep variable name same so could copy and paste histogram code over
samples = 10000 # the size of our sample set

for i in range(samples):  #I am construction code that generates appropriate matrices for the game
    
    matrix = np.empty(shape=(dim,dim)) # starting with an empty matrix, here have called matrix as we are making them 
    matrix.fill(2) # we fill it with a value that is not 1 or 0 here I have decided to use 2

    iteration = 0 # a way to count how many values have been filled in
    zeroTurn = True # this ensures I (player 1) goes first

    for i in range(dim**2): # this function goes through our matrix of 2's ramdomly finds an elements that has not been defined and sets it to 0 or 1 depending on who's turn it is
        randPosition = random.randrange(1, (dim**2) - iteration + 1)
        iteration = iteration + 1 # there is now one less space that needs to be filled 
        numberUnfilled = 0
        for i in range(dim**2):
            if (matrix[int(i/dim),i % dim] == 2): # goes to the generated position and if unclassed sets it to the appropriate value
                numberUnfilled = numberUnfilled + 1
                if (numberUnfilled == randPosition):
                    if zeroTurn:
                        matrix[int(i/dim),i % dim]= 0
                        zeroTurn = False # ends my turn so that the next random space to be filled is defined to be a 1
                    else:
                        matrix[int(i/dim),i % dim]= 1
                        zeroTurn = True
    Det.append(LA.det(matrix)) # then adds the determinant of our random matrix to the list 
                    
#print(Det) - For testing
#time to plot a hisogram with our results and hopefully will still confirm my previous results:
Max = max(Det)
Min = min(Det)
bins = np.arange(Min,Max+2)-0.5

fig, ax = plt.subplots(1,1)

ax.hist(Det, bins, density = True, rwidth = 1, color = 'blue')
ax.set_xlim(Min-1, Max+1)
ax.set_xticks(np.arange(Min, Max+1))
ax.set_xlabel('Det A')
ax.set_ylabel('probability mass function')
#plt.hist(Det)

plt.grid('on')
plt.show()


# In[107]:


#Testing with the assumption that the friend is starting:


# new code so can generate a more representative sample size without needing extremely long running times
dim = 3 # dimension of our matrix
Det = [] # array/list of our determinants to determine who wins the game, wanted to keep variable name same so could copy and paste histogram code over
samples = 10000 # the size of our sample set

for i in range(samples):  #I am construction code that generates appropriate matrices for the game
    
    matrix = np.empty(shape=(dim,dim)) # starting with an empty matrix, here have called matrix as we are making them 
    matrix.fill(2) # we fill it with a value that is not 1 or 0 here I have decided to use 2

    iteration = 0 # a way to count how many values have been filled in
    oneTurn = True # this ensures I (player 1) goes first

    for i in range(dim**2): # this function goes through our matrix of 2's ramdomly finds an elements that has not been defined and sets it to 0 or 1 depending on who's turn it is
        randPosition = random.randrange(1, (dim**2) - iteration + 1)
        iteration = iteration + 1 # there is now one less space that needs to be filled 
        numberUnfilled = 0
        for i in range(dim**2):
            if (matrix[int(i/dim),i % dim] == 2): # goes to the generated position and if unclassed sets it to the appropriate value
                numberUnfilled = numberUnfilled + 1
                if (numberUnfilled == randPosition):
                    if oneTurn:
                        matrix[int(i/dim),i % dim]= 1
                        oneTurn = False # ends my turn so that the next random space to be filled is defined to be a 1
                    else:
                        matrix[int(i/dim),i % dim]= 0
                        oneTurn = True
    Det.append(LA.det(matrix)) # then adds the determinant of our random matrix to the list 
                    
#print(Det) - For testing
#time to plot a hisogram with our results and hopefully will still confirm my previous results:
Max = max(Det)
Min = min(Det)
bins = np.arange(Min,Max+2)-0.5

fig, ax = plt.subplots(1,1)

ax.hist(Det, bins, density = True, rwidth = 1, color = 'green')
ax.set_xlim(Min-1, Max+1)
ax.set_xticks(np.arange(Min, Max+1))
ax.set_xlabel('Det A')
ax.set_ylabel('probability mass function')
#plt.hist(Det)

plt.grid('on')
plt.show()


# In[108]:


#Testing with the assumption that the friend is starting:
# new code so can generate a more representative sample size without needing extremely long running times
dim = 7 # dimension of our matrix
Det = [] # array/list of our determinants to determine who wins the game, wanted to keep variable name same so could copy and paste histogram code over
samples = 10000 # the size of our sample set

for i in range(samples):  #I am construction code that generates appropriate matrices for the game
    
    matrix = np.empty(shape=(dim,dim)) # starting with an empty matrix, here have called matrix as we are making them 
    matrix.fill(2) # we fill it with a value that is not 1 or 0 here I have decided to use 2

    iteration = 0 # a way to count how many values have been filled in
    oneTurn = True # this ensures I (player 1) goes first

    for i in range(dim**2): # this function goes through our matrix of 2's ramdomly finds an elements that has not been defined and sets it to 0 or 1 depending on who's turn it is
        randPosition = random.randrange(1, (dim**2) - iteration + 1)
        iteration = iteration + 1 # there is now one less space that needs to be filled 
        numberUnfilled = 0
        for i in range(dim**2):
            if (matrix[int(i/dim),i % dim] == 2): # goes to the generated position and if unclassed sets it to the appropriate value
                numberUnfilled = numberUnfilled + 1
                if (numberUnfilled == randPosition):
                    if oneTurn:
                        matrix[int(i/dim),i % dim]= 1
                        oneTurn = False # ends my turn so that the next random space to be filled is defined to be a 1
                    else:
                        matrix[int(i/dim),i % dim]= 0
                        oneTurn = True
    Det.append(LA.det(matrix)) # then adds the determinant of our random matrix to the list 
                    
#print(Det) - For testing
#time to plot a hisogram with our results and hopefully will still confirm my previous results:
Max = max(Det)
Min = min(Det)
bins = np.arange(Min,Max+2)-0.5

fig, ax = plt.subplots(1,1)

ax.hist(Det, bins, density = True, rwidth = 1, color = 'teal')
ax.set_xlim(Min-1, Max+1)
ax.set_xticks(np.arange(Min, Max+1))
ax.set_xlabel('Det A')
ax.set_ylabel('probability mass function')
#plt.hist(Det)

plt.grid('on')
plt.show()


# In[109]:


#Testing with the assumption that the friend is starting:
# new code so can generate a more representative sample size without needing extremely long running times
dim = 6 # dimension of our matrix
Det = [] # array/list of our determinants to determine who wins the game, wanted to keep variable name same so could copy and paste histogram code over
samples = 10000 # the size of our sample set

for i in range(samples):  #I am construction code that generates appropriate matrices for the game
    
    matrix = np.empty(shape=(dim,dim)) # starting with an empty matrix, here have called matrix as we are making them 
    matrix.fill(2) # we fill it with a value that is not 1 or 0 here I have decided to use 2

    iteration = 0 # a way to count how many values have been filled in
    oneTurn = True # this ensures I (player 1) goes first

    for i in range(dim**2): # this function goes through our matrix of 2's ramdomly finds an elements that has not been defined and sets it to 0 or 1 depending on who's turn it is
        randPosition = random.randrange(1, (dim**2) - iteration + 1)
        iteration = iteration + 1 # there is now one less space that needs to be filled 
        numberUnfilled = 0
        for i in range(dim**2):
            if (matrix[int(i/dim),i % dim] == 2): # goes to the generated position and if unclassed sets it to the appropriate value
                numberUnfilled = numberUnfilled + 1
                if (numberUnfilled == randPosition):
                    if oneTurn:
                        matrix[int(i/dim),i % dim]= 1
                        oneTurn = False # ends my turn so that the next random space to be filled is defined to be a 1
                    else:
                        matrix[int(i/dim),i % dim]= 0
                        oneTurn = True
    Det.append(LA.det(matrix)) # then adds the determinant of our random matrix to the list 
                    
#print(Det) - For testing
#time to plot a hisogram with our results and hopefully will still confirm my previous results:
Max = max(Det)
Min = min(Det)
bins = np.arange(Min,Max+2)-0.5

fig, ax = plt.subplots(1,1)

ax.hist(Det, bins, density = True, rwidth = 1, color = 'red')
ax.set_xlim(Min-1, Max+1)
ax.set_xticks(np.arange(Min, Max+1))
ax.set_xlabel('Det A')
ax.set_ylabel('probability mass function')
#plt.hist(Det)

plt.grid('on')
plt.show()


# In[114]:


# seeing if I can find dimension that gives maximal probability that I win 

dim = 4 # dimension of our matrix starting low to test code is accurate
Det = [] # array/list of our determinants to determine who wins the game, wanted to keep variable name same so could copy and paste histogram code over
samples = 10000 # the size of our sample set

for i in range(samples):  #I am construction code that generates appropriate matrices for the game
    
    matrix = np.empty(shape=(dim,dim)) # starting with an empty matrix, here have called matrix as we are making them 
    matrix.fill(2) # we fill it with a value that is not 1 or 0 here I have decided to use 2

    iteration = 0 # a way to count how many values have been filled in
    zeroTurn = True # this ensures I (player 1) goes first

    for i in range(dim**2): # this function goes through our matrix of 2's ramdomly finds an elements that has not been defined and sets it to 0 or 1 depending on who's turn it is
        randPosition = random.randrange(1, (dim**2) - iteration + 1)
        iteration = iteration + 1 # there is now one less space that needs to be filled 
        numberUnfilled = 0
        for i in range(dim**2):
            if (matrix[int(i/dim),i % dim] == 2): # goes to the generated position and if unclassed sets it to the appropriate value
                numberUnfilled = numberUnfilled + 1
                if (numberUnfilled == randPosition):
                    if zeroTurn:
                        matrix[int(i/dim),i % dim]= 0
                        zeroTurn = False # ends my turn so that the next random space to be filled is defined to be a 1
                    else:
                        matrix[int(i/dim),i % dim]= 1
                        zeroTurn = True
    Det.append(LA.det(matrix)) # then adds the determinant of our random matrix to the list 
                    
#print(Det) - For testing
#time to plot a hisogram with our results and hopefully will still confirm my previous results:
Max = max(Det)
Min = min(Det)
bins = np.arange(Min,Max+2)-0.5

fig, ax = plt.subplots(1,1)

ax.hist(Det, bins, density = True, rwidth = 1, color = 'teal')
ax.set_xlim(Min-1, Max+1)
ax.set_xticks(np.arange(Min, Max+1))
ax.set_xlabel('Det A')
ax.set_ylabel('probability mass function')
#plt.hist(Det)

plt.grid('on')
plt.show()


# In[115]:


# seeing if I can find dimension that gives maximal probability that I win 

dim = 5 # dimension of our matrix starting low to test code is accurate
Det = [] # array/list of our determinants to determine who wins the game, wanted to keep variable name same so could copy and paste histogram code over
samples = 10000 # the size of our sample set

for i in range(samples):  #I am construction code that generates appropriate matrices for the game
    
    matrix = np.empty(shape=(dim,dim)) # starting with an empty matrix, here have called matrix as we are making them 
    matrix.fill(2) # we fill it with a value that is not 1 or 0 here I have decided to use 2

    iteration = 0 # a way to count how many values have been filled in
    zeroTurn = True # this ensures I (player 1) goes first

    for i in range(dim**2): # this function goes through our matrix of 2's ramdomly finds an elements that has not been defined and sets it to 0 or 1 depending on who's turn it is
        randPosition = random.randrange(1, (dim**2) - iteration + 1)
        iteration = iteration + 1 # there is now one less space that needs to be filled 
        numberUnfilled = 0
        for i in range(dim**2):
            if (matrix[int(i/dim),i % dim] == 2): # goes to the generated position and if unclassed sets it to the appropriate value
                numberUnfilled = numberUnfilled + 1
                if (numberUnfilled == randPosition):
                    if zeroTurn:
                        matrix[int(i/dim),i % dim]= 0
                        zeroTurn = False # ends my turn so that the next random space to be filled is defined to be a 1
                    else:
                        matrix[int(i/dim),i % dim]= 1
                        zeroTurn = True
    Det.append(LA.det(matrix)) # then adds the determinant of our random matrix to the list 
                    
#print(Det) - For testing
#time to plot a hisogram with our results and hopefully will still confirm my previous results:
Max = max(Det)
Min = min(Det)
bins = np.arange(Min,Max+2)-0.5

fig, ax = plt.subplots(1,1)

ax.hist(Det, bins, density = True, rwidth = 1, color = 'teal')
ax.set_xlim(Min-1, Max+1)
ax.set_xticks(np.arange(Min, Max+1))
ax.set_xlabel('Det A')
ax.set_ylabel('probability mass function')
#plt.hist(Det)

plt.grid('on')
plt.show()


# In[112]:


# seeing if I can find dimension that gives maximal probability that I win 

dim = 3 # dimension of our matrix starting low to test code is accurate
Det = [] # array/list of our determinants to determine who wins the game, wanted to keep variable name same so could copy and paste histogram code over
samples = 10000 # the size of our sample set

for i in range(samples):  #I am construction code that generates appropriate matrices for the game
    
    matrix = np.empty(shape=(dim,dim)) # starting with an empty matrix, here have called matrix as we are making them 
    matrix.fill(2) # we fill it with a value that is not 1 or 0 here I have decided to use 2

    iteration = 0 # a way to count how many values have been filled in
    zeroTurn = True # this ensures I (player 1) goes first

    for i in range(dim**2): # this function goes through our matrix of 2's ramdomly finds an elements that has not been defined and sets it to 0 or 1 depending on who's turn it is
        randPosition = random.randrange(1, (dim**2) - iteration + 1)
        iteration = iteration + 1 # there is now one less space that needs to be filled 
        numberUnfilled = 0
        for i in range(dim**2):
            if (matrix[int(i/dim),i % dim] == 2): # goes to the generated position and if unclassed sets it to the appropriate value
                numberUnfilled = numberUnfilled + 1
                if (numberUnfilled == randPosition):
                    if zeroTurn:
                        matrix[int(i/dim),i % dim]= 0
                        zeroTurn = False # ends my turn so that the next random space to be filled is defined to be a 1
                    else:
                        matrix[int(i/dim),i % dim]= 1
                        zeroTurn = True
    Det.append(LA.det(matrix)) # then adds the determinant of our random matrix to the list 
                    
#print(Det) - For testing
#time to plot a hisogram with our results and hopefully will still confirm my previous results:
Max = max(Det)
Min = min(Det)
bins = np.arange(Min,Max+2)-0.5

fig, ax = plt.subplots(1,1)

ax.hist(Det, bins, density = True, rwidth = 1, color = 'teal')
ax.set_xlim(Min-1, Max+1)
ax.set_xticks(np.arange(Min, Max+1))
ax.set_xlabel('Det A')
ax.set_ylabel('probability mass function')
#plt.hist(Det)

plt.grid('on')
plt.show()


# In[116]:


# seeing if my friend using 2 affects the results

dim = 3 # dimension of our matrix starting low to test code is accurate
Det = [] # array/list of our determinants to determine who wins the game, wanted to keep variable name same so could copy and paste histogram code over
samples = 10000 # the size of our sample set

for i in range(samples):  #I am construction code that generates appropriate matrices for the game
    
    matrix = np.empty(shape=(dim,dim)) # starting with an empty matrix, here have called matrix as we are making them 
    matrix.fill(2) # we fill it with a value that is not 1 or 0 here I have decided to use 2

    iteration = 0 # a way to count how many values have been filled in
    zeroTurn = True # this ensures I (player 1) goes first

    for i in range(dim**2): # this function goes through our matrix of 2's ramdomly finds an elements that has not been defined and sets it to 0 or 1 depending on who's turn it is
        randPosition = random.randrange(1, (dim**2) - iteration + 1)
        iteration = iteration + 1 # there is now one less space that needs to be filled 
        numberUnfilled = 0
        for i in range(dim**2):
            if (matrix[int(i/dim),i % dim] == 2): # goes to the generated position and if unclassed sets it to the appropriate value
                numberUnfilled = numberUnfilled + 1
                if (numberUnfilled == randPosition):
                    if zeroTurn:
                        matrix[int(i/dim),i % dim]= 0
                        zeroTurn = False # ends my turn so that the next random space to be filled is defined to be a 1
                    else:
                        matrix[int(i/dim),i % dim]= 2
                        zeroTurn = True
    Det.append(LA.det(matrix)) # then adds the determinant of our random matrix to the list 
                    
#print(Det) - For testing
#time to plot a hisogram with our results and hopefully will still confirm my previous results:
Max = max(Det)
Min = min(Det)
bins = np.arange(Min,Max+2)-0.5

fig, ax = plt.subplots(1,1)

ax.hist(Det, bins, density = True, rwidth = 1, color = 'teal')
ax.set_xlim(Min-1, Max+1)
ax.set_xticks(np.arange(Min, Max+1))
ax.set_xlabel('Det A')
ax.set_ylabel('probability mass function')
#plt.hist(Det)

plt.grid('on')
plt.show()


# In[117]:


# seeing if my friend using 2 affects the results

dim = 7 # dimension of our matrix starting low to test code is accurate
Det = [] # array/list of our determinants to determine who wins the game, wanted to keep variable name same so could copy and paste histogram code over
samples = 10000 # the size of our sample set

for i in range(samples):  #I am construction code that generates appropriate matrices for the game
    
    matrix = np.empty(shape=(dim,dim)) # starting with an empty matrix, here have called matrix as we are making them 
    matrix.fill(2) # we fill it with a value that is not 1 or 0 here I have decided to use 2

    iteration = 0 # a way to count how many values have been filled in
    zeroTurn = True # this ensures I (player 1) goes first

    for i in range(dim**2): # this function goes through our matrix of 2's ramdomly finds an elements that has not been defined and sets it to 0 or 1 depending on who's turn it is
        randPosition = random.randrange(1, (dim**2) - iteration + 1)
        iteration = iteration + 1 # there is now one less space that needs to be filled 
        numberUnfilled = 0
        for i in range(dim**2):
            if (matrix[int(i/dim),i % dim] == 2): # goes to the generated position and if unclassed sets it to the appropriate value
                numberUnfilled = numberUnfilled + 1
                if (numberUnfilled == randPosition):
                    if zeroTurn:
                        matrix[int(i/dim),i % dim]= 0
                        zeroTurn = False # ends my turn so that the next random space to be filled is defined to be a 1
                    else:
                        matrix[int(i/dim),i % dim]= 2
                        zeroTurn = True
    Det.append(LA.det(matrix)) # then adds the determinant of our random matrix to the list 
                    
#print(Det) - For testing
#time to plot a hisogram with our results and hopefully will still confirm my previous results:
Max = max(Det)
Min = min(Det)
bins = np.arange(Min,Max+2)-0.5

fig, ax = plt.subplots(1,1)

ax.hist(Det, bins, density = True, rwidth = 1, color = 'teal')
ax.set_xlim(Min-1, Max+1)
ax.set_xticks(np.arange(Min, Max+1))
ax.set_xlabel('Det A')
ax.set_ylabel('probability mass function')
#plt.hist(Det)

plt.grid('on')
plt.show()

