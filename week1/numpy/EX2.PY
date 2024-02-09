import numpy as np
# Create a 3x3 array of all zeros
a = np.zeros((3,3))
print(a)
# Create a 2x2 array of all ones
b = np.ones((2,2))
print(b)
# Create a 3x3 constant array

c = np.full((3,3), 7)
print(c)
# Create a 3x3 array filled with random values
d = np.random.random((3,3))
print(d)
# Create a 3x3 identity matrix
e = np.eye(3)
print(e)
# convert list to array
f = np.array([2, 3, 1, 0])
print(f)
# arange() will create arrays with regularly incrementing values
g = np.arange(20)
print(g)
# note mix of tuple and lists
h = np.array([[0, 1,2.0],[0,0,0],(1+1j,3.,2.)])
print(h)
# create an array of range with float data type
i = np.arange(1, 8, dtype=np.float)
print(i)
# linspace() will create arrays with a specified number of items which are
# spaced equally between the specified beginning and end values j= np.linspace(start, stop, num)
j = np.linspace(2, 4, 5)
print(j)