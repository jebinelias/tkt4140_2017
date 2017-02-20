# Solution to Python exercise 1 in TKT4140 Numerical Methods.

# 1a
print 'Hello, World!'

# 1b
import numpy as np
a = np.array([1,2,3])
b = np.array([4,5,6])
A = np.array([[1,1,2],[2,3,3],[4,4,5]])
B = np.array([[2,4,6],[8,10,12],[14,16,18]])
print a+b
print np.dot(a,b)
print np.dot(A,B)
print np.transpose(A)
print np.linalg.inv(A)
print np.linalg.solve(A,b)

# 1c
def fib(n):
    """Write the Fibonacci series up to and including element n."""
    a, b, i = 0, 1, 1
    while i <= n:
        print a,
        a, b, i = b, a+b, i+1
        
fib(20)

# 1d
import matplotlib.pylab as plt
import numpy as np

x1 = np.linspace(0,2*np.pi,10)
x2 = np.linspace(0,2*np.pi,100)
u1 = np.sin(x1)
u2 = np.sin(x2)

plt.figure()
plt.rcParams['font.size'] = 16
plt.plot(x2,u2,'g.-',x1,u1,'r-v',markersize=10,linewidth=2)
plt.legend(['100 points','10 points'], loc='best', frameon=False)
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Plot of sin(x)')
plt.grid()
plt.savefig("sinus_plot.png", transparent=True)
plt.show()

# 1e
import matplotlib.pylab as plt
def fib_sum(n):
    """Plots the sum of the elements in the Fibonacci series up to and including element n."""
    a, b, i = 0, 1, 0
    fibsum = []
    while i < n:
        fibsum.append(a)
        a, b, i = b, a+b, i+1
    return fibsum
    
fibsum = fib_sum(30)

plt.plot(fibsum, color='r', marker='o', linewidth=2, markersize=8, markerfacecolor='g', markeredgecolor='b')
plt.rcParams['font.size'] = 16
plt.ylabel('Sum of the Fibonacci series')
plt.title('Fibonacci')
plt.yscale('log')
plt.grid()
plt.savefig('fibonacci_plot.png', transparent=True)
plt.show()