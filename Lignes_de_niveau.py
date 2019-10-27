import autograd 
from autograd import numpy as np
import matplotlib.pyplot as plt

## Amorce :

def find_seed(g, c=0, eps= 2**(-26)):
    def h(t):
        return (g(0,t) -c)
    if h(0) * h(1) > 0:
        return None
    i,j = 0,1
    while (j-i)>eps:
        t = (i+j)/2
        if h(i)*h(t)<=0:
            j = t
        else:
            i = t
    return i

##Fonctions de test

def oui(x,y):
    return x**2+y**2
def f(x,y):
    return np.exp(-x**2 -y**2)
    
def g(x,y):
    return np.exp(-(x-1)**2-(y-1)**2)

def l(x,y):
    return 2*(f(x,y) - g(x,y))

def i(x,y):
    return ((x+1)**2 - y**2)
    
    
## Propagation :

def grad(f,x,y):
    g = autograd.grad
    return np.r_[g(f,0)(x,y),g(f,1)(x,y)]

def simple_contour(f, c=0.0, delta = 0.01):
    x,y = [],[]
    if find_seed(f,c) != None:
        t = find_seed(f,c)
        x.append(0.0)
        y.append(t)
        [a,b] = grad(f,0.0,t)
        while abs(y[-1]) > 2*delta:
            x_0,y_0 = x[-1],y[-1]
            if b !=0:
                x.append(x_0 + delta/np.sqrt((1 + (a/b)**2)))
                y.append(-a/b*(x[-1] - x_0) + y_0)
            else:
                x.append(x_0)
                y.append(y_0 + delta)
            [a,b] = grad(f,x[-1],y[-1])
        return x,y
    return [],[]
    
x1,y1 = simple_contour(l,0.5)
x2,y2 = simple_contour(l,1.0)
x3,y3 = simple_contour(l,1.5)
x4,y4 = simple_contour(l,0.0)

plt.plot(x1,y1)
plt.plot(x2,y2)
plt.plot(x3,y3)
plt.plot(x4,y4)
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#La méthode de Newton est équivalente et donne la même formule

## Contour complexe


def dichotomie(h,a,b,l):
    if h_l(a) * h_l(b) <= 0:
        i,j = a,b
        while (j-i)>eps:
            t = (i+j)/2
            if h_l(i)*h_l(t)<=0:
                j = t
            else:
                i = t
        return [i,l]
    
def find_seed_2(g,a,b,c=0,eps= 2**(-26)):
    def h_0(t):
        return (g(0,t) -c)
    def h_1(t):
        return (g(1,t) -c)
    def h_2(t):
        return (g(t,0) -c)
    def h_3(t):
        return (g(t,1) -c)
    for k in range(4):
        dichotomie(h,a,b,k)
    return None
    
def simple_contour_2(f,a,b, c=0.0, delta = 0.01):
    x,y = [],[]
    if find_seed_2(f,c) != None:
        t = find_seed_2(f,c)[0]
        x.append(0.0)
        y.append(t)
        if i ==0:
            [a,b] = grad(f,0.0,t)
        if i ==1:
            [a,b] = grad(f,1.0,t)
        if i ==2:
            [a,b] = grad(f,t,0.0)
        if i ==2:
            [a,b] = grad(f,t,1.0)
        while abs(y[-1]) > 2*delta:
            x_0,y_0 = x[-1],y[-1]
            if b !=0:
                x.append(x_0 + delta**2/(1 + (a/b)**2))
                y.append(-a/b*(x[-1] - x_0) + y_0)
            else:
                x.append(x_0)
                y.append(y_0 + delta)
            [a,b] = grad(f,x[-1],y[-1])
        return x,y
    return [],[]

def contour(f, c=0.0, xc = [0.0,1.0], yc = [0.0,1.0], delta = 0.01):
    xs,ys = [],[]
    for i in range(len(xc)-1): 
        for j in range(len(yc)-1):
            x,y=simple_contour_2(f,xc[i],xc[i+1],c,delta)
            xs.append(x)
            ys.append(y)
    return xs,ys
        
    
    
#valeurs de test :
xc = [0.0,0.5,1]
yc = np.linspace(0,1,100)























