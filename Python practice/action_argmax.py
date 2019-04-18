import numpy as np
import pylab as plt
import math
import random
from random import randint


Q = [[7, 5],
                       [4, 2],
                       [5, 3],
                       [4, 6],
                       [2, 4],
                       [5, 2]]


##state =3
###print(np.argmax(Q[3]))
###print(max(Q[3]))
##
###t = 1#float(list(range(1, 21)))
###print(t)
##hh=[]
##for pp in range(1000):
##    #epsilon = float(np.exp(-0.008*pp))
##    epsilon = float(np.exp(-0.009*pp))
##    
##    hh.append(epsilon)
##

#print(hh)

##plt.plot(hh)
##plt.show()
##
##
##low=-0.6
##high=-0.4
##for ww in range(3):
##    bb=random.uniform(low, high)
##    print(bb)
##
##cc=np.array([random.uniform(-0.6, -0.4), 0])
###print(cc)
##
##kk = np.random.randint(0, 99)
##print(kk)

##x =np.array([2,3,8])
##y=np.array([-2,0,6])
x =[2,3,8]
y=[-2,0,6]
sumx = np.add(x,y)

print(sumx)
##
##store =[]
##for inde in range(3):
##    
##    if np.random.randint(0,10)>5:
##        ff =x[inde]
##    else:
##        ff=y[inde]
##    store.append(ff)
##print(store)
##        

##
##roundy =[]
##for indr in range(3):
##    
##    if indr%2==1:
##        ffc =x[indr]
##    else:
##        ffc=y[indr]
##    roundy.append(ffc)
##print(roundy)
##        
##print(roundy[1])
##for kk in range(10):
##    print(np.random.randint(0, 3))
