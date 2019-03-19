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


state =3
#print(np.argmax(Q[3]))
#print(max(Q[3]))

#t = 1#float(list(range(1, 21)))
#print(t)
hh=[]
for pp in range(1000):
    #epsilon = float(np.exp(-0.008*pp))
    epsilon = float(np.exp(-0.009*pp))
    
    hh.append(epsilon)


#print(hh)

plt.plot(hh)
plt.show()
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
