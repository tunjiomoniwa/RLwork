import numpy as np
import pylab as plt
import math
import random



Q = [[7, 5],
                       [4, 2],
                       [5, 3],
                       [4, 6],
                       [2, 4],
                       [5, 2]]


state =3
#print(np.argmax(Q[3]))
#print(max(Q[3]))

t = 200#float(list(range(1, 21)))
#print(t)
epsilon = float(np.exp(-0.02*t))
#print(epsilon)

plt.plot(t,epsilon)


low=-0.6
high=-0.4
for ww in range(3):
    bb=random.uniform(low, high)
    print(bb)

cc=np.array([random.uniform(-0.6, -0.4), 0])
#print(cc)
