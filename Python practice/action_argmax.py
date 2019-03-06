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
print(np.argmax(Q[3]))
print(max(Q[3]))

t = 200#float(list(range(1, 21)))
print(t)
epsilon = float(np.exp(-0.015*t))
print(epsilon)

plt.plot(t,epsilon)
