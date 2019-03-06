import numpy as np
import pylab as plt
import math
import random
from random import randint

def  pickBestAction(state):
    qvalues = [Q[state][0], Q[state][1], Q[state][2]]

    maxQVal = 0
    bestAction = randint(0,2)

    for i in range(3):
        print(qvalues[i])
        if qvalues[i]>maxQVal:
            maxQVal = qvalues[i]
            bestAction = i

    return bestAction


Q = [[7, 5, 4],
                       [4, 2, 8],
                       [5, 3, 9],
                       [4, 6, 0],
                       [2, 4, 4],
                       [0, 0, 0]]

state =5

action = pickBestAction(state)

print(action)
