# fognet.py
import numpy as np
import pylab as plt

# =============================================================

####Format print
def my_print(Q):
  rows = len(Q); cols = len(Q[0])
  print("       0      1      2      3      4      5\
      6      7      8     9    10    ")
  for i in range(rows):
    print("%d " % i, end="")
    if i < 10: print(" ", end="")
    for j in range(cols): print(" %6.2f" % Q[i,j], end="")
    print("")
  print("")

def train(F, R, Q, gamma, lrn_rate, goal, ns, max_epochs): #### Training fxn
  tt=0
  maxQv= []
  for i in range(0,max_epochs):
    curr_s = np.random.randint(0,ns)  # random start state
    tt=tt+1
    print(tt)
    maxQv.append(Q.sum())
    #print(Q.sum())
    #my_print(Q)
    while(True):
      next_s = get_rnd_next_state(curr_s, F, ns)
      poss_next_next_states = \
        get_poss_next_states(next_s, F, ns)

      max_Q = -9999.99
      for j in range(len(poss_next_next_states)):
        nn_s = poss_next_next_states[j]
        q = Q[next_s,nn_s]
        if q > max_Q:
          max_Q = q
      # Q = [(1-a) * Q]  +  [a * (rt + (g * maxQ))]
      Q[curr_s][next_s] = ((1 - lrn_rate) * Q[curr_s] \
        [next_s]) + (lrn_rate * (R[curr_s][next_s] + \
        (gamma * max_Q)))

      
      curr_s = next_s
      
      if curr_s == goal:  break

  plt.plot(maxQv)
  plt.show()
  #print(str(maxQv))
    


###Walking the Q matrix
def walk(start, goal, Q):
  curr = start
  print(str(curr) + "->", end="")
  while curr != goal:
    next = np.argmax(Q[curr])
    print(str(next) + "->", end="")
    curr = next
  print("done")

####Format print
def my_print(Q):
  rows = len(Q); cols = len(Q[0])
  print("       0      1      2      3      4      5\
      6      7      8     9    10    ")
  for i in range(rows):
    print("%d " % i, end="")
    if i < 10: print(" ", end="")
    for j in range(cols): print(" %6.2f" % Q[i,j], end="")
    print("")
  print("")

def get_poss_next_states(s, F, ns):
  poss_next_states = []
  for j in range(ns):
    if F[s,j] == 1: poss_next_states.append(j)
  return poss_next_states

def get_rnd_next_state(s, F, ns):
  poss_next_states = get_poss_next_states(s, F, ns)
  next_state = \
    poss_next_states[np.random.randint(0,\
    len(poss_next_states))]
  return next_state


#def my_print(Q, dec): . . . 
#def get_poss_next_states(s, F, ns): . . . 
#def get_rnd_next_state(s, F, ns): . . . 
#def train(F, R, Q, gamma, lrn_rate, goal, ns, max_epochs): . .
#def walk(start, goal, Q): . . . 

# =============================================================

def main():
  np.random.seed(1)
  print("Setting up Network in memory")

  F = np.zero                                                                                                                                                                                                                               s(shape=[11,11], dtype=np.int)  # Feasible action range
  F[0,1] = 1; F[1,0] = 1; F[0,2] = 1; F[2,0] = 1; F[1,4] = 1
  F[4,1] = 1; F[1,3] = 1; F[3,1] = 1; F[2,6] = 1; F[6,2] = 1
  F[2,5] = 1; F[5,2] = 1; F[5,8] = 1; F[8,5] = 1; F[5,7] = 1
  F[7,5] = 1; F[7,9] = 1; F[9,7] = 1; F[7,10] = 1; F[10,7] = 1; F[10,10] = 1
  
  #print(F)
  R = np.zeros(shape=[11,11], dtype=np.int)  # Rewards
  #R[0,1] = -0.1; R[1,0] = -0.1; R[0,2] = -0.1; R[2,0] = -0.1; R[1,4] = -10
  #R[4,1] = -0.1; R[1,3] = -0.1; R[3,1] = -0.1; R[2,6] = -0.1; R[6,2] = -0.1
  #R[2,5] = -0.1; R[5,2] = -0.1; R[5,8] = -0.1; R[8,5] = -0.1; R[5,7] = -0.1
  #R[7,5] = -0.1; R[7,9] = -0.1; R[9,7] = -0.1; R[7,10] = 10; R[10,7] = -0.1; R[10,10] = -0.1

  R[7,10] = 10
  #print(R)

# =============================================================

  Q = np.zeros(shape=[11,11], dtype=np.float32)  # Quality
  
  print("Analyzing network with RL Q-learning")
  start = 0; goal = 10
  ns = 11  # number of states
  gamma = 0.9
  lrn_rate = 0.1
  max_epochs = 1000
  train(F, R, Q, gamma, lrn_rate, goal, ns, max_epochs)
  print("Done ")
  
  print("The Q matrix is: \n ")
  my_print(Q)

  
  print("Using Q to go from 0 to goal (10)")
  walk(start, goal, Q)

if __name__ == "__main__":
    main()

