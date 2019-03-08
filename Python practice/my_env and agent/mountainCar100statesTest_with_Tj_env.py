import numpy as np
import pylab as plt
from collections import deque
from collections import defaultdict
from tjays import spaces
import seeding
<<<<<<< HEAD


=======
>>>>>>> d94b4a01cbda845b2e9ff262d175a1302e3e22db
import math
import random
from random import randint



min_position = -1.2
max_position = 0.6
max_speed = 0.07
goal_position = 0.5

low = np.array([min_position, -max_speed])
high = np.array([max_position, max_speed])

action_space = spaces.Discrete(3)
observation_space = spaces.Box(low, high, dtype=np.float32)

iteration_steps = 100000
episodes=500
alpha = 0.1
gamma =0.91

len_action=3
len_states =100
buckets =(10,10,) # learn

Q = np.zeros(shape=[len_states, len_action], dtype=np.float32)
#Q = np.zeros(buckets + (action_space.n,))
#Q = defaultdict(lambda: np.zeros(action_space.n))

print(Q)


np.array([random.uniform(-0.6, -0.4), 0])



def np_random(seed=None):
    if seed is not None and not (isinstance(seed, integer_types) and 0 <= seed):
        raise error.Error('Seed must be a non-negative integer or omitted, not {}'.format(seed))

    seed = create_seed(seed)

    rng = np.random.RandomState()
    rng.seed(_int_list_from_bigint(hash_seed(seed)))
    return rng, seed


def seed(seed=None):
    np_random, seed = seeding.np_random(seed)
    return [seed]

def step(action):
    assert action_space.contains(action), "%r (%s) invalid" % (action, type(action))
    
    global obs
    position, velocity = obs #np.array([random.uniform(-0.6, -0.4), 0])
    velocity += (action-1)*0.001 + math.cos(3*position)*(-0.0025)
    velocity = np.clip(velocity, -max_speed, max_speed)
    position += velocity
    position = np.clip(position, min_position, max_position)
    if (position==min_position and velocity<0): velocity = 0

    done = bool(position >= goal_position)
    rew = -1.0

    obs = (position, velocity)
    return np.array(obs), rew, done, {}
    #return obs, rew, done, {}


def grouping(obs):
    upper_bounds = [observation_space.high[0], observation_space.high[1]]
    lower_bounds = [observation_space.low[0], observation_space.low[1]]
    ratios = [(obs[i] + abs(lower_bounds[i]))/ (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
    new_obs = [int(round((buckets[i]-1)*ratios[i])) for i in range(len(obs))]
    new_obs = [min(buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
    return tuple(new_obs)

def mapping(observe):
    if grouping(obs) == (0, 0):
        state =0
    elif grouping(obs) == (1, 0):
        state =1
    elif grouping(obs) == (2, 0):
        state =2
    elif grouping(obs) == (3, 0):
        state =3
    elif grouping(obs) == (4, 0):
        state =4
    elif grouping(obs) == (5, 0):
        state =5
    elif grouping(obs) == (6, 0):
        state =6
    elif grouping(obs) == (7, 0):
        state =7
    elif grouping(obs) == (8, 0):
        state =8
    elif grouping(obs) == (9, 0):
        state =9
    ##
    elif grouping(obs) == (0, 1):
        state =10
    elif grouping(obs) == (1, 1):
        state =11
    elif grouping(obs) == (2, 1):
        state =12
    elif grouping(obs) == (3, 1):
        state =13
    elif grouping(obs) == (4, 1):
        state =14
    elif grouping(obs) == (5, 1):
        state =15
    elif grouping(obs) == (6, 1):
        state =16
    elif grouping(obs) == (7, 1):
        state =17
    elif grouping(obs) == (8, 1):
        state =18
    elif grouping(obs) == (9, 1):
        state =19
    ##
    elif grouping(obs) == (0, 2):
        state = 20
    elif grouping(obs) == (1, 2):
        state =21
    elif grouping(obs) == (2, 2):
        state =22
    elif grouping(obs) == (3, 2):
        state =23
    elif grouping(obs) == (4, 2):
        state =24
    elif grouping(obs) == (5, 2):
        state =25
    elif grouping(obs) == (6, 2):
        state =26
    elif grouping(obs) == (7, 2):
        state =27
    elif grouping(obs) == (8, 2):
        state =28
    elif grouping(obs) == (9, 2):
        state =29
    ##
    elif grouping(obs) == (0, 3):
        state = 30
    elif grouping(obs) == (1, 3):
        state = 31
    elif grouping(obs) == (2, 3):
        state = 32
    elif grouping(obs) == (3, 3):
        state = 33
    elif grouping(obs) == (4, 3):
        state = 34
    elif grouping(obs) == (5, 3):
        state = 35
    elif grouping(obs) == (6, 3):
        state = 36
    elif grouping(obs) == (7, 3):
        state = 37
    elif grouping(obs) == (8, 3):
        state = 38
    elif grouping(obs) == (9, 3):
        state = 39
    ##
    elif grouping(obs) == (0, 4):
        state = 40
    elif grouping(obs) == (1, 4):
        state = 41
    elif grouping(obs) == (2, 4):
        state = 42
    elif grouping(obs) == (3, 4):
        state = 43
    elif grouping(obs) == (4, 4):
        state = 44
    elif grouping(obs) == (5, 4):
        state = 45
    elif grouping(obs) == (6, 4):
        state = 46
    elif grouping(obs) == (7, 4):
        state = 47
    elif grouping(obs) == (8, 4):
        state = 48
    elif grouping(obs) == (9, 4):
        state = 49
    ##
    elif grouping(obs) == (0, 5):
        state = 50
    elif grouping(obs) == (1, 5):
        state = 51
    elif grouping(obs) == (2, 5):
        state = 52
    elif grouping(obs) == (3, 5):
        state = 53
    elif grouping(obs) == (4, 5):
        state = 54
    elif grouping(obs) == (5, 5):
        state = 55
    elif grouping(obs) == (6, 5):
        state = 56
    elif grouping(obs) == (7, 5):
        state = 57
    elif grouping(obs) == (8, 5):
        state = 58
    elif grouping(obs) == (9, 5):
        state = 59
    ##
    elif grouping(obs) == (0, 6):
        state = 60
    elif grouping(obs) == (1, 6):
        state = 61
    elif grouping(obs) == (2, 6):
        state = 62
    elif grouping(obs) == (3, 6):
        state = 63
    elif grouping(obs) == (4, 6):
        state = 64
    elif grouping(obs) == (5, 6):
        state = 65
    elif grouping(obs) == (6, 6):
        state = 66
    elif grouping(obs) == (7, 6):
        state = 67
    elif grouping(obs) == (8, 6):
        state = 68
    elif grouping(obs) == (9, 6):
        state = 69
    ##
    elif grouping(obs) == (0, 7):
        state = 70
    elif grouping(obs) == (1, 7):
        state = 71
    elif grouping(obs) == (2, 7):
        state = 72
    elif grouping(obs) == (3, 7):
        state = 73
    elif grouping(obs) == (4, 7):
        state = 74
    elif grouping(obs) == (5, 7):
        state = 75
    elif grouping(obs) == (6, 7):
        state = 76
    elif grouping(obs) == (7, 7):
        state = 77
    elif grouping(obs) == (8, 7):
        state = 78
    elif grouping(obs) == (9, 7):
        state = 79
    ##
    elif grouping(obs) == (0, 8):
        state = 80
    elif grouping(obs) == (1, 8):
        state = 81
    elif grouping(obs) == (2, 8):
        state = 82
    elif grouping(obs) == (3, 8):
        state = 83
    elif grouping(obs) == (4, 8):
        state = 84
    elif grouping(obs) == (5, 8):
        state = 85
    elif grouping(obs) == (6, 8):
        state = 86
    elif grouping(obs) == (7, 8):
        state = 87
    elif grouping(obs) == (8, 8):
        state = 88
    elif grouping(obs) == (9, 8):
        state = 89
    ##
    elif grouping(obs) == (0, 9):
        state = 90
    elif grouping(obs) == (1, 9):
        state = 91
    elif grouping(obs) == (2, 9):
        state = 92
    elif grouping(obs) == (3, 9):
        state = 93
    elif grouping(obs) == (4, 9):
        state = 94
    elif grouping(obs) == (5, 9):
        state = 95
    elif grouping(obs) == (6, 9):
        state = 96
    elif grouping(obs) == (7, 9):
        state = 97
    elif grouping(obs) == (8, 9):
        state = 98
    else:# grouping(obs) == (9, 9):
        state = 99
    return state


def map_reward(state):
    
    if(state==99):
        return 100
    else:
        return 0


def update_q(current_state, new_state, action, reward, alpha, gamma):
    Q[current_state][action] = (1- alpha)*Q[current_state][action] + alpha*(reward + gamma* max(Q[new_state]))
    


def select_action(epsilon, state, Q):
    '''
    If the random number is greater than epsilon
    then we exploit else we explore.
    '''
   
    if random.random() < epsilon:
        action = action_space.sample() # Explore action space
    else:
        action = np.argmax(Q[state]) # Exploit learned values
    return action


aa = []
for epi in range(episodes):

    cur_action = action_space.sample()

    if epi<300:
        obs = np.array([random.uniform(-1.2, 0.5), random.uniform(-0.07, 0.07)])
    else:
        obs = np.array([random.uniform(-0.6, -0.4), 0])

    obs, reward, done, _ = step(cur_action)

    p_state = grouping(obs)
    
    current_state = mapping(p_state)

    
    iter=0
    while ((iter < iteration_steps) and current_state!=99):

        iter+=1
      
        #linear
        #epsilon =1-(epi/1000)

        #exp decay
        epsilon =float(np.exp(-0.015*epi))

        #print(epsilon)
        action = select_action(epsilon, current_state, Q)
        obs, reward, done, _ = step(action)
        #print(step(action))

        #do the mapping from obs to state

        new_observe = grouping(obs)
        new_state = mapping(new_observe)
        reward_tj = map_reward(new_state)
   
        # do learning thingy

          
        if epi<300:
            # update q values
            update_q(current_state, new_state, action, reward_tj, alpha, gamma)

        
        #save current state
        current_state = new_state
        #print(obs)
        #print(current_state)

        if(current_state==99):
            print("Reached goal state")
    
    print("End of episode #",epi, "  in ", iter , "iterations")
    aa.append(iter)
    

    
         
print("The Q matrix is: \n ")
print(Q)

line=plt.plot(aa)
plt.setp(line, color='r', linewidth=1.0)
plt.ylabel('Iteration steps')
plt.xlabel('Episodes')
plt.show()
    



