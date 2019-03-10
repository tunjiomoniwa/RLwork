import numpy as np
import pylab as plt
from collections import deque
from collections import defaultdict
from tjays import spaces
import seeding
import math
import random
from random import randint


##Paremeters
cdelta = 0.1 # in meters
p1 = 0.01 # in watts
p2 = 0.1
p3 = 0.3
p4 = 0.7
p5 = 0.9
p6 = 1

## energy drain
ed1 = 0.4
ed2 = 0.3
ed3 = 0.1
ed4 = 0.3
ed5 = 0.5
ed6 =0.8
ed7 = 1
ed8 = 2


min_outage = 0
max_outage = 100
min_energy_fog = 0
max_energy_fog = 100
min_energy_IoT = 0
max_energy_IoT = 100

goal_outage = 90

low = np.array([min_outage, min_energy_fog, min_energy_IoT])
high = np.array([max_outage, max_energy_fog, max_energy_IoT])

action_space = spaces.Discrete(8)
observation_space = spaces.Box(low, high, dtype=np.float32)

iteration_steps = 100000
episodes=500
#epsilon =0.5
alpha = 0.1
gamma =0.9

#len_action=8
#len_states =100
buckets =(10,10,10,) # learn

#Q = np.zeros(shape=[len_states, len_action], dtype=np.float32)
Q = np.zeros(buckets + (action_space.n,))
#Q = defaultdict(lambda: np.zeros(action_space.n))

print(Q)


np.array([random.uniform(0, 100), random.uniform(0, 100), random.uniform(0, 100)])



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
    
    global obs, ECF, ECI
    outage, ef, ei = obs 
    if action==1:
        delta = cdelta
        ECF= ed1 #Energy consumed by fog node
    elif action==2:
        delta = -cdelta
        ECF= ed2
    elif action==3:
        sensorpower = p1
        ECI= ed3
    elif action==4:
        sensorpower = p2
        ECI= ed4
    elif action==5:
        sensorpower = p3
        ECI= ed5
    elif action==6:
        sensorpower = p4
        ECI= ed6
    elif action==8:
        sensorpower = p5
        ECI= ed7
    else:
        sensorpower = p6
        ECI= ed8

    if action==1 or action==2:
        Power_sensor = 0.1
        Power_relay = 0.33 
        alpha = 3
        Noise = 2*(10**-7)
        gamma = 1
        
        PP= np.sqrt((Noise * gamma)/(Power_relay *(35 + delta)**(-alpha)))
        ZZ = (-Noise * gamma)/(Power_sensor *(40+delta)**(-alpha))
        p_out = 100*(1 - (1 + 2* (PP**2) * np.log(2))*(np.exp(ZZ)))
        outage = np.max([0,p_out])
    else:
        Power_relay = 0.33 
        dist_sensor =40
        dist_dest = 35
        alpha = 3
        Noise = 2*(10**-7)
        gamma = 1

        PP= np.sqrt((Noise * gamma)/(Power_relay *(dist_dest)**(-alpha)))
        ZZ = (-Noise * gamma)/(sensorpower *(dist_sensor)**(-alpha))
        p_out = 100*(1 - (1 + 2* (PP**2) * np.log(2))*(np.exp(ZZ)))
        outage = np.max([0,p_out])
        
    outage = np.clip(outage, min_outage, max_outage)


    ef -= ECF
    ei -= ECI
    ef = np.clip(ef, min_energy_fog, max_energy_fog)
    ei = np.clip(ei, min_energy_IoT, max_energy_IoT)
    
    done = bool(outage >= goal_outage)
    rew = -1.0

    obs = (outage, ef, ei)
    return np.array(obs), rew, done, {}
    #return obs, rew, done, {}



def grouping(obs):
    upper_bounds = [observation_space.high[0], observation_space.high[1], observation_space.high[2]]
    lower_bounds = [observation_space.low[0], observation_space.low[1], observation_space.low[2]]
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

    ECI = 0
    ECF = 0
    

    obs = np.array([random.uniform(0, 100), random.uniform(95, 100), random.uniform(95, 100)])

    obs, reward, done, _ = step(cur_action)
    print(obs)

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
        #print(obs)
        print(step(action))

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
    


