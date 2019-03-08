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
buckets =(100,100,100,) # learn

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




def map_reward(state):
    for ii in range(90, 100):
        if(state==(0,ii,ii)):
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

goals =  (0,90,90) or (0,91,90) or (0,92,90) or (0,93,90) or (0,94,90) or (0,95,90) or (0,96,90) or (0,97,90) or (0,98,90) or (0,99,90) or (0,90,91) or (0,91,91) or (0,92,91) or (0,93,91) or (0,94,91) or (0,95,91) or (0,96,91) or (0,97,91) or (0,98,91)


        

aa = []
for epi in range(episodes):

    cur_action = action_space.sample()

    ECI = 0
    ECF = 0
    

    obs = np.array([random.uniform(20, 80), random.uniform(95, 100), random.uniform(95, 100)])

    obs, reward, done, _ = step(cur_action)
    print(obs)

    current_state  = grouping(obs)
    print(current_state)
    
    

    
    iter=0
    while ((iter < iteration_steps) and current_state!=goals):

        iter+=1
      
        #linear
        #epsilon =1-(epi/1000)

        #exp decay
        epsilon =float(np.exp(-0.015*epi))

        #print(epsilon)
        action = select_action(epsilon, current_state, Q)
        obs, reward, done, _ = step(action)
        #print(obs)
        

        #do the mapping from obs to state

        
        new_state = grouping(obs)
        reward_tj = map_reward(new_state)
   
        # do learning thingy

          
        if epi<300:
            # update q values
            update_q(current_state, new_state, action, reward_tj, alpha, gamma)

        
        #save current state
        current_state = new_state
        #print(obs)
        #print(current_state)

        if(current_state==goals):
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
    



