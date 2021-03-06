import numpy as np
import pylab as plt
from collections import deque
from collections import defaultdict
from tjays import spaces
import seeding
import math
import random
from random import randint


##Paremeters of actions
cdelta = 0.25 # in meters
p1 = 0.001 # in watts
p2 = 0.01
p3 = 0.15
p4 = 0.2
p5 = 0.25
p6 = 0.3

## energy drain J expressed as percentage
ed1 = 0.006
ed2 = 0.004
ed3 = 0.001
ed4 = 0.002
ed5 = 0.003
ed6 = 0.005
ed7 = 0.007
ed8 = 0.009


min_outage = 0
max_outage = 100
min_energy_fog = 0
max_energy_fog = 100
min_energy_IoT = 0
max_energy_IoT = 100
min_delta = -35
max_delta = 35
goal_outage = 5 # <5% tolerable outage
goal_ef = 0 # >0% so long as the fog agent is alive, it moves and transmits
goal_ei = 0 # >0% so long as the IoT sensor is alive, it transmits

low = np.array([min_outage, min_energy_fog, min_energy_IoT])
high = np.array([max_outage, max_energy_fog, max_energy_IoT])

action_space = spaces.Discrete(8)
observation_space = spaces.Box(low, high, dtype=np.float32)

iteration_steps = 100000
episodes=800
 
alpha = 0.1
gamma =0.9

buckets =(50,10,10,) # learn

#Q = np.zeros(shape=[len_states, len_action], dtype=np.float32)
Q = np.zeros(buckets + (action_space.n,))

#Q = defaultdict(lambda: np.zeros(action_space.n))

print(Q)




def step(action):
    assert action_space.contains(action), "%r (%s) invalid" % (action, type(action))
    
    global obs, ECF, ECI, deltak, delta, sensorpower
    outage, ef, ei = obs 
    if action==0:
        delta = cdelta
        ECF= ed1 #Energy consumed by fog node
    elif action==1:
        delta = -cdelta
        ECF= ed2
    elif action==2:
        sensorpower = p1
        ECI= ed3
    elif action==3:
        sensorpower = p2
        ECI= ed4
    elif action==4:
        sensorpower = p3
        ECI= ed5
    elif action==5:
        sensorpower = p4
        ECI= ed6
    elif action==6:
        sensorpower = p5
        ECI= ed7
    else:
        sensorpower = p6
        ECI= ed8

    if action==0 or action==1:
        Power_sensor = 0.1
        Power_relay = 0.30 
        alpha = 3
        Noise = 2*(10**-7)
        gamma = 1

        deltak += delta
        deltak = np.clip(deltak, min_delta, max_delta)
        PP= np.sqrt((Noise * gamma)/(Power_relay *(35 + deltak)**(-alpha)))
        ZZ = (-Noise * gamma)/(Power_sensor *(40+deltak)**(-alpha))
        p_out = 100*(1 - (1 + 2* (PP**2) * np.log(2))*(np.exp(ZZ)))
        outage = np.max([0,p_out])
    else:
        Power_relay = 0.30 
        dist_sensor =40
        dist_dest = 35
        alpha = 3
        Noise = 100*(10**-7)
        gamma = 1

        PP= np.sqrt((Noise * gamma)/(Power_relay *(dist_dest)**(-alpha)))
        ZZ = (-Noise * gamma)/(sensorpower *(dist_sensor)**(-alpha))
        p_out = 100*(1 - (1 + 2* (PP**2) * np.log(2))*(np.exp(ZZ)))
        outage = np.max([0,p_out])

    if ef==0 or ei==0:
        outage = max_outage
        
    outage = np.clip(outage, min_outage, max_outage)


    ef -= ECF
    ei -= ECI
    ef = np.clip(ef, min_energy_fog, max_energy_fog)
    ei = np.clip(ei, min_energy_IoT, max_energy_IoT)
    
    done = bool(outage< goal_outage and ef>goal_ef and ei> goal_ei)
    dead = bool(ef == 0 or ei == 0)
    
    rew = -1.0

    obs = (outage, ef, ei)
    return np.array(obs), rew, done, dead, {}
    #return obs, rew, done, {}



def grouping(obs):
    upper_bounds = [observation_space.high[0], observation_space.high[1], observation_space.high[2]]
    lower_bounds = [observation_space.low[0], observation_space.low[1], observation_space.low[2]]
    ratios = [(obs[i] + abs(lower_bounds[i]))/ (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
    new_obs = [int(round((buckets[i]-1)*ratios[i])) for i in range(len(obs))]
    new_obs = [min(buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
    return tuple(new_obs)




def map_reward(state):
    if done:
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

deadnodes = 0
packets_holder = []
fog_energy_holder = []
IoT_energy_holder = []

for epi in range(episodes):

    
    if epi < episodes/2:
        ECI = 0
        ECF = 0
        deltak = np.random.randint(-5, 5)
        sensorpower= random.uniform(0, 0.3)
        delta=0.1
        dd=0
        #np.random.randint() --discrete uniform distribution
        obs = np.array([np.random.randint(0, 60), np.random.randint(65, max_energy_fog), np.random.randint(65, max_energy_IoT)])
        #obs = np.array([np.random.randint(min_outage, max_outage), np.random.randint(min_energy_fog, max_energy_fog), np.random.randint(min_energy_IoT, max_energy_IoT)])
    else:
        ECI = 0
        ECF = 0
        deltak = np.random.randint(-1, 1)
        sensorpower= random.uniform(0, 0.3)
        delta=0.1
        dd=0
        obs = (30, 90, 90)


    cur_action = action_space.sample()
    obs, reward, done, dead, _ = step(cur_action)
    #print(obs)

    current_state  = grouping(obs)
    #print('state before',current_state)
    
    

    
    
    iter=0
    sum_pack = 0
    while ((iter < iteration_steps) and  not done):#(current_state[0]>=8 and current_state[1]>0 and current_state[2]> 0)): #current_state[0]!= 0):

        iter+=1
      
        #linear
        #epsilon =1-(epi/1000)

        #exp decay
        epsilon =float(np.exp(-0.0015*epi))

        #print(epsilon)
        action = select_action(epsilon, current_state, Q)
        obs, reward, done, dead, _ = step(action)
         
        

        #do the mapping from obs to state

        
        new_state = grouping(obs)
        reward_tj = map_reward(new_state)
        #print(obs)
        #print(new_state)
        sum_pack+=obs[0]
   
        # do learning thingy

          
        if epi<(2*episodes/3):
            # update q values
            update_q(current_state, new_state, action, reward_tj, alpha, gamma)

        
        #save current state
        current_state = new_state
        

        if done:
            print("Reached goal state")
            #print(reward_tj)
            break
        if dead:
            print("No more communications")
            dd=1
            break
    deadnodes += dd
    print("End of episode #",epi, "  in ", iter , "iterations")
    aa.append(iter)
    
    

     
         
##print("The Q matrix is: \n ")
##print(Q)
print(deadnodes)

line=plt.plot(aa)
#line2=plt.plot(deadfog)
plt.setp(line, color='r', linewidth=1.0)
plt.ylabel('Iteration steps')
plt.xlabel('Episodes')

plt.show()
    



