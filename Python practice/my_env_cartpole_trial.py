#import gym


import numpy as np
import pylab as plt
from collections import deque
from collections import defaultdict
from tjays import spaces
import seeding
##import tj_env_mountainCar100states
##from tj_env_mountainCar100states import MountainCarEnv
##import tjays


import math
import random
from random import randint



gravity = 9.8
masscart = 1.0
masspole = 0.1
total_mass = (masspole + masscart)
length = 0.5 # actually half the pole's length
polemass_length = (masspole * length)
force_mag = 10.0
tau = 0.02  # seconds between state updates
kinematics_integrator = 'euler'

# Angle at which to fail the episode
theta_threshold_radians = 12 * 2 * math.pi / 360
x_threshold = 2.4

# Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
high = np.array([
    x_threshold * 2,
    np.finfo(np.float32).max,
    theta_threshold_radians * 2,
    np.finfo(np.float32).max])

action_space = spaces.Discrete(2)
observation_space = spaces.Box(-high, high, dtype=np.float32)

#seed()
viewer = None
state = None

steps_beyond_done = None





iteration_steps = 100000
episodes=500
#epsilon =0.5
alpha = 0.1
gamma =0.9
buckets =(1,1,6,12,)
len_action =2

Q = np.zeros(shape=[len_states, len_action], dtype=np.float32)
#Q = np.zeros(buckets + (action_space.n,))
#Q = defaultdict(lambda: np.zeros(action_space.n))

print(Q)


def step(action):
        assert action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        global obs
        x, x_dot, theta, theta_dot = obs
        force = force_mag if action==1 else -force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass
        thetaacc = (gravity * sintheta - costheta* temp) / (length * (4.0/3.0 - masspole * costheta * costheta / total_mass))
        xacc  = temp - polemass_length * thetaacc * costheta / total_mass
        if kinematics_integrator == 'euler':
            x  = x + tau * x_dot
            x_dot = x_dot + tau * xacc
            theta = theta + tau * theta_dot
            theta_dot = theta_dot + tau * thetaacc
        else: # semi-implicit euler
            x_dot = x_dot + tau * xacc
            x  = x + tau * x_dot
            theta_dot = theta_dot + tau * thetaacc
            theta = theta + tau * theta_dot
        obs = (x,x_dot,theta,theta_dot)
        done =  x < -x_threshold \
                or x > x_threshold \
                or theta < -theta_threshold_radians \
                or theta > theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif steps_beyond_done is None:
            # Pole just fell!
            steps_beyond_done = 0
            reward = 1.0
        else:
            if steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            steps_beyond_done += 1
            reward = 0.0

        return np.array(obs), reward, done, {}





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



def mapping(obs):
    upper_bounds = [observation_space.high[0], 0.5, observation_space.high[2], math.radians(50)]
    lower_bounds = [observation_space.low[0], -0.5, observation_space.low[2], -math.radians(50)]
    ratios = [(obs[i] + abs(lower_bounds[i]))/ (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
    new_obs = [int(round((buckets[i]-1)*ratios[i])) for i in range(len(obs))]
    new_obs = [min(buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
    return tuple(new_obs)

def map_reward(state):
    
    if(state==(0, 0, 5, 11)):
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

    obs = np.array([random.uniform(low=-0.05, high=0.05, size=(4,))])

    obs, reward, done, _ = step(cur_action)

    current_state = mapping(obs)

    
    iter=0
    while ((iter < iteration_steps) and current_state!=(0, 0, 5, 11)):

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

        new_state = mapping(obs)
        reward_tj = map_reward(new_state)
   
        # do learning thingy

          
        if epi<300:
            # update q values
            update_q(current_state, new_state, action, reward_tj, alpha, gamma)

        
        #save current state
        current_state = new_state
        #print(obs)
        #print(current_state)

        if(current_state==(0, 0, 5, 11)):
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
    



