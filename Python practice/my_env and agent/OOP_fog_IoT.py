import numpy as np
import pylab as plt
from collections import deque
from collections import defaultdict
from tjays import spaces
import seeding
import math
import random
from random import randint

class FogIoT:
    def __init__(self, deltaval):
        ##Paremeters of actions
        self.cdelta = deltaval#0.000001#0.25 # in meters
        self.p1 = 0.001 # in watts
        self.p2 = 0.01
        self.p3 = 0.15
        self.p4 = 0.2
        self.p5 = 0.25
        self.p6 = 0.3

        ## energy drain J expressed as percentage
        self.ed1 = 0.006
        self.ed2 = 0.004
        self.ed3 = 0.001
        self.ed4 = 0.002
        self.ed5 = 0.003
        self.ed6 = 0.005
        self.ed7 = 0.007
        self.ed8 = 0.009


        self.min_outage = 0
        self.max_outage = 100
        self.min_energy_fog = 0
        self.max_energy_fog = 100
        self.min_energy_IoT = 0
        self.max_energy_IoT = 100
        self.min_delta = -35
        self.max_delta = 35
        self.goal_outage = 5 # 5% tolerable outage
        self.goal_ef = 0
        self.goal_ei = 0

        self.low = np.array([self.min_outage, self.min_energy_fog, self.min_energy_IoT])
        self.high = np.array([self.max_outage, self.max_energy_fog, self.max_energy_IoT])

        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.iteration_steps = 100000
        self.episodes=800

        self.alpha = 0.1
        self.gamma =0.9

        #len_action=8
        #len_states =100
        self.buckets =(50,10,10,) # learn

        #Q = np.zeros(shape=[len_states, len_action], dtype=np.float32)
        self.Q = np.zeros(self.buckets + (self.action_space.n,))

        #Q = defaultdict(lambda: np.zeros(action_space.n))




    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        
        #global self.obs, self.ECF, self.ECI, self.deltak, self.delta, self.sensorpower
        outage, ef, ei = self.obs 
        if action==0:
            self.delta = self.cdelta
            self.ECF= self.ed1 #Energy consumed by fog node
        elif action==1:
            self.delta = -self.cdelta
            self.ECF= self.ed2
        elif action==2:
            self.sensorpower = self.p1
            self.ECI= self.ed3
        elif action==3:
            self.sensorpower = self.p2
            self.ECI= self.ed4
        elif action==4:
            self.sensorpower = self.p3
            self.ECI= self.ed5
        elif action==5:
            self.sensorpower = self.p4
            self.ECI= self.ed6
        elif action==6:
            self.sensorpower = self.p5
            self.ECI= self.ed7
        else:
            self.sensorpower = self.p6
            self.ECI= self.ed8

        if action==0 or action==1:
            self.Power_sensor = 0.1
            self.Power_relay = 0.30 
            self.alpha = 3
            self.Noise = 2*(10**-7)
            self.gamma = 1

            self.deltak += self.delta
            self.deltak = np.clip(self.deltak, self.min_delta, self.max_delta)
            self.PP= np.sqrt((self.Noise * self.gamma)/(self.Power_relay *(35 + self.deltak)**(-self.alpha)))
            self.ZZ = (-self.Noise * self.gamma)/(self.Power_sensor *(40+self.deltak)**(-self.alpha))
            self.p_out = 100*(1 - (1 + 2* (self.PP**2) * np.log(2))*(np.exp(self.ZZ)))
            outage = np.max([0,self.p_out])
        else:
            self.Power_relay = 0.30 
            self.dist_sensor =40
            self.dist_dest = 35
            self.alpha = 3
            self.Noise = 100*(10**-7)
            self.gamma = 1

            self.PP= np.sqrt((self.Noise * self.gamma)/(self.Power_relay *(self.dist_dest)**(-self.alpha)))
            self.ZZ = (-self.Noise * self.gamma)/(self.sensorpower *(self.dist_sensor)**(-self.alpha))
            self.p_out = 100*(1 - (1 + 2* (self.PP**2) * np.log(2))*(np.exp(self.ZZ)))
            outage = np.max([0,self.p_out])

        if ef==0 or ei==0:
            outage = self.max_outage
            
        outage = np.clip(outage, self.min_outage, self.max_outage)


        ef -= self.ECF
        ei -= self.ECI
        ef = np.clip(ef, self.min_energy_fog, self.max_energy_fog)
        ei = np.clip(ei, self.min_energy_IoT, self.max_energy_IoT)
        
        done = bool(outage< self.goal_outage and ef>self.goal_ef and ei> self.goal_ei)
        dead = bool(ef == 0 or ei == 0)
        if done:
            rew = 100
        else:
            rew = 0
        reward = rew


        self.obs = (outage, ef, ei)
        return np.array(self.obs), reward, done, dead, {}
        

    def grouping(self, obs):
        upper_bounds = [self.observation_space.high[0], self.observation_space.high[1], self.observation_space.high[2]]
        lower_bounds = [self.observation_space.low[0], self.observation_space.low[1], self.observation_space.low[2]]
        ratios = [(obs[i] + abs(lower_bounds[i]))/ (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.buckets[i]-1)*ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)




    def update_q(self, current_state, new_state, action, reward, alpha, gamma):
        self.Q[current_state][action] = (1- alpha)*self.Q[current_state][action] + alpha*(reward + self.gamma* max(self.Q[new_state]))
        


    def select_action(self, epsilon, state):
        '''
        If the random number is greater than epsilon
        then we exploit else we explore.
        '''
       
        if random.random() < epsilon:
            action = self.action_space.sample() # Explore action space
        else:
            action = np.argmax(self.Q[state]) # Exploit learned values
        return action



        

    def run(self,colorx):
        #packets_holder = []
        #fog_energy_holder = []
        self.IoT_energy_holder = []

        for epi in range(self.episodes):

            

            if epi < self.episodes/2:
                self.ECI = 0
                self.ECF = 0
                self.deltak = np.random.randint(-5, 5)
                self.sensorpower= random.uniform(0, 0.3)
                #self.delta=0.1
                self.dd=0
                #np.random.randint() --discrete uniform distribution
                self.obs = np.array([np.random.randint(0, 60), np.random.randint(65, self.max_energy_fog), np.random.randint(65, self.max_energy_IoT)])
                #obs = np.array([np.random.randint(min_outage, max_outage), np.random.randint(min_energy_fog, max_energy_fog), np.random.randint(min_energy_IoT, max_energy_IoT)])
            else:
                self.ECI = 0
                self.ECF = 0
                self.deltak = np.random.randint(-1, 1)
                self.sensorpower= random.uniform(0, 0.3)
                #delta=0.1
                self.dd=0
                self.obs = (30, 90, 90)

            cur_action = self.action_space.sample()
            obs, reward, done, dead, _ = self.step(cur_action)
            #print(obs)

            current_state  = self.grouping(obs)
            #print('state before',current_state)
            
            

            
            iter=0
            #sum_pack = 0
            while ((iter < self.iteration_steps) and  not done):#(current_state[0]>=8 and current_state[1]>0 and current_state[2]> 0)): #current_state[0]!= 0):

                iter+=1
              
                #linear
                #epsilon =1-(epi/1000)

                #exp decay
                epsilon =float(np.exp(-0.0015*epi))
                alpha=0.1
                gamma =0.9

                #print(epsilon)
                action = self.select_action(epsilon, current_state)
                obs, reward, done, dead, _ = self.step(action)
                 
                

                #do the mapping from obs to state

                
                new_state = self.grouping(obs)
                reward_tj = reward#self.map_reward(new_state)
                #print(obs[0])
                #print(new_state)
                
           
                # do learning thingy

                  
                if epi< (2*self.episodes/3):
                    # update q values
                    self.update_q(current_state, new_state, action, reward_tj, alpha, gamma)

                
                #save current state
                current_state = new_state
                

                if done:
                    print("Reached goal state")
                    #print(reward_tj)
                    break
                if dead:
                    print("No more communications")
                    break
            self.IoT_cons = 100 - obs[2]
            print("End of episode #",epi, "  in ", iter , "iterations")
            #aa.append(iter)
            self.IoT_energy_holder.append(self.IoT_cons)
        line=plt.plot(self.IoT_energy_holder)
        plt.setp(line, color= colorx, linewidth=1.0)

        

kk1 = FogIoT(0.0001)
data1 = kk1.run('r')

kk2 = FogIoT(0.25)
data2 = kk2.run('b')


plt.ylabel('Energy consumed by IoT end-device (%)')
plt.xlabel('Episodes')

plt.show()
    



