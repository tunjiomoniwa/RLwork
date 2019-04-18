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
    def __init__(self, deltaval, pw1, pw2, pw3, pw4, pw5, pw6):
        ##Paremeters of actions
        self.cdelta = deltaval#0.000001#0.25 # in meters
        self.p1 = pw1 #0.001 # in watts
        self.p2 = pw2 #0.01
        self.p3 = pw3 #0.15
        self.p4 = pw4 #0.2
        self.p5 = pw5 #0.25
        self.p6 = pw6 #0.3
        #self.plev = [0.001, 0.01, 0.15, 0.2, 0.25, 0.3]

        ## energy drain J expressed as percentage
        self.ed1 = 0.006
        self.ed2 = 0.004
        self.ed3 = 0.001
        self.ed4 = 0.002
        self.ed5 = 0.003
        self.ed6 = 0.005
        self.ed7 = 0.007
        self.ed8 = 0.009
        self.ed9 = 0 #not moving or transmiting


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

        self.min_nebo = 0
        self.max_nebo = 2 #means greater than one nebo

        #new attributes
        self.obs = np.array([np.random.randint(0, 80), np.random.randint(95, 100), np.random.randint(0, 3)])
        self.ECI = 0
        self.ECF = 0
        self.sum_pack = 0
        self.final_pack = 0
        self.deltak = np.random.randint(-5, 5)
        self.sensorpower= random.uniform(0, 0.3)
        self.packets_holder = []
        #fog_energy_holder = []
        self.final_packets_holder = []
        self.outage, self.ef, self.neigbour_support = self.obs
        self.done = bool(self.outage< self.goal_outage and self.ef>self.goal_ef and self.neigbour_support == 0)
        self.dead = bool(self.ef == 0)


        self.low = np.array([self.min_outage, self.min_energy_fog, self.min_nebo])
        self.high = np.array([self.max_outage, self.max_energy_fog, self.max_nebo])

        self.action_space = spaces.Discrete(3)
        self.trigger_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.iteration_steps = 100000
        self.episodes=100

        self.alpha = 0.1
        self.gamma =0.9

        #len_action=8
        #len_states =100
        self.buckets =(3,3,3,) # learn

        #Q = np.zeros(shape=[len_states, len_action], dtype=np.float32)
        self.Q = np.zeros(self.buckets + (self.action_space.n,))

        #Q = defaultdict(lambda: np.zeros(action_space.n))




    def step(self, action, IoTTrigger):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        assert self.trigger_space.contains(IoTTrigger), "%r (%s) invalid" % (IoTTrigger, type(IoTTrigger))
        
        #global self.obs:#, self.ECF, self.ECI, self.deltak, self.delta, self.sensorpower
        self.outage, self.ef, self.neigbour_support = self.obs 
        if action==0:
            self.delta = self.cdelta
            self.ECF= self.ed1 #Energy consumed by fog node
            #self.ef -= self.ECF
        elif action==1:
            self.delta = -self.cdelta
            self.ECF= self.ed2
            #self.ef -= self.ECF
        elif action==2: #not moving or transmiting
            self.ECF= self.ed9
            self.ef -= self.ECF

        if IoTTrigger==0:
            self.sensorpower = self.p1
            self.ECI= self.ed3
        elif IoTTrigger==1:
            self.sensorpower = self.p2
            self.ECI= self.ed4
        elif IoTTrigger==2:
            self.sensorpower = self.p3
            self.ECI= self.ed5
        elif IoTTrigger==3:
            self.sensorpower = self.p4
            self.ECI= self.ed6
        elif IoTTrigger==4:
            self.sensorpower = self.p5
            self.ECI= self.ed7
        elif IoTTrigger==5:
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
            self.outage = np.max([0,self.p_out])
        elif action == 2:
            self.p_out = 100
            self.outage = np.max([0,self.p_out])


        if self.ef==0:
            self.outage = self.max_outage
            
        self.outage = np.clip(self.outage, self.min_outage, self.max_outage)


        self.ef -= self.ECF
        
        self.ef = np.clip(self.ef, self.min_energy_fog, self.max_energy_fog)
        
        self.neigbour_support = np.random.randint(0, 3)
        
        self.done = bool(self.outage< self.goal_outage and self.ef>self.goal_ef)
        self.dead = bool(self.ef == 0)
        if self.done:
            rew = 100
        else:
            rew = 0
        self.reward = rew


        self.obs = (self.outage, self.ef, self.neigbour_support)
        return np.array(self.obs), self.reward, self.done, self.dead, {}
        

        
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


    def runRL(self,colorx, cutoff, labelx):
        self.packets_holder = []
        #fog_energy_holder = []
        self.final_packets_holder = []

        for epi in range(self.episodes):

            #cutoff is to minimize


            self.ECI = 0
            self.ECF = 0
            self.deltak = np.random.randint(-5, 5)
            self.sensorpower= random.uniform(0, 0.3)
            #self.delta=0.1
            self.dd=0
            #np.random.randint() --discrete uniform distribution
            self.obs = np.array([np.random.randint(0, 80), np.random.randint(95, self.max_energy_fog), np.random.randint(0, 2)])
            

            cur_action = self.action_space.sample()
            trigger = self.trigger_space.sample()
            obs, reward, done, dead, _ = self.step(cur_action, trigger)
            #print(obs)

            current_state  = self.grouping(obs)
            #print('state before',current_state)
            
            

            
            iter=0
            self.sum_pack = 0
            self.final_pack = 0
            while ((iter < self.iteration_steps) and  not done):#(current_state[0]>=8 and current_state[1]>0 and current_state[2]> 0)): #current_state[0]!= 0):

                iter+=1
              
                #linear
                #epsilon =1-(epi/1000)

                #exp decay
                self.epsilon =float(np.exp(-0.009*epi))
                alpha=0.1
                gamma =0.9

                #print(epsilon)
                action = self.select_action(self.epsilon, current_state)
                triggerx = self.trigger_space.sample()
                obs, reward, done, dead, _ = self.step(action, triggerx)
                 
                

                #do the mapping from obs to state

                
                new_state = self.grouping(obs)
                reward_tj = reward#self.map_reward(new_state)
                #print(obs[0])
                #print(new_state)
                self.sum_pack+=(100 - obs[0])
                
           
                # do learning thingy

                  
                if epi< (2*self.episodes/3):
                    # update q values
                    self.update_q(current_state, new_state, action, reward_tj, alpha, gamma)

                
                #save current state
                current_state = new_state
                

                if done:
                    #print("Reached goal state")
                    #print(reward_tj)
                    break
                if dead:
                    #print("No more communications")
                    break
            self.ave_pack = self.sum_pack/iter
            self.final_pack = (100 - obs[0])
            #print("End of episode #",epi, "  in ", iter , "iterations")
            self.packets_holder.append(self.ave_pack)
            self.final_packets_holder.append(self.final_pack)
        #print(self.final_packets_holder)  
        #self.line2, =plt.plot(self.packets_holder, label=labelx)
        self.line1, =plt.plot(self.final_packets_holder, label=labelx)
        plt.setp(self.line1, color= colorx, linewidth=1.0)
        #plt.setp(self.line2, color= colory, linewidth=1.0,  linestyle='dashed')
        


        


dc1 = FogIoT(0.25, 0.001, 0.01, 0.15, 0.2, 0.25, 0.3)
data1 = dc1.runRL('b', 1000, "Agent - 1")

dc2 = FogIoT(0.25, 0.001, 0.01, 0.15, 0.2, 0.25, 0.3)
data2 = dc2.runRL('g', 1000, "Agent - 2")

arr_fogdc1= dc1.final_packets_holder
arr_fogdc2= dc2.final_packets_holder
#print(arr_fog1)
#print(arr_fog2)

decentralized = np.maximum(arr_fog1dc, arr_fogdc2)  #packets received successfully

#####
sumfogdc1 = np.sum(arr_fogdc1)
sumfogdc2 = np.sum(arr_fogdc2)

RLdecentralized = np.sum(decentralized)


print("Sum of packets fog 1#", sumfogdc1)
print("Sum of packets fog 2#", sumfogdc2)

print("Sum of packets Decentralized RL scheme#", RLdecentralized)




plt.legend([kk1.line1, kk2.line1], ["Agent - 1", "Agent - 2"])
plt.ylabel('Packets successfully  transmitted (%)')
plt.xlabel('Episodes')
##
plt.show()

