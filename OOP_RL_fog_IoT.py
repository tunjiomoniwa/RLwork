import numpy as np
import pylab as plt
from scipy.stats import norm
from scipy import stats

class FogAgent:

    def __init__(self, state_list, initial_state, gamma):
    
    
        goal= 4

        MATRIX_SIZE =9

        self.initial_state = initial_state
        self.gamma =gamma
        self.state_list = state_list
        
    
        #TO MATRIX

        R=np.matrix(np.ones(shape=(MATRIX_SIZE, MATRIX_SIZE)))
        R*=0

        #Assign reward to specific actions

        for self.state in self.state_list:
            #print(state)
            if self.state[1] == goal:
                R[self.state] = 25
            else:
                R[self.state] = 0

            if self.state[0] == goal:
                R[self.state[::-1]] =25
            else:
                R[self.state[::-1]] = 0
        R[goal,goal] = 25
        
        self.R = R
        ################

        Q = np.matrix(np.zeros([MATRIX_SIZE, MATRIX_SIZE]))
        self.Q = Q
        
    

        self.scores = []
        iteration_steps = 1000
        for self.i in range(iteration_steps):
            self.current_state_row = self.R[self.state,]
            av_act = np.where(self.current_state_row >= 0)[1]
            self.av_act = av_act

            action = int(np.random.choice(self.av_act,1))
            self.action =action

            
            self.current_state = np.random.randint(0, int(Q.shape[0]))            
            self.max_index = np.where(Q[self.action,] == np.max(self.Q[self.action,]))[1]
            if self.max_index.shape[0] >1:
                self.max_index = int(np.random.choice(self.max_index, size = 1))
            else:
                self.max_index = int(self.max_index)
            self.max_value = self.Q[self.action, self.max_index]
                
            self.Q[self.current_state, self.action] = self.R[self.current_state, self.action] + self.gamma* self.max_value
            #print('max_value', self.R[self.current_state, self.action] + self.gamma* self.max_value)
            if (np.max(self.Q) > 0):
                self.score = np.sum(self.Q/np.max(self.Q)*25)
            else:
                self.score=0
            self.scores.append(self.score)
        
            
            


fog_list = []
agents_num =1000
for pp in range(agents_num):
    pp= FogAgent([(0,1), (0,3), (1,2), (1,4), (2,5), (3,4), (4,5), (4,7), (5,8), (6,7), (7,8)],2,0.8)
    fog_list.append(pp)

kk=[]
for tt in fog_list:
    kk.append(np.sum(tt.scores))
#print(kk)



my_xticks = range(agents_num)#['fog1', 'fog2', 'fog3', 'fog4', 'fog5', 'fog6']
plt.figure(1)
plt.subplot(221)
plt.bar(my_xticks, kk/np.max(kk)*1, align='center', alpha=0.9)
#plt.axis([-0.5, 5.5, 0.89, 1.01])
plt.title('a) Average reward.')
plt.ylabel('Average Reward')
plt.xlabel('Fog Agents')
#plt.show()


plt.subplot(222)
for jj in fog_list:
    #print(np.sum(tt.scores))
    plt.plot(jj.scores/np.max(jj.scores)*1, 'r', linewidth=1.0) 

plt.title('b) Accumulated reward.')
plt.ylabel('Accumulated Reward')
plt.xlabel('Episodes')
plt.grid(True)
#plt.show()

res = stats.relfreq(kk/np.max(kk)*1, numbins=agents_num)
#print(res)
x = res.lowerlimit + np.linspace(0, res.binsize*res.frequency.size, res.frequency.size)
#print(x)
plt.subplot(212)
plt.bar(x, res.frequency, width=res.binsize)
plt.title('c) Frequency.')
plt.ylabel('Frequency')
plt.xlabel('Normalized Reward')

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5, wspace=0.45)
plt.show()
