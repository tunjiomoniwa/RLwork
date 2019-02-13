import numpy as np
import pylab as plt
from scipy.stats import norm
from scipy import stats

class FogAgent:

    def __init__(self, state_list, initial_state, gamma):
    
    
        goal= 11

        MATRIX_SIZE =12

        self.initial_state = initial_state
        self.gamma =gamma
        self.state_list = state_list
        
    
        #TO MATRIX

        R=np.matrix(np.ones(shape=(MATRIX_SIZE, MATRIX_SIZE)))
        R*=-10.2

        #Assign reward to specific actions

        for self.state in self.state_list:
            #print(state)
            if self.state[1] == goal:
                R[self.state] = 100
            else:
                R[self.state] = 0

            if self.state[0] == goal:
                R[self.state[::-1]] =100
            else:
                R[self.state[::-1]] = 0
        R[goal,goal] = 100
        R[0,0] = -50
        self.R = R
        ################

        Q = np.matrix(np.zeros([MATRIX_SIZE, MATRIX_SIZE]))
        self.Q = Q

        learning_rate=0.5       
    

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

            if self.max_index.shape[0] > 1:
                self.max_index = int(np.random.choice(self.max_index, size = 1))
                
            else:
                self.max_index = int(np.max(self.max_index))
            self.max_value = self.Q[self.action, self.max_index]
            
                
            self.Q[self.current_state, self.action] = (1 - learning_rate)*self.Q[self.current_state, self.action] + learning_rate*(self.R[self.current_state, self.action] + self.gamma* self.max_value)

            if (np.max(self.Q) > 0):
                #self.score = np.sum(self.Q/np.max(self.Q)*25)
                self.score = np.sum(self.Q)
            else:
                self.score=0
            self.scores.append(self.score)
        
            
            


fog_list = []
agents_num =2
for pp in range(agents_num):
    #pp= FogAgent([(0,1), (0,3), (1,2), (1,4), (2,5), (3,4), (4,5), (4,7), (5,8), (6,7), (7,8)], 2, 0.8)
    ppp= [(0,0), (0,1), (1,0), (1,2), (2,1), (1,4), (4,1), (3,3), (3,4), (4,3), (4,5), (5,4), (4,7), (7,4), (7,6), (6, 7), (6,6), (7,8), (8,7), (7,10), (10,7), (10,9), (9,10), (9,9), (10,11), (11,10), (10,10), (2,2), (5,5), (8,8), (11,11)]
    pp= FogAgent(ppp, 2, 0.8)
    fog_list.append(pp)

kk=[]
for tt in fog_list:
    kk.append(np.sum(tt.scores))










for jj in fog_list:
    #print(np.sum(tt.scores))
    plt.plot(jj.scores, 'r', linewidth=1.0) 

plt.title('Energy saved with over episodes.')
plt.ylabel('Energy [Joules]')
plt.xlabel('Episodes')
plt.grid(True)
plt.show()

