import numpy as np
import pylab as plt
from matplotlib.font_manager import FontProperties
from math import sqrt

fx = []
fy = []
for fog in range(1):
    fx.append(np.random.randint(1,11))
    fy.append(np.random.randint(1,11))

num_sensors = np.random.randint(1,11)   
x=[]
y=[]
for sensor in range(num_sensors):
    x.append(np.random.randint(1,11))
    y.append(np.random.randint(1,11))

##print(x)
##print(y)
dist_list = []
for xx in range(num_sensors):
    out = np.sqrt((x[xx] - fx[0])**2  + (y[xx] - fy[0])**2)
    dist_list.append(out)
     
print(dist_list)

action = ['hi', 'mo', 'lo']
act_index =np.random.randint(1,3)
print('action was',action[act_index])
energy_wasted = []
for inodes in range(num_sensors):
    if dist_list[inodes] >= 0 and dist_list[inodes] < 3:
        if action[act_index] == 'hi':
            print('LDHP')
            energy_spent = 20
            energy_wasted.append(energy_spent)
        elif action[act_index] == 'mo':
            print('LDMP')
            energy_spent = 10
            energy_wasted.append(energy_spent)
        else:
            print('LDLP')
            energy_spent = 0
            energy_wasted.append(energy_spent)
        
    elif dist_list[inodes] >= 3 and dist_list[inodes] < 7:
        if action[act_index] == 'hi':
            print('MDHP')
            energy_spent = 10
            energy_wasted.append(energy_spent)
        elif action[act_index] == 'mo':
            print('MDMP')
            energy_spent = 0
            energy_wasted.append(energy_spent)
        else:
            print('MDLP')
            energy_spent = 10
            energy_wasted.append(energy_spent)
    else:
        if action[act_index] == 'hi':
            print('HDHP')
            energy_spent = 0
            energy_wasted.append(energy_spent)
        elif action[act_index] == 'mo':
            print('HDMP')
            energy_spent = 10
            energy_wasted.append(energy_spent)
        else:
            print('HDLP')
            energy_spent = 10
            energy_wasted.append(energy_spent)

print(energy_wasted)
plt.plot(x,y,'ro',  markersize=2, label='IoT sensors')
plt.plot(fx,fy,'bs', label='Mobile Fog-Relay Agents (MFRA)')
plt.gca().axes.get_yaxis().set_visible(False)
plt.gca().axes.get_xaxis().set_visible(False)
plt.legend(bbox_to_anchor=(0.2,0), title="Key", loc= "upper left")

plt.grid(True)
plt.show()
