###Dictionaries in python
##
##d = {} # or dict()
##
##d["Tom"] = 24
##d["Jason"] = 3
##d["Jamin"] = 1
##
##print(d["Tom"])
##
###alternatively
##
##
### how to iterate over key-value pairs
##
##for key, value in d.items():
##    print(key)
##    print(value)



import numpy as np
import pylab as plt
from matplotlib.font_manager import FontProperties


x=[]
y=[]
for sensor in range(60):
    x.append(np.random.randint(30,71))
    y.append(np.random.randint(30,71))

fx1 = []
fy1 = []
fx2 = []
fy2 = []
fx3 = []
fy3 = []
fx4 = []
fy4 = []
fx5 = []
fy5 = []
for fog in range(2):
    fx1.append(np.random.randint(68,70))
    fy1.append(np.random.randint(30,70))
    fx2.append(np.random.randint(30,32))
    fy2.append(np.random.randint(30,70))
    fy3.append(np.random.randint(30,32))
    fx3.append(np.random.randint(30,70))
    fy4.append(np.random.randint(68,70))
    fx4.append(np.random.randint(30,70))
    fy5.append(np.random.randint(40,60))
    fx5.append(np.random.randint(40,60))


dx1 = []
dy1 = []
dx2 = []
dy2 = []
dx3 = []
dy3 = []
dx4 = []
dy4 = []

for dest in range(1):
    dx1.append(np.random.randint(25,30))
    dy1.append(np.random.randint(25,30))
    dx2.append(np.random.randint(25,30))
    dy2.append(np.random.randint(70,75))
    dy3.append(np.random.randint(25,30))
    dx3.append(np.random.randint(70,75))
    dx4.append(np.random.randint(70,75))
    dy4.append(np.random.randint(70,75))
    

    
plt.plot(x,y,'ro',  markersize=2, label='IoT sensors')
plt.plot(fx1,fy1,'b>', label='Mobile Fog-Relay Agents (MFRA)')
plt.plot(dx1,dy1,'gs', label='Local service provider')
plt.plot(fx2,fy2,'b>',  fx3,fy3,'b>', fx4,fy4,'b>', fx5,fy5,'b>', label='Mobile Fog-Relay Agents (MFRA)')
plt.plot(dx2,dy2,'gs',  dx3,dy3,'gs', dx4,dy4,'gs', label='Local service provider')
plt.gca().axes.get_yaxis().set_visible(True)
plt.gca().axes.get_xaxis().set_visible(True)
plt.legend(['IoT sensor', 'Mobile Fog-Relay Agent (MFRA)', 'Local service provider'])


plt.grid(True)
plt.show()
