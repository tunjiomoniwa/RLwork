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
for sensor in range(500):
    x.append(np.random.randint(1,5001))
    y.append(np.random.randint(1,5001))

fx = []
fy = []
for fog in range(50):
    fx.append(np.random.randint(1,5001))
    fy.append(np.random.randint(1,5001))

plt.plot(x,y,'ro',  markersize=2, label='IoT sensors')
plt.plot(fx,fy,'bs', label='Mobile Fog-Relay Agents (MFRA)')
plt.gca().axes.get_yaxis().set_visible(False)
plt.gca().axes.get_xaxis().set_visible(False)
plt.legend(bbox_to_anchor=(0.2,0), title="Key", loc= "upper left")

plt.grid(True)
plt.show()
