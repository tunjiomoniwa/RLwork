
  
import numpy as np
import pylab as plt

def range_val(start, stop=None, step=None):
    if stop == None:
        stop = start + 0.0
        start = 0.0
    if step == None:
        step = 1.0
    while start < stop:
        yield start
        start += step


delta=[]    

for x in range_val(-33, 100, 0.1):
   delta.append(x)

#print(Power_sensor)

Power_sensor = 0.1
Power_relay = 0.33 
#dist_sensor =40+delta
#dist_dest = 35 +delta
alpha = 3
Noise = 2*(10**-7)
gamma = 1
outage=[]

for ii in range(len(delta)):
    PP= np.sqrt((Noise * gamma)/(Power_relay *(35 + delta[ii])**(-alpha)))
    ZZ = (-Noise * gamma)/(Power_sensor *(40+delta[ii])**(-alpha))
    p_out = 1 - (1 + 2* (PP**2) * np.log(2))*(np.exp(ZZ))
    p_out = np.max([0,p_out])

    #p_out = 1 - (1 + 2* ((np.sqrt((Noise * gamma)/(Power_relay[ii] *(dist_dest)**(-alpha))))**2) * np.log(2))*(np.exp((-Noise * gamma)/(Power_sensor *(dist_sensor)**(-alpha))))
    outage.append(p_out)

#print(outage)
plt.plot(delta, outage, 'r', linewidth=1.0)
plt.ylabel('Outage Probability')
plt.xlabel('Tuning delta')
plt.show()
     

