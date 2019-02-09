
##PP= np.sqrt((Noise * gamma)/(Power_relay[ii] *(dist_dest)**(-alpha)))
##    ZZ = (-Noise * gamma)/(Power_sensor *(dist_sensor)**(-alpha))
##    p_out = 1 - (1 + 2* (PP**2) * np.log(2))*(np.exp(ZZ))
    
import numpy as np
import pylab as plt

def range_positve(start, stop=None, step=None):
    if stop == None:
        stop = start + 0.0
        start = 0.0
    if step == None:
        step = 1.0
    while start < stop:
        yield start
        start += step

Power_sensor=[]    

for x in range_positve(0.1, 1.0, 0.05):
   Power_sensor.append(x)

#print(Power_sensor)

#Power_sensor = [0.05, 0.15, 0.25, 0.4, 0.6]
Power_relay = 0.33 
dist_sensor =40
dist_dest = 29
alpha = 3
Noise = 2*(10**-7)
gamma = 1
outage=[]

for ii in range(len(Power_sensor)):
    PP= np.sqrt((Noise * gamma)/(Power_relay *(dist_dest)**(-alpha)))
    ZZ = (-Noise * gamma)/(Power_sensor[ii] *(dist_sensor)**(-alpha))
    p_out = 1 - (1 + 2* (PP**2) * np.log(2))*(np.exp(ZZ))
    p_out = np.max([0,p_out])

    #p_out = 1 - (1 + 2* ((np.sqrt((Noise * gamma)/(Power_relay[ii] *(dist_dest)**(-alpha))))**2) * np.log(2))*(np.exp((-Noise * gamma)/(Power_sensor *(dist_sensor)**(-alpha))))
    outage.append(p_out)

print(outage)
plt.plot(Power_sensor, outage, 'r', linewidth=1.0)
plt.ylabel('Outage Probability')
plt.xlabel('Power_{sensor} [Watts]')
plt.show()
     

