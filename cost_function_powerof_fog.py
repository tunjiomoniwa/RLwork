   
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

Power_relay=[]    

for x in range_positve(0.1, 1.0, 0.05):
   Power_relay.append(x)

#print(Power_relay)

#Power_sensor = [0.05, 0.15, 0.25, 0.4, 0.6]
Power_sensor = 0.33 
dist_sensor =40
dist_dest = 29
alpha = 3
Noise = 2*(10**-7)
gamma = 1
outageR=[]

for ii in range(len(Power_relay)):
    PP= np.sqrt((Noise * gamma)/(Power_relay[ii] *(dist_dest)**(-alpha)))
    ZZ = (-Noise * gamma)/(Power_sensor *(dist_sensor)**(-alpha))
    p_out = 1 - (1 + 2* (PP**2) * np.log(2))*(np.exp(ZZ))
    p_out = np.max([0,p_out])

    #p_out = 1 - (1 + 2* ((np.sqrt((Noise * gamma)/(Power_relay[ii] *(dist_dest)**(-alpha))))**2) * np.log(2))*(np.exp((-Noise * gamma)/(Power_sensor *(dist_sensor)**(-alpha))))
    outageR.append(p_out)



Power_sensor=[]  
for x in range_positve(0.1, 1.0, 0.05):
   Power_sensor.append(x)

Power_relay = 0.33 
dist_sensor =40
dist_dest = 29
alpha = 3
Noise = 2*(10**-7)
gamma = 1
outageS=[]

for iis in range(len(Power_sensor)):
    PPS= np.sqrt((Noise * gamma)/(Power_relay *(dist_dest)**(-alpha)))
    ZZS = (-Noise * gamma)/(Power_sensor[iis] *(dist_sensor)**(-alpha))
    p_outS = 1 - (1 + 2* (PPS**2) * np.log(2))*(np.exp(ZZS))
    p_outS = np.max([0,p_out])

    #p_out = 1 - (1 + 2* ((np.sqrt((Noise * gamma)/(Power_relay[ii] *(dist_dest)**(-alpha))))**2) * np.log(2))*(np.exp((-Noise * gamma)/(Power_sensor *(dist_sensor)**(-alpha))))
    outageS.append(p_outS)



def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

r = np.linspace(0, 6, 20)
theta = np.linspace(-0.9 * np.pi, 0.8 * np.pi, 40)
r, theta = np.meshgrid(r, theta)

X =  outageS
Y =  outageR
Z = f(X, Y)

ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none');


##plt.plot(Power_relay, outage, 'r', linewidth=1.0)
##plt.ylabel('Outage Probability')
##plt.xlabel('Power_{relay} [Watts]')
plt.show()
     

