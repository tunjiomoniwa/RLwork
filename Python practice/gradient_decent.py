
import numpy as np
import pylab as plt
 
import math

cur_x = 0.4 # The algorithm starts at x=3
d1=30
d2 =30
rate = 0.08 # Learning rate
precision = 0.00001 #This tells us when to stop the algorithm
previous_step_size = 1 #
max_iters = 1000 # maximum number of iterations
iters = 0 #iteration counter

#Gradient of our function
df = lambda x: (-2*10**-6)* np.exp((-6.66667*10**-7) *(x + 35)**3) * (x + 35)**2 + (2.66667*10**-12)* np.exp((-6.66667*10**-7 )*(x + 35)**3) *((x + 35)**5)* np.log(0.000816497 * np.sqrt((x + 35)**3)) - (4*10**-6)* np.exp((-6.66667*10**-7)*(x + 35)**3) * ((x + 35)**2)* np.log(0.000816497*(np.sqrt((x + 35)**3)))

rec_pct_gd = []
while previous_step_size > precision and iters < max_iters and  cur_x > 0.01:
    prev_x = cur_x #Store current x value in prev_x
    cur_x = cur_x - rate * df(prev_x) #Grad descent
    previous_step_size = abs(cur_x - prev_x) #Change in x
    iters = iters+1 #iteration count
    print("Iteration",iters,"\nX value is",cur_x) #Print iterations
    rec_pct_gd.append(100- 100*cur_x)
plt.plot(rec_pct_gd)
plt.show()

print("The local minimum occurs at", cur_x)

plt.plot(rec_pct_gd)
plt.show()


