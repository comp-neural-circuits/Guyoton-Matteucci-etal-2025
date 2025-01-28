'''
Parameters for base model
2025 m.getz
'''
import numpy as np
import scipy.ndimage as image
from scipy.stats import norm

## MODEL SIZE
# equivalent to an iso-tuned population; i.e. 2xdim per region
dim = 50 

# set equal to 3*2*dim1 x 3*2*dim2
# (4 pops)
dt = 0.01 #msec
T = 1000 #msec
burn = 5000 #time steps
ntrials = 4 #num. trials

##INPUTS
extstim = 'S-const' #or 'V-const'
c1 = 20 #input magnitude
c1e = c1 
c1i = c1
b = 0 #constant baseline input
bi = 0
switch = False

## READOUT PARAMS
th0 = 80 #drift-diffusion detection threshold
sigd = 0.08 #noise magnitude on output (integrator) units

#DYNAMICS
split = False
m = 1 #power of transfer function
me = 2 #if different for E/I
mi = 2
taue = 20 #msec
taui = 10 #msec
taun = 1  #msec 

#CONNECTIVITY
sigw = 0.1 #width of initial readout weights

pie = 1 #0.4
pee = 1 #0.4
pei = 1 #0.4
pii = 1 #0.2

## WEIGHT UPDATE PARAMS
wmax = 1
gam1 = 0.0000004 #learning rate

##NOISE VARS
c_ne = 0.75 #noise amplitude, E pop.
c_ni = 0 #noise amplitude, I pop.

##NETWORK PROPERTIES
Jee = 0.05
Jei = 0.065
Jie = 0.055
Jii = 0.045

Jz = 0.5 #sets mean output weights Wz

#for freezing random weight matrices:
Wie = np.matrix(np.random.rand(dim, dim)<pie)
Wee = np.matrix(np.random.rand(dim, dim)<pee)
Wei = np.matrix(np.random.rand(dim, dim)<pei)
Wii = np.matrix(np.random.rand(dim, dim)<pii)
