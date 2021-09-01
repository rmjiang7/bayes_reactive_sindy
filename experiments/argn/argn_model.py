import numpy as np
from scipy import integrate    
import matplotlib.pyplot as plt
from rsindy.utils import *

# True stoichiometric matrix
Strue = [[-1, -1,  1,  0,  0,  0],
         [ 1,  1, -1,  0,  0,  0],
         [ 0,  0,  0,  1,  0,  0],
         [ 0,  0,  0,  0,  1,  0],
         [ 0,  1,  0,  0, -2,  0],
         [ 0, -1,  0,  0,  2,  0],
         [ 0,  0,  0, -1,  0,  0],
         [ 0,  0,  0,  0, -1,  0]
]
Strue = np.array(Strue)

# True reaction rate matrix
Rtrue = [[1,  1,  0,  0,  0,  0],
         [0,  0,  1,  0,  0,  0],
         [1,  0,  0,  0,  0,  0],
         [0,  0,  0,  1,  0,  0],
         [0,  0,  0,  0,  2,  0],
         [0,  1,  0,  0,  0,  0],
         [0,  0,  0,  1,  0,  0],
         [0,  0,  0,  0,  1,  0]
]
Rtrue = np.array(Rtrue)

# True parameters
theta = [0.5 , 1.  , 0.15, 1.  , 0.5 , 0.5 , 1.5 , 0.15]

# Species name
species_names = ['g', 'pt', 'g.pt', 'r', 'p']

def simulate_data(theta, t):
    
    Z0 = [20, 20, 20, 20, 20]
    k1, k2, k3, k4, k5, k6, k7, k8 = theta
    def dZdt(Z, t = 0):
        g, p2, gp2, r, p = Z
        
        dgdt = k2 * gp2 - k1 * g * p2
        dp2dt = k5 * p * p + k2 * gp2 - k1 * g * p2 - k6 * p2
        dgp2dt = k1 * g * p2 - k2 * gp2
        drdt = k3 * g - k7 * r
        dpdt = k4 * r + 2 * k6 * p2 - 2 * k5 * p * p - k8 * p
        
    
        return [dgdt, dp2dt, dgp2dt, drdt, dpdt]
    
    Z_obs = integrate.odeint(dZdt, Z0, t)
    Z_obs_noisy = Z_obs.copy()
    Z_obs_noisy[1:,:] = np.exp(np.log(Z_obs_noisy[1:,:]) + np.random.normal(0, 0.07, size = Z_obs_noisy[1:,:].shape))

    return Z_obs, Z_obs_noisy