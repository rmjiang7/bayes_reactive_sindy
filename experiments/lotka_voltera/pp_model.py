import numpy as np
from scipy import integrate    
import matplotlib.pyplot as plt
from rsindy.utils import *

# True stoichiometric matrix
Strue = [[1, 0, 0],
         [-1, 1, 0],
         [0, -1, 0]]
Strue = np.array(Strue)

# True reaction rate matrix
Rtrue = [[1, 0, 0],
         [1, 1, 0],
         [0, 1, 0]]
Rtrue = np.array(Rtrue)

# True parameters
theta = [1.0, 0.01, 0.3]

# Species name
species_names = ['X', 'Y']

def simulate_data(theta, t):
    
    Z0 = [100, 50]
    k1, k2, k3 = theta
    def dZdt(Z, t = 0):
        X, Y = Z

        dXdt = k1 * X - k2 * X * Y
        dYdt = k2 * X * Y - k3 * Y

        return [dXdt, dYdt]
    
    Z_obs = integrate.odeint(dZdt, Z0, t)
    Z_obs_noisy = Z_obs.copy()
    Z_obs_noisy[1:,:] = np.exp(np.log(Z_obs_noisy[1:,:]) + np.random.normal(0, 0.2, size = Z_obs_noisy[1:,:].shape))
    
    return Z_obs, Z_obs_noisy

def custom_reaction_basis(species):
    
    species_to_idx_map = {s: i for i, s in enumerate(species)}
    stoichiometry = []
    rates = []
    reaction_description = []
    
    stoichiometry.append(np.array([-2, 0, 0]))
    rates.append(np.array([2, 0, 0]))
    reaction_description.append("2X -> 0")
    
    stoichiometry.append(np.array([0, -2, 0]))
    rates.append(np.array([0, 2, 0]))
    reaction_description.append("2Y -> 0")
    
    stoichiometry.append(np.array([1, 0, 0]))
    rates.append(np.array([1, 0, 0]))
    reaction_description.append("X -> 2X")
    
    stoichiometry.append(np.array([-1, 1, 0]))
    rates.append(np.array([1, 1, 0]))
    reaction_description.append("X + Y -> 2Y")
    
    stoichiometry.append(np.array([0, -1, 0]))
    rates.append(np.array([0, 1, 0]))
    reaction_description.append("Y -> 0")
    
    stoichiometry.append(np.array([1, -1, 0]))
    rates.append(np.array([1, 1, 0]))
    reaction_description.append("X + Y -> 2X")
    
    stoichiometry.append(np.array([-1, 0, 0]))
    rates.append(np.array([1, 0, 0]))
    reaction_description.append("X -> 0")
    
    stoichiometry.append(np.array([0, -1, 0]))
    rates.append(np.array([0, 2, 0]))
    reaction_description.append("2Y -> Y")
    
    stoichiometry.append(np.array([0, 1, 0]))
    rates.append(np.array([0, 1, 0]))
    reaction_description.append("Y -> 2Y")
    
    stoichiometry.append(np.array([-1, 0, 0]))
    rates.append(np.array([2, 0, 0]))
    reaction_description.append("2X -> X")
    
    stoichiometry.append(np.array([0, -1, 0]))
    rates.append(np.array([1, 1, 0]))
    reaction_description.append("X + Y -> X")
    
    stoichiometry.append(np.array([-1, 0, 0]))
    rates.append(np.array([1, 1, 0]))
    reaction_description.append("X + Y -> Y")
    
    stoichiometry.append(np.array([-2, 1, 0]))
    rates.append(np.array([2, 0, 0]))
    reaction_description.append("2X -> Y")
    
    stoichiometry.append(np.array([-1, 1, 0]))
    rates.append(np.array([1, 0, 0]))
    reaction_description.append("X -> Y")
    
    stoichiometry.append(np.array([1, -1, 0]))
    rates.append(np.array([0, 1, 0]))
    reaction_description.append("Y -> X")
    
    stoichiometry.append(np.array([-1, 2, 0]))
    rates.append(np.array([1, 0, 0]))
    reaction_description.append("X -> 2Y")
    
    reaction_mappings = {}
    for i in range(len(stoichiometry)):
        reaction_mappings[encode(stoichiometry[i], rates[i])] = i
    return np.vstack(stoichiometry), np.vstack(rates), reaction_description, reaction_mappings