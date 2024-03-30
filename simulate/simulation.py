import numpy as np
from fast_utils import *
from tqdm import tqdm
import pickle
import os
from datetime import datetime


class Simulation():
    def __init__(self, masses=None, springs=None):
        self.masses = self.initialize_masses() if masses is None else masses
        self.springs = self.initialize_springs(self.masses) if springs is None else springs
        self.sim_log = {}

    def initialize_masses(self):
        masses = np.zeros((8,5,3))
        n = 0
        for x in [0,1]:
            for y in [0,1]:
                for z in [0,1]:
                    masses[n][0] = np.array([float(x),float(y),float(z)])
                    masses[n][4] = np.array([float(x),float(y),float(z)])
                    n+=1
        return(masses)

    def initialize_springs(self, masses):
        first = True
        for mass_idx1 in range(len(masses)):
            for mass_idx2 in range(mass_idx1+1,len(masses)):
                m1_p = masses[mass_idx1][0] #get pos
                m2_p = masses[mass_idx2][0] #get pos
                length = np.linalg.norm(m2_p-m1_p)
                if first == True:
                    springs = np.array([[[mass_idx1, mass_idx2, 0, 0],[length, 0,0,5000]]])
                    first = False
                else:
                    springs = np.concatenate((springs, np.array([[[mass_idx1, mass_idx2, 0, 0],[length, 0,0,5000]]])))
        return(springs)

    def get_COM(self, masses):
        return(get_COM_fast(masses))
    
    def get_spring_l(self, masses, springs):
        return(get_spring_l_fast(masses, springs))
    
    def integrate(self, m):
        integrate_fast(m, self.time_step)
    
    def interact(self, springs, masses, t):
        interact_fast(springs, masses, t, self.time_step, self.mu_s, self. mu_k, self.floor_z_pos)

    def run_simulation(self, sim_length, time_step, log_k, mu_s, mu_k, floor_z_position, save=False):   
        for t in tqdm(np.arange(0, sim_length, time_step)):
            if t % log_k == 0:
                p1 = self.get_COM(self.masses)
                interact_fast(self.springs, self.masses, t, time_step, mu_s, mu_k, floor_z_position)
                p2 = self.get_COM(self.masses)
                displacement = np.linalg.norm(np.array([p2[0],0,p2[2]])-np.array([p1[0],0,p1[2]]))
                speed = displacement/time_step
                self.sim_log[t] = {
                    'masses': np.copy(self.masses),
                    'springs': np.copy(self.springs),
                    'speed': speed
                }
            else:
                interact_fast(self.springs, self.masses, t, time_step, mu_s, mu_k, floor_z_position)
        if save is True:
            path = os.getcwd()+'/simulate/saved_logs'
            now = datetime.now()
            now = f'{now.month}-{now.day}-{now.year}-{now.hour}-{now.minute}-{now.second}.pkl'

            with open(os.path.join(path, now), 'wb') as handle:
                pickle.dump(self.sim_log, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    simy = Simulation()
    simy.run_simulation(300, .001, 10, 0.9, 0.8, -4, save=True)
        