import numpy as np
import matplotlib.pyplot as plt

class Mass:
    
    def __init__(self,mass, position=[], velocity=[], acceleration=[], f_ext=[]):
        self.mass = mass #float in kg
        self.position = position #list in m [x,y,z]
        self.velocity = velocity #list in m/s [dx/dt, dy/dt, dz/dt]
        self.acceleration = acceleration #list in m/s^2 [d^2x/dt^2, d^2y/dt^2, d^2z/dt^2]
        self.f_ext = f_ext #list in N [Fx, Fy, Fz]

class Spring:
    
    def __init__(self, l, k, idcs):
        self.rest_length = l
        self.spring_constant = k
        self.mass_indices = idcs
    
class Simulation:

    def __init__(self, dt, T, n_masses, w_masses, n_springs, k_springs):
        self.increment = dt
        self.final_T = T
        self.mass_n = n_masses
        self.mass_w = w_masses
        self.spring_n = n_springs
        self.spring_k = k_springs
        self.masses = self.initialize_masses()
        self.springs = self.initialize_springs()
    
    def initialize_masses(self):
        masses = []
        for x in [0,1]:
            for y in [0,1]:
                for z in [0,1]:
                    masses.append(Mass(0.1, np.array([x,y,z]), [0,0,0], [0,0,0], [0,0,0]))
        return(masses)
        

    def initialize_springs(self):
        springs = []
        for mass_idx1 in range(len(self.masses)):
            for mass_idx2 in range(mass_idx1+1,len(self.masses)):
                m1_p = self.masses[mass_idx1].position
                m2_p = self.masses[mass_idx2].position
                length = dist = np.linalg.norm(m2_p-m1_p)
                springs.append(Spring(length, 10000, [mass_idx1, mass_idx2]))
        return(springs)

    def plot_masses(self):
        # plots masses and springs
        fig = plt.figure()
        ax = plt.axes(projection ='3d')
        for mass in self.masses:
            ax.plot3D(mass.position[0], mass.position[1], mass.position[2], "ro")
        
        for spring in self.springs:
            xs = [self.masses[i].position[0] for i in spring.mass_indices]
            ys = [self.masses[i].position[1] for i in spring.mass_indices]
            zs = [self.masses[i].position[2] for i in spring.mass_indices]
            ax.plot3D(xs, ys, zs, "green")
        plt.show()


    def interact(self):
        pass

    def integrate(self):
        pass

if __name__ == "__main__":
    a = Simulation(0.1,5,4,4,5,6)
    a.plot_masses()
    # masses = a.masses
    # springs = a.springs
    # print("obj",masses)
    # print([i.position for i in masses])
    # print("obj", springs)
    # print([(i.rest_length, i.mass_indices) for i in springs])
    # fig = plt.figure()
    # ax = plt.axes(projection ='3d')
    
    # ax.plot3D([1,2,3],[1,2,3],[1,2,3], "ro")
    # plt.show()
