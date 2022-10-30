import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Mass:

    def __init__(self,mass, position=[], velocity=[], acceleration=[], f_ext=[]):
        self.mass = mass #float in kg
        self.position = position #list in m [x,y,z]
        self.velocity = velocity #list in m/s [dx/dt, dy/dt, dz/dt]
        self.acceleration = acceleration #list in m/s^2 [d^2x/dt^2, d^2y/dt^2, d^2z/dt^2]
        self.f_ext = f_ext #list of Ns [[x,y,z],[x,y,z]]

class Spring:
    
    def __init__(self, l, k, idcs):
        self.rest_length = l
        self.spring_constant = k
        self.mass_indices = idcs
    
class Simulation:

    def __init__(self, dt, T, w_masses, k_springs):
        self.increment = dt
        self.final_T = T
        self.mass_w = w_masses
        self.spring_k = k_springs
        self.masses = self.initialize_masses()
        self.springs = self.initialize_springs()
    
    def initialize_masses(self):
        masses = []
        for x in [0,1]:
            for y in [0,1]:
                for z in [0,1]:
                    masses.append(Mass(self.mass_w, np.array([x,y,z]), np.array([0,0,0]), np.array([0,0,0]), []))
        return(masses)
        
    def initialize_springs(self):
        springs = []
        for mass_idx1 in range(len(self.masses)):
            for mass_idx2 in range(mass_idx1+1,len(self.masses)):
                m1_p = self.masses[mass_idx1].position
                m2_p = self.masses[mass_idx2].position
                length = np.linalg.norm(m2_p-m1_p)
                springs.append(Spring(length, self.spring_k, [mass_idx1, mass_idx2]))
        return(springs)

    def plot_cube(self):
        ax = plt.axes(projection ='3d')
        
        #ax = fig.add_subplot(111,projection='3d')

        xx, yy = np.meshgrid(np.linspace(-2,1.3,100), np.linspace(-2,1.3,100))
        z = (xx-xx) +(yy-yy) -2


        # plot the plane
        ax.plot_surface(xx, yy, z, alpha=0.5, color = "blue")

        #plt.show()
        for mass in self.masses:
            ax.plot3D(mass.position[0], mass.position[1], mass.position[2], "ro")
            
        for spring in self.springs:
            xs = [self.masses[i].position[0] for i in spring.mass_indices]
            ys = [self.masses[i].position[1] for i in spring.mass_indices]
            zs = [self.masses[i].position[2] for i in spring.mass_indices]
            ax.plot3D(xs, ys, zs, "green")
            plt.xlim((-2,1.3))
            plt.xlabel("X")
            plt.ylim((-2,1.3))
            plt.ylabel("Y")
            ax.set_zlim(-2,1.3)
        plt.pause(.1)
 
    def interact(self):
        for s in self.springs:
            l = self.get_spring_l(s)
            l_vect = np.array( self.masses[s.mass_indices[1]].position -  self.masses[s.mass_indices[0]].position  )
            f_k = s.spring_constant * (l - s.rest_length)
            f_kx = f_k * l_vect[0]/np.sqrt(l_vect[0]**2 + l_vect[1]**2 + l_vect[2]**2)
            f_ky = f_k * l_vect[1]/np.sqrt(l_vect[0]**2 + l_vect[1]**2 + l_vect[2]**2)
            f_kz = f_k * l_vect[2]/np.sqrt(l_vect[0]**2 + l_vect[1]**2 + l_vect[2]**2)
            self.masses[s.mass_indices[0]].f_ext.append( np.array([f_kx, f_ky, f_kz]) )


            l_vect = np.array( self.masses[s.mass_indices[0]].position -  self.masses[s.mass_indices[1]].position  )
            f_kx = f_k * l_vect[0]/np.sqrt(l_vect[0]**2 + l_vect[1]**2 + l_vect[2]**2)
            f_ky = f_k * l_vect[1]/np.sqrt(l_vect[0]**2 + l_vect[1]**2 + l_vect[2]**2)
            f_kz = f_k * l_vect[2]/np.sqrt(l_vect[0]**2 + l_vect[1]**2 + l_vect[2]**2)
            self.masses[s.mass_indices[1]].f_ext.append( np.array([f_kx, f_ky, f_kz]) )

        for m in self.masses:
            if m.position[2] < -2:
                m.f_ext.append(np.array([0,0,-100000*(m.position[2]+2)])) #hit the floor
            m.f_ext.append(np.array([0,0,-9.81])) #gravity

            f_netx = sum([force[0] for force in m.f_ext])
            f_nety = sum([force[1] for force in m.f_ext])
            f_netz = sum([force[2] for force in m.f_ext])
            m.f_ext = [np.array([f_netx, f_nety, f_netz])]
            self.integrate(m)

    
    def integrate(self, mass):
        mass.acceleration = mass.f_ext[0]/mass.mass 
        mass.velocity = mass.velocity + mass.acceleration*self.increment
        mass.velocity = mass.velocity * 0.999
        mass.position = mass.position + mass.velocity*self.increment
        mass.f_ext = []

    def run_simulation(self):
        self.masses[0].f_ext.append(np.array([0,100,0]))
        times = np.arange(0,self.final_T, 0.03)
        for t in np.arange(0,self.final_T, self.increment):
            self.interact()
            if t in times:
                self.plot_cube()
            
        plt.show()

    def get_spring_l(self, spring):
        m1_idx1 = spring.mass_indices[0]
        m2_idx2 = spring.mass_indices[1]
        m1_p = self.masses[m1_idx1].position
        m2_p = self.masses[m2_idx2].position
        length = np.linalg.norm(m2_p-m1_p)
        return(length)
            

if __name__ == "__main__":
    a = Simulation(dt=0.001,T=5, w_masses=0.1, k_springs=10000)
    a.run_simulation()
    #a.plot_cube()
    #plt.show()
