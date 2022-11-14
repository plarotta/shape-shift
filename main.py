import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from vpython import *
import secrets
import random

class Mass:

    def __init__(self,mass, position=[], velocity=[], acceleration=[], f_ext=[]):
        self.mass = mass #float in kg
        self.position = position #list in m [x,y,z]
        self.velocity = velocity #list in m/s [dx/dt, dy/dt, dz/dt]
        self.acceleration = acceleration #list in m/s^2 [d^2x/dt^2, d^2y/dt^2, d^2z/dt^2]
        self.f_ext = f_ext #list of Ns [[x,y,z],[x,y,z]]

class Spring:
    
    def __init__(self, l, k, idcs):
        
        
        self.mass_indices = idcs
        self.constants = {"a": l,"b": 0.3, "c":0.4, "k": k}
        self.rest_length = l
        self.spring_constant = self.constants["k"]
    
class Simulation:

    def __init__(self, dt, T, w_masses, k_springs):
        self.increment = dt
        self.final_T = T
        self.mass_w = w_masses
        self.spring_k = k_springs
        self.masses = self.initialize_masses()
        self.springs = self.initialize_springs()
        self.mu_static = 0.67
        self.mu_kinetic = 0.47
    
    def initialize_masses(self):
        masses = []
        for x in [0,1]:
            for y in [0,1]:
                for z in [0,1]:
                    masses.append(Mass(self.mass_w, np.array([x,y,z]), np.array([0,0,0]), np.array([0,0,0]), []))
        
        # for theta in np.arange(0,2*pi, pi/4):
        #     for z in [0,5]:
        #         masses.append(Mass(self.mass_w, np.array([1+np.cos(theta),z,1+np.sin(theta)]), np.array([0,0,0]), np.array([0,0,0]), []))

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
 
    def interact(self,t,floor=-4,breath=False):
        for s in self.springs:
            l = self.get_spring_l(s)
            l_vect = np.array( self.masses[s.mass_indices[1]].position -  self.masses[s.mass_indices[0]].position  )
            L0 = s.constants["a"] + s.constants["b"] * np.sin(10*t + s.constants["c"])
            f_k = s.spring_constant * (l - L0)
            f_kx = f_k * l_vect[0]/np.sqrt(l_vect[0]**2 + l_vect[1]**2 + l_vect[2]**2)
            f_ky = f_k * l_vect[1]/np.sqrt(l_vect[0]**2 + l_vect[1]**2 + l_vect[2]**2)
            f_kz = f_k * l_vect[2]/np.sqrt(l_vect[0]**2 + l_vect[1]**2 + l_vect[2]**2)
            self.masses[s.mass_indices[0]].f_ext.append( np.array([f_kx, f_ky, f_kz]) )
            self.masses[s.mass_indices[1]].f_ext.append( np.array([-f_kx, -f_ky, -f_kz]) )

        for m in self.masses:
            
            
            m.f_ext.append(np.array([0,-9.81,0])) #gravity

            f_netx = sum([force[0] for force in m.f_ext])
            f_nety = sum([force[1] for force in m.f_ext])
            f_netz = sum([force[2] for force in m.f_ext])

            if m.position[1] <= floor:
                Fp = np.array([f_netx, 0, f_netz])
                Fp_norm = np.linalg.norm(Fp)
                Fn = np.array([0, f_nety, 0])
                Fn_norm = np.linalg.norm(Fn)
                if Fp_norm < Fn_norm * self.mu_static:
                    f_netx += -f_netx
                    f_netz += -f_netz
                else:
                    dirFn = self.mu_kinetic*Fn_norm*np.array([f_netx, 0, f_netz])/Fp_norm
                    f_netx = sum([force[0] for force in m.f_ext])
                    f_netz = sum([force[2] for force in m.f_ext])
                    
            m.f_ext = [np.array([f_netx, f_nety, f_netz])]
            self.integrate(m,floor)

    
    def integrate(self, mass,floor):
        mass.acceleration = mass.f_ext[0]/mass.mass 
        mass.velocity = mass.velocity + mass.acceleration*self.increment
        if mass.position[1] >=floor and (mass.position[1] + 0.999*mass.velocity[1]*self.increment < floor):
            mass.velocity[1] = -mass.velocity[1]*.9
        mass.velocity = mass.velocity * 0.999
        mass.position = mass.position + mass.velocity*self.increment
        mass.f_ext = []


    def plot_energies(self, es):
        plt.plot(range(len(es["springs"])), es["springs"], 'r',label="Energy from the springs")
        plt.plot(range(len(es["masses"])), es["masses"],'g',label="Energy from the masses")
        plt.plot(range(len(es["masses"])), [es["springs"][i] + es["masses"][i] for i in range(len(es["springs"]))],'b',label="Total energy")
        plt.xlabel("time step")
        plt.ylabel("J")
        plt.title("Energy curve at 0.999 damping")
        plt.show()

    def calc_energy(self):
        spring_e = sum([abs(0.5*spring.spring_constant * (self.get_spring_l(spring)-spring.rest_length)**2) for spring in self.springs])
        mass_e = sum([0.5 * mass.mass * mass.velocity**2 for mass in self.masses]) 
        uM = []
        for mass in self.masses:
            if mass.position[1] > -4:
                uM.append(mass.position[1] + 4)
        mass_e += sum([abs(i*9.81*0.1) for i in uM])
        return(spring_e, mass_e)
       

    def get_spring_l(self, spring):
        m1_idx1 = spring.mass_indices[0]
        m2_idx2 = spring.mass_indices[1]
        m1_p = self.masses[m1_idx1].position
        m2_p = self.masses[m2_idx2].position
        length = np.linalg.norm(m2_p-m1_p)
        return(length)
            
    def initialize_scene(self,z, breath=False):
        # walls
        if not breath:
            floor = box(pos=vector(0.5,z-0.06,0), height=0.02, length=7, width=7,color=color.white)

        m_plots = []
        for mass in self.masses:
            m_temp = sphere(pos=vector(mass.position[0],mass.position[1],mass.position[2]), radius=0.06, color=color.green)
            m_plots.append(m_temp)

        s_plots = []
        for spring in self.springs:
            m1_pos = self.masses[spring.mass_indices[0]].position
            m2_pos = self.masses[spring.mass_indices[1]].position
            s_temp = cylinder(pos=vector(m1_pos[0], m1_pos[1], m1_pos[2]), axis=vector(m2_pos[0]-m1_pos[0], m2_pos[1]-m1_pos[1], m2_pos[2]-m1_pos[2]), length=self.get_spring_l(spring),color=color.red, radius=0.01)
            s_plots.append(s_temp)
        COM_pos = self.get_COM()
        COM = sphere(pos=vector(COM_pos[0],COM_pos[1],COM_pos[2]), radius=0.06, color=color.yellow)

        return(m_plots,s_plots,COM)

    def update_scene(self, m_plots, s_plots, COM):
        for idx,m in enumerate(m_plots):
            m.pos = vector(self.masses[idx].position[0], self.masses[idx].position[1], self.masses[idx].position[2])
        
        for idx,s in enumerate(s_plots):
            m1_pos = self.masses[self.springs[idx].mass_indices[0]].position
            m2_pos = self.masses[self.springs[idx].mass_indices[1]].position
            s.pos = vector(m1_pos[0], m1_pos[1], m1_pos[2])
            s.axis = vector(m2_pos[0]-m1_pos[0], m2_pos[1]-m1_pos[1], m2_pos[2]-m1_pos[2])
            s.length=self.get_spring_l(self.springs[idx])
        COM_pos = self.get_COM()
        COM.pos = vector(COM_pos[0],COM_pos[1],COM_pos[2])

    def run_simulation(self,plot_energies=False):
        self.masses[0].f_ext.append(np.array([3000,3000,0]))
        m_obs, s_obs, COM = self.initialize_scene(z=-4.06)
        time.sleep(10)
        
        times = np.arange(0,self.final_T, 0.02)
        if plot_energies:
            energies = {"springs": [], "masses": []}
        for t in np.arange(0, self.final_T, self.increment):
            if plot_energies:
                spre, mase = self.calc_energy()
                energies["springs"].append(spre)
                energies["masses"].append(mase)
            self.interact(t,floor=-4)
            if t in times:
                rate(50)
                self.update_scene(m_obs, s_obs, COM)
        if plot_energies:
            self.plot_energies(energies)
    
    def get_COM(self):
        M = sum([m.mass for m in self.masses])
        COMx = sum([m.mass*m.position[0] for m in self.masses])/M
        COMy = sum([m.mass*m.position[1] for m in self.masses])/M
        COMz = sum([m.mass*m.position[2] for m in self.masses])/M
        return(np.array([COMx, COMy, COMz]))

    def eval_springs(self, springs):
        for idx in range(len(self.springs)):
            self.springs[idx].constants["b"] = springs[idx][0]
            self.springs[idx].constants["c"] = springs[idx][1]
            self.springs[idx].constants["k"] = springs[idx][2]
        p1 = self.get_COM()
        for t in np.arange(0, self.final_T, self.increment):
            self.interact(t,floor=-4)
        p2 = self.get_COM()
        return(np.linalg.norm([p2[0],0,p2[2]]-[p1[0],0,p1[2]]))









def evolve_robot(n_gen):

    #initialize population [done]
    #loop
        #select
        #breed [done]
        #mutate [done]
    #return best indiv

    '''
    Encoding: 
    [
        [b1,c1,k1], spring 1 params
        [b2,c2,k2], spring 2 params
        [b3,c3,k3], spring 3 params
        [b4,c4,k4] spring 4 params
    ] 

    Mutation:
    -choose random spring
    -change the value of each of its params by some delta. different delta for each param

    Selection:
    -rank-based

    Crossover:
    *two-point crossover*
    -choose n1 and n2
    -child1: for 0 to n1 parent1.springs[0:n1]=parent2.springs[0:n1], 
             for n1 to n2 pass
             for n2 to end parent1.springs[n1:n2]=parent2.springs[n1:n2]
    -child2: complement of child2
    
    '''
    pass


class Evolution():
    def __init__(self, springs):
        self.generations = 10
        self.population_size = 10
        self.mutation_rate = 1
        self.crossover_rate = 0.75
        self.springs = springs
    
    def initialize_population(self):
        b_uppers = [spring.constants["a"] for spring in self.springs]
        k_upper = self.springs[0].constants["k"]*100
        num_springs = len(self.springs)
        pool = []
        for i in range(self.population_size):
            individual = []
            for s in range(num_springs):
                # random.random() *(upper - lower) + lower
                b = random.random() *(b_uppers[s] - 0) + 0
                c = random.random() *(2*pi - 0) + 0
                k = random.random() *(k_upper - 100) + 100
                individual.append([b,c,k])
            pool.append(individual)
        return(pool)

    def mutate_individual(self, individual):
        if random.choices([True, False], weights=[self.mutation_rate, 1-self.mutation_rate], k=1)[0]:
            
            chosen_s_idx = secrets.choice(range(len(individual)))
            newb = individual[chosen_s_idx][0]+random.random() *(individual[chosen_s_idx][0]*0.2 + individual[chosen_s_idx][0]*0.2) - individual[chosen_s_idx][0]*0.2
            newc = individual[chosen_s_idx][1]+random.random() *(individual[chosen_s_idx][1]*0.2 + individual[chosen_s_idx][1]*0.2) - individual[chosen_s_idx][1]*0.2
            newk = individual[chosen_s_idx][2]+random.random() *(individual[chosen_s_idx][2]*0.2 + individual[chosen_s_idx][2]*0.2) - individual[chosen_s_idx][2]*0.2
            individual[chosen_s_idx] = [newb, newc, newk]
    
    def breed(self, parents):
        n1 = secrets.choice(range(len(self.springs)))
        n2 = secrets.choice(range(len(self.springs)))
        if n1 == n2:
            n2 = secrets.choice(range(len(self.springs)))
        n1, n2 = sorted([n1,n2])
        child1 = parents[0][0:n1] + parents[1][n1:n2] + parents[0][n2:]
        child2 = parents[1][0:n1] + parents[0][n1:n2] + parents[1][n2:]
        return(child1, child2)
    


            











if __name__ == "__main__":
    a = Simulation(dt=0.0002,T=5, w_masses=0.1, k_springs=10000)
    springs = a.springs
    b = Evolution(springs)
    indiv = b.initialize_population()[0]
    print(a.eval_springs(indiv))
    b.mutate_individual(indiv)
    print(a.eval_springs(indiv))


    




