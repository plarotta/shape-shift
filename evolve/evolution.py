import sys
import os
sys.path.append(os.getcwd())
from simulate.simulation import Simulation
import numpy as np
import secrets
import random
import copy

class Evolution(Simulation):
    def __init__(self, masses=None, springs=None):
        super().__init__(masses, springs)
        # population = self.initialize_population()

    def initialize_population(self, population_size, springs):
        pool = []
        for i in range(population_size):
            pool.append(self.spawn_individual(springs))
        return(pool)
    
    def rehome(self, masses):
        for m in masses:
            pos0 = m[4]
            m[0] = pos0
            m[1] = np.array([0.0, 0.0, 0.0])
            m[2] = np.array([0.0, 0.0, 0.0])
            m[3] = np.array([0.0, 0.0, 0.0])
    
    def eval_springs(self, masses,springs,spring_constants,final_T=4,increment=0.001):
        for idx in range(len(springs)):
            springs[idx][1][1] = spring_constants[idx][0]
            springs[idx][1][2] = spring_constants[idx][1]
            springs[idx][1][3] = spring_constants[idx][2]

        p1 = self.get_COM(masses)
        for t in np.arange(0, final_T, increment):
            self.interact(springs,masses,t,increment,0.9,0.8,floor=0)
        p2 = self.get_COM(masses)
        self.rehome(masses) #reset current state of the masses
        return(np.linalg.norm(np.array([p2[0],0,p2[2]])-np.array([p1[0],0,p1[2]])))

    def mutate_morphology(self, masses,springs,spring_constants):
        operation = np.random.choice(["Fatten", "Slim"],p=[0.75,0.25])
        if len(masses) < 10 or operation == "Fatten":
            masses2,springs2,spring_constants2 = self.fatten_cube(masses,springs,spring_constants)
        else:
            masses2,springs2,spring_constants2 = self.slim_cube(masses,springs,spring_constants) #this seems to work fine
        
        return(masses2,springs2,spring_constants2)
    
    def slim_cube(self, masses,springs,spring_constants):
        chosen_mass_idx = secrets.choice(range(len(masses)))
        masses2 = np.delete(masses,chosen_mass_idx,axis=0)
        springs_idxs = springs[:,0,0:2]
        springs_to_remove = np.argwhere(springs_idxs == float(chosen_mass_idx))[:,0]
        springs2 = np.delete(springs, springs_to_remove,axis=0)
        spring_constants2 = np.delete(spring_constants, springs_to_remove,axis=0)
        springs2[:,0,0:2] = np.where(springs2[:,0,0:2] > float(chosen_mass_idx),springs2[:,0,0:2] -1,springs2[:,0,0:2])
        return(masses2,springs2,spring_constants2)
    
    def fatten_cube(self, masses,springs,spring_constants):
        invalid_point = True
        while invalid_point:
            chosen_masses = np.array(random.choices(masses,k=3))
            ch_m_pos = chosen_masses[:,0,:] 
            v1 = ch_m_pos[0]-ch_m_pos[1]
            v2 = ch_m_pos[2]-ch_m_pos[1]
            normal_vect = np.cross(v1,v2)
            range_x = np.array([np.mean(ch_m_pos[:,0])-random.random(), np.mean(ch_m_pos[:,0])*1.8+random.random()])
            range_y = np.array([np.mean(ch_m_pos[:,1])-random.random(), np.mean(ch_m_pos[:,1])*1.8+random.random()])
            range_z = np.array([np.mean(ch_m_pos[:,2])-random.random(), np.mean(ch_m_pos[:,2])*1.8+random.random()])
            point = np.array([ random.random() *(range_x[1] - range_x[0]) + range_x[0] , random.random() *(range_y[1] - range_y[0]) + range_y[0] , random.random() *(range_z[1] - range_z[0]) + range_z[0] ])
            if point[1] < 0:
                continue
            point = point + random.random()*normal_vect
            if point[1] < 0:
                continue
            invalid_point = False
            masses2,springs2,spring_constants2 = self.append_point(point,masses,springs,spring_constants)
        return(masses2,springs2,spring_constants2)
    
    def append_point(self, point,masses,springs,spring_constants):
        mass = np.array([point,np.array([0.0,0.0,0.0]),np.array([0.0,0.0,0.0]),np.array([0.0,0.0,0.0]),point])
        masses = np.append(masses,mass).reshape(len(masses)+1,5,3)
        for mass_idx1 in range(len(masses)-1):
            m1_p = masses[mass_idx1][0] #get pos
            length = np.linalg.norm(m1_p-point)
            spring = np.array([[[mass_idx1, len(masses)-1, 0, 0],[length, 0,0,5000]]])
            springs = np.concatenate((springs,spring))
            spring_constants = np.concatenate((spring_constants, np.array([self.spawn_spring()])))
        return(masses, springs,spring_constants)
    
    def spawn_individual(self, springs):
        individual = []
        num_springs = len(springs)
        individual = np.zeros(3*num_springs).reshape(num_springs,3)
        for s in range(num_springs):
                b = random.random() *(1 - 0) + 0
                c = random.random() *(2*np.pi - 0) + 0
                k = random.random() *(10000 - 0) + 0
                individual[s] = np.array([b,c,k])
        return(individual)

    def spawn_spring(self):
        b = random.random() *(1 - 0) + 0
        c = random.random() *(2*np.pi - 0) + 0
        k = random.random() *(10000 - 0) + 0
        individual = np.array([b,c,k])
        return(individual)
    
    def mutate_individual(self,mutation_rate, individual):
        if random.choices([True, False], weights=[mutation_rate, 1-mutation_rate], k=1)[0]:
            indiv = copy.deepcopy(individual)
            chosen_s_idx = secrets.choice(range(len(indiv)))
            newb = indiv[chosen_s_idx][0]+random.random() *(indiv[chosen_s_idx][0]*0.35 + indiv[chosen_s_idx][0]*0.35) - indiv[chosen_s_idx][0]*0.35
            newc = indiv[chosen_s_idx][1]+random.random() *(indiv[chosen_s_idx][1]*0.35 + indiv[chosen_s_idx][1]*0.35) - indiv[chosen_s_idx][1]*0.35
            newk = indiv[chosen_s_idx][2]+random.random() *(indiv[chosen_s_idx][2]*0.35 + indiv[chosen_s_idx][2]*0.35) - indiv[chosen_s_idx][2]*0.35
            indiv[chosen_s_idx] = (newb, newc, newk)
            return(indiv)
        else:
            return(individual)
        
    def ranked_selection(self, population_pool,p_c):
        probabilities = np.array([((1-p_c)**((i+1)-1))*p_c for i in range(len(population_pool)-1)] + [(1-p_c)**(len(population_pool))])
        probabilities /= probabilities.sum()
        indices = list(range(len(population_pool)))
        indices = np.random.choice(indices,size=2, p=probabilities, replace=False)
        chosen_ones = [population_pool[c] for c in indices]
        return(chosen_ones,indices)

    def breed_v3(self, parents,parent_indices, pool_masses, pool_springs):
        p1,p2 = parents
        n1 = secrets.choice(range(1,len(p1)))
        A = p1[:n1]
        subset_max_size = len(p1)-len(A) if len(p1) - len(A) < len(p2) else len(p2)
        subset_size = 1 if subset_max_size == 1 else secrets.choice(range(1,subset_max_size))
        n2 = secrets.choice([i for i in range(0,len(p2)+1-subset_size)])
        B = p2[n2:n2+subset_size]
        C = p1[-(len(p1)-len(np.concatenate((A,B)))):] if len(np.concatenate((A,B))) < len(p1) else []
        child1 = np.concatenate((A,B,C)) if C != [] else np.concatenate((A,B))
        
        ######
            
        p2,p1 = parents
        n1 = secrets.choice(range(1,len(p1)))
        A = p1[:n1]
        subset_max_size = len(p1)-len(A) if len(p1) - len(A) < len(p2) else len(p2)
        subset_size = 1 if subset_max_size == 1 else secrets.choice(range(1,subset_max_size))
        n2 = secrets.choice([i for i in range(0,len(p2)+1-subset_size)])
        B = p2[n2:n2+subset_size]
        C = p1[-(len(p1)-len(np.concatenate((A,B)))):] if len(np.concatenate((A,B))) < len(p1) else []

        child2 = np.concatenate((A,B,C)) if C != [] else np.concatenate((A,B))
        return(child1,child2)