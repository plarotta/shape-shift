import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from vpython import *
import secrets
import random
import copy
import pickle

class Mass:

    def __init__(self,mass, position=np.array([0,0,0]), velocity=np.array([0,0,0]), acceleration=np.array([0,0,0]), f_ext=np.array([0,0,0])):
        self.mass = mass #float in kg
        self.position = position #list in m [x,y,z]
        self.velocity = velocity #list in m/s [dx/dt, dy/dt, dz/dt]
        self.acceleration = acceleration #list in m/s^2 [d^2x/dt^2, d^2y/dt^2, d^2z/dt^2]
        self.f_ext = f_ext #list of Ns [[x,y,z],[x,y,z]] # now just a np.array

class Spring:
    
    def __init__(self, l, k, idcs):
        
        
        self.mass_indices = idcs
        self.constants = {"a": l,"b": 0, "c":0, "k": k}
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
        self.mu_static = 0.9
        self.mu_kinetic = 0.7
    
    def initialize_masses(self):
        masses = []
        for x in [0,1]:
            for y in [0,1]:
                for z in [0,1]:
                    masses.append(Mass(self.mass_w, position=np.array([x,y,z])))
        
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
        # start = time.time()
        for s in self.springs:
            l = self.get_spring_l(s)
            l_vect = np.array( self.masses[s.mass_indices[1]].position -  self.masses[s.mass_indices[0]].position  )
            L0 = s.constants["a"] + s.constants["b"] * np.sin(5*t + s.constants["c"])
            f_k = s.spring_constant * (l - L0)
            f_dir = f_k/np.sqrt(l_vect[0]**2 + l_vect[1]**2 + l_vect[2]**2)
            f_kx = f_dir * l_vect[0]
            f_ky = f_dir * l_vect[1]
            f_kz = f_dir * l_vect[2]
            self.masses[s.mass_indices[0]].f_ext = self.masses[s.mass_indices[0]].f_ext + np.array([f_kx, f_ky, f_kz])
            self.masses[s.mass_indices[1]].f_ext = self.masses[s.mass_indices[1]].f_ext + np.array([-f_kx, -f_ky, -f_kz]) 
        # print("spring calcs: ", time.time()-start)
        # print("\n")
        start = time.time()
        for m in self.masses:   
            m.f_ext = m.f_ext + np.array([0,-9.81,0]) #gravity

            # if round(m.position[1],2) <= floor:
            if m.position[1] <= floor:
                
                f_netx=m.f_ext[0]
                f_nety=m.f_ext[1]
                f_netz=m.f_ext[2]
                Fp = np.array([f_netx, 0, f_netz])
                Fp_norm = np.linalg.norm(Fp)
                Fn = np.array([0, f_nety, 0])
                Fn_norm = np.linalg.norm(Fn)
                if Fp_norm < Fn_norm * self.mu_static: #friction
                    # pass
                    #print("A",m.f_ext)
                    m.f_ext = m.f_ext - np.array([f_netx,0,f_netz])
                    #print("B",m.f_ext)
                    

                else:
                    #print([f_netx,f_netz])
                    dirFn = self.mu_kinetic*Fn_norm*np.array([f_netx, 0, f_netz])/Fp_norm #friction
                    # print(dirFn)
                    m.f_ext = m.f_ext - np.array([dirFn[0],0,dirFn[2]])

                # if round(m.position[1],2) < floor: #ground reaction force
                if m.position[1] < floor:
                    # print("now")
                    ry = 10000*(abs(round(m.position[1],3) - floor))
                    m.f_ext = m.f_ext + np.array([0,ry,0])


            self.integrate(m,floor)
        #print("mass calcs: ", time.time()-start)

    
    def integrate(self, mass,floor):
        # mass.f_ext = mass.f_ext.round(2)
        # mass.acceleration = (mass.f_ext/mass.mass).round(4)
        mass.acceleration = mass.f_ext/0.1
        # mass.velocity = (mass.velocity + mass.acceleration*self.increment).round(4)
        mass.velocity = mass.velocity + mass.acceleration*self.increment
        # mass.velocity = (mass.velocity * 0.9985).round(4)
        mass.velocity = mass.velocity * 0.9985
        # print(mass.velocity)
        # mass.position = (mass.position + mass.velocity*self.increment).round(4)
        mass.position = mass.position + mass.velocity*0.0002
        mass.f_ext = np.array([0,0,0])


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
        # m1_idx1 = spring.mass_indices[0]
        # m2_idx2 = spring.mass_indices[1]
        # m1_p = self.masses[spring.mass_indices[0]].position
        # m2_p = self.masses[spring.mass_indices[1]].position
        # length = np.linalg.norm(m2_p-m1_p)
        return(np.linalg.norm(self.masses[spring.mass_indices[1]].position-self.masses[spring.mass_indices[0]].position))
            
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
        # self.masses[0].f_ext = self.masses[0].f_ext +np.array([30000,0,4000])
        m_obs, s_obs, COM = self.initialize_scene(z=-0.06)
        time.sleep(3)
        
        times = np.arange(0,self.final_T, 0.02)
        if plot_energies:
            energies = {"springs": [], "masses": []}
        for t in np.arange(0, self.final_T, self.increment):
            if plot_energies:
                spre, mase = self.calc_energy()
                energies["springs"].append(spre)
                energies["masses"].append(mase)
            self.interact(t,floor=0)
            if t in times:
                rate(50)
                print("\n")
                print(t,"s")
                self.update_scene(m_obs, s_obs, COM)
        if plot_energies:
            self.plot_energies(energies)
    
    def get_COM(self):
        M = sum([m.mass for m in self.masses])
        COMx = sum([m.mass*m.position[0] for m in self.masses])/M
        COMy = sum([m.mass*m.position[1] for m in self.masses])/M
        COMz = sum([m.mass*m.position[2] for m in self.masses])/M
        return(np.array([COMx, COMy, COMz]))

def eval_springs(springs, render = False,T=2):
    start = time.time()
    sim = Simulation(dt=0.0002,T=T, w_masses=0.1, k_springs=10000)

    print("call to eval")
    for idx in range(len(sim.springs)):
        sim.springs[idx].constants["b"] = springs[idx][0]
        sim.springs[idx].constants["c"] = springs[idx][1]
        sim.springs[idx].constants["k"] = springs[idx][2]
        sim.springs[idx].spring_constant = springs[idx][2]

    if render:
        sim.run_simulation()
    else:
        p1 = sim.get_COM()
        for t in np.arange(0, sim.final_T, sim.increment):
            sim.interact(t,floor=0)
        p2 = sim.get_COM()
        print("eval time: ", time.time()-start)
        return(np.linalg.norm(np.array([p2[0],0,p2[2]])-np.array([p1[0],0,p1[2]])))


def evolve_robot(n_gen):

    #initialize population [done]
    #loop
        #select [done]
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
    sim = Simulation(dt=0.0002,T=1.5, w_masses=0.1, k_springs=10000)
    springs = sim.springs
    evolve = Evolution(springs=springs, ngen=n_gen)
    population_pool = evolve.initialize_population()
    
    # eval_springs(population_pool[0],True)
    # raise ValueError
    fits = [eval_springs(i) for i in population_pool]
    population_pool = [population_pool[i] for i in np.argsort(fits)]
    population_pool = list(reversed(population_pool))
    next_gen = []
    best_fits = []
    


    for i in range(evolve.generations):
        print("Began generation ", i, "...")
        population_pool = population_pool[:int(len(population_pool)/2)]
        while len(next_gen) < evolve.population_size:
            
            parents = evolve.ranked_selection(population_pool, 0.2)
            children = evolve.breed(parents)
            children = [evolve.mutate_individual(c) for c in children]
            [next_gen.append(p) for p in parents]
            [next_gen.append(c) for c in children]
        print("Done making next generation.")
        population_pool = [i for i in next_gen]
        fits = [eval_springs(i) for i in population_pool]
        population_pool = [population_pool[i] for i in np.argsort(fits)]
        population_pool = list(reversed(population_pool))
        gen_best = max(fits)
        print("Longest distance in generation was ",gen_best)
        best_fits.append(gen_best)
        next_gen = []
    input("Render best solution?")
    eval_springs(population_pool[0], render=True)
    
    return(best_fits, population_pool[0])
    


def evolve_robot2(n_gen):
     # initialize starting population
    # calculate the fitnesses for the population and sort population accordingly

    # repeat for each generation
        # initialize set for storing the individuals for the next generation
        # repeat until the population for the next generation is ready
            # select 2 parents from current generation
            # crossover the 2 parents
            # select who moves to the next generation via deterministic crowding
            # add the 2 diverse and fit bois to the next generation's population
        # replace the current generation population with the next generation's population
        # calculate the fitnesses for the population and sort population accordingly
        # check generation diversity
        # store the value of the best fitness of the generation
    
    #return the best individual of the latest generation
    sim = Simulation(dt=0.0002,T=1.5, w_masses=0.1, k_springs=10000)
    springs = sim.springs
    evolve = Evolution(springs=springs, ngen=n_gen)
    population_pool = evolve.initialize_population()
    # eval_springs(population_pool[0],True)
    # raise ValueError
    
    fits = [eval_springs(i) for i in population_pool]
    print("succeeded first set of evals")
    
    population_pool = [population_pool[i] for i in np.argsort(fits)]
    population_pool = list(reversed(population_pool))
    next_gen = []
    best_fits = []

    for i in range(n_gen):
        print("Starting generation ", str(i+1),"...")
        next_gen = set()
        while len(next_gen) < evolve.population_size:
            #print('len =', len(next_gen))
            parents = evolve.ranked_selection(population_pool, 0.2)
            children = evolve.breed(parents)
            evolve.det_crowd(next_gen,parents,children,population_pool,sim)
            next_gen.add(tuple(evolve.spawn_individual()))
            next_gen.add(tuple(evolve.spawn_individual()))

        
        population_pool = [list(i) for i in next_gen] 
        fits = [eval_springs(i) for i in population_pool]
        population_pool = [population_pool[i] for i in np.argsort(fits)]
        population_pool = list(reversed(population_pool))
        print("Gen best: ",max(fits))
        best_fits.append(max(fits))
    input("Render best solution?")
    eval_springs(population_pool[0], render=True)
    
    return(population_pool[0],best_fits)


class Evolution():
    def __init__(self, springs, ngen):
        self.generations = ngen
        self.population_size = 30
        self.mutation_rate = 0.8
        self.crossover_rate = 1
        self.springs = springs
        self.material_dict = {1:(0,0,1000),2:(0,0,20000),3:(0.25,0,5000),4:(0.25,pi,5000)}
    

    def det_crowd(self,next_pop,parents,children,population_pool,sim):
        p1_f = eval_springs(parents[0])
        p2_f = eval_springs(parents[1])
        c1_f = eval_springs(children[0]) 
        c2_f = eval_springs(children[1])
        if self.distance([parents[0], children[0]]) + self.distance([parents[1], children[1]]) < self.distance([parents[0], children[1]]) + self.distance([parents[1], children[0]]):
            if c2_f > p2_f:
                mut = self.mutate_individual(children[1])
                if eval_springs(mut) > c2_f:
                    next_pop.add(tuple(mut))
                else:
                    next_pop.add(tuple(children[1])) 
            else:
                next_pop.add(tuple(parents[1]))
            if c1_f > p1_f:
                mut = self.mutate_individual(children[0])
                if eval_springs(mut) > c1_f:
                    next_pop.add(tuple(mut))
                else:
                    next_pop.add(tuple(children[0]))
            else:
                next_pop.add(tuple(parents[0]))
        else:
            if c2_f > p1_f:
                mut = self.mutate_individual(children[1])
                if eval_springs(mut) > c2_f:
                    next_pop.add(tuple(mut))
                else:
                    next_pop.add(tuple(children[1])) 
            else:
                next_pop.add(tuple(parents[0]))
            if c1_f > p2_f:
                mut = self.mutate_individual(children[0])
                if eval_springs(mut) > c1_f:
                    next_pop.add(tuple(mut))
                else:
                    next_pop.add(tuple(children[0])) 
            else:
                next_pop.add(tuple(parents[1]))


    def spawn_individual(self):
        individual = []
        num_springs = len(self.springs)
        for s in range(num_springs):
                # 4 materials
                material = secrets.choice([1,2,3,4])
                # b = random.random() *(1 - 0) + 0
                # c = random.random() *(2*pi - 0) + 0
                # k = random.random() *(100000 - 0) + 0
                # individual.append((b,c,k))

                individual.append(self.material_dict[material])
        return(individual)

    def initialize_population(self):
        pool = []
        for i in range(self.population_size):
            pool.append(self.spawn_individual())
        return(pool)

    def mutate_individual(self, individual):
        
        if random.choices([True, False], weights=[self.mutation_rate, 1-self.mutation_rate], k=1)[0]:
            indiv = copy.deepcopy(individual)
            chosen_s_idx = secrets.choice(range(len(indiv)))
            newb = indiv[chosen_s_idx][0]+random.random() *(indiv[chosen_s_idx][0]*0.2 + indiv[chosen_s_idx][0]*0.2) - indiv[chosen_s_idx][0]*0.2
            newc = indiv[chosen_s_idx][1]+random.random() *(indiv[chosen_s_idx][1]*0.2 + indiv[chosen_s_idx][1]*0.2) - indiv[chosen_s_idx][1]*0.2
            newk = indiv[chosen_s_idx][2]+random.random() *(indiv[chosen_s_idx][2]*0.2 + indiv[chosen_s_idx][2]*0.2) - indiv[chosen_s_idx][2]*0.2
            indiv[chosen_s_idx] = (newb, newc, newk)
            return(indiv)
        else:
            return(individual)
    
    def breed(self, parents):
        n1 = secrets.choice(range(len(self.springs)))
        n2 = secrets.choice(range(len(self.springs)))
        if n1 == n2:
            n2 = secrets.choice(range(len(self.springs)))
        n1, n2 = sorted([n1,n2])
        child1 = parents[0][0:n1] + parents[1][n1:n2] + parents[0][n2:]
        child2 = parents[1][0:n1] + parents[0][n1:n2] + parents[1][n2:]
        return(child1, child2)

    def ranked_selection(self, population_pool,p_c):
        probabilities = np.array([((1-p_c)**((i+1)-1))*p_c for i in range(len(population_pool)-1)] + [(1-p_c)**(len(population_pool))])
        probabilities /= probabilities.sum()
        indices = list(range(len(population_pool)))
        chosen_ones = [population_pool[c] for c in np.random.choice(indices,size=2, p=probabilities, replace=False)]
        return(chosen_ones)

    def distance(self, individuals):
        b_tot = sum([abs(individuals[0][i][0] - individuals[1][i][0]) for i in range(len(individuals[0]))])
        c_tot = sum([abs(individuals[0][i][1] - individuals[1][i][1])/(2*pi) for i in range(len(individuals[0]))])
        k_tot = sum([abs(individuals[0][i][2] - individuals[1][i][2])/100000 for i in range(len(individuals[0]))])
        return(sum([b_tot,c_tot,k_tot]))

    



            











if __name__ == "__main__":
    # a = Simulation(dt=0.0002,T=10, w_masses=0.1, k_springs=5000)
    # a.run_simulation()
    # springs = a.springs
    # b = Evolution(springs,10)
    # pop = b.initialize_population()
    # print("1",b.distance([pop[0],pop[1]]))
    # print("2",b.distance([pop[1],pop[0]]))
    # print("3",b.distance([pop[0],pop[0]]))
    # print("4",b.distance([pop[0],pop[2]]))
    # print("5",b.distance([pop[0],pop[1]])+ b.distance([pop[1],pop[2]]))
    # print(a.eval_springs(indiv))
    # b.mutate_individual(indiv)
    # print(a.eval_springs(indiv))
    best_pal, best_fits = evolve_robot2(50)

    with open('best_indiv4.pkl', 'wb') as f:
        pickle.dump(best_pal, f)

    with open('all_fits4.pkl', 'wb') as f:
        pickle.dump(best_fits, f)


    # with open('best_indiv3.pkl', 'rb') as f:
    #     params = pickle.load(f)
    # with open('all_fits3.pkl', 'rb') as f:
    #     fits = pickle.load(f)
    # print(fits)

    # eval_springs(params,render=True,T=10)

    # print(best_pal, '\n')
    # print("======","\n")
    # print(best_fits)
    


    




