import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from vpython import *
import secrets
import random
import copy
import pickle
from numba import njit
import cProfile
import re

    


# mass is going to become a 3x4 np array
# mass = np.array(
#                   np.array([0,0,0]), {position}
#                   np.array([0,0,0]), {velocity}
#                   np.array([0,0,0]), {acceleration}
#                   np.array([0,0,0])  {net force}
# )

# spring is going to become a 3x2 np array
# spring = np.array(
#                   np.array([m1_idx, m2_idx, 0,0]),
#                   np.array([a,b,c,k])  {a = rest length, b = sinusoid amplitude, c = sinusoid phase shift, k = spring constant}
# )

# @njit()
def initialize_masses():
    first = True
    for x in [0,1]:
        for y in [0,1]:
            for z in [0,1]:
                if first == True:
                    masses = np.array([[[x,y,z],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]])
                    first = False
                else:
                    masses = np.concatenate((masses, np.array([[[x,y,z],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]])))
    return(masses)

# @njit()
def initialize_springs(masses):
    first = True
    for mass_idx1 in range(len(masses)):
        for mass_idx2 in range(mass_idx1+1,len(masses)):
            m1_p = masses[mass_idx1][0] #get pos
            m2_p = masses[mass_idx2][0] #get pos
            length = np.linalg.norm(m2_p-m1_p)
            # print(length)
            if first == True:
                springs = np.array([[[mass_idx1, mass_idx2, 0, 0],[length, 0,0,5000]]])
                first = False
            else:
                springs = np.concatenate((springs, np.array([[[mass_idx1, mass_idx2, 0, 0],[length, 0,0,5000]]])))
    return(springs)

def interact(springs,masses,t,increment,mu_static,mu_kinetic,floor=-4,breath=False):
    for s in springs:
        l = get_spring_l(masses,s)
        l_vect = np.array( masses[int(s[0][1])][0] -  masses[int(s[0][0])][0]  )
        L0 = s[1][0] + s[1][1] * np.sin(5*t + s[1][2])
        f_k = s[1][3] * (l - L0)
        f_dir = f_k/np.sqrt(l_vect[0]**2 + l_vect[1]**2 + l_vect[2]**2)
        f_kx = f_dir * l_vect[0]
        f_ky = f_dir * l_vect[1]
        f_kz = f_dir * l_vect[2]
        masses[int(s[0][0])][3] = masses[int(s[0][0])][3] + np.array([f_kx, f_ky, f_kz])
        masses[int(s[0][1])][3] = masses[int(s[0][1])][3] + np.array([-f_kx, -f_ky, -f_kz]) 
    start = time.time()
    for m in masses:   
        m[3] = m[3] + np.array([0,-9.81,0]) #gravity
        if m[0][1] <= floor:
            
            f_netx=m[3][0]
            f_nety=m[3][1]
            f_netz=m[3][2]
            Fp = np.array([f_netx, 0.0, f_netz])
            Fp_norm = np.linalg.norm(Fp)
            Fn = np.array([0.0, f_nety, 0.0])
            Fn_norm = np.linalg.norm(Fn)
            if Fp_norm < Fn_norm * mu_static: #friction
                m[3] = m[3] - np.array([f_netx,0.0,f_netz])
                

            else:
                dirFn = mu_kinetic*Fn_norm*np.array([f_netx, 0.0, f_netz])/Fp_norm #friction
                m[3] = m[3] - np.array([dirFn[0],0.0,dirFn[2]])

            if m[0][1] < floor:
                ry = 10000*(abs(round(m[0][1],3) - floor))
                m[3] = m[3] + np.array([0,ry,0])
        integrate(m,increment)

# @njit()
def integrate(mass,increment):
    mass[2] = mass[3]/0.1
    mass[1] = mass[1] + mass[2]*increment
    mass[1] = mass[1] * 0.9985
    mass[0] = mass[0] + mass[1]*increment
    mass[3] = np.array([0.0,0.0,0.0])

    

def get_spring_l(masses, spring):
    m1_idx1 = spring[0][0]
    m2_idx2 = spring[0][1]
    m1_p = masses[int(m1_idx1)][0]
    m2_p = masses[int(m2_idx2)][0]
    return(np.linalg.norm(m2_p-m1_p))
        
def initialize_scene(masses,springs,z, breath=False):
    # walls
    if not breath:
        floor = box(pos=vector(0.5,z-0.06,0), height=0.02, length=7, width=7,color=color.white)

    m_plots = []
    for mass in masses:
        m_temp = sphere(pos=vector(mass[0][0],mass[0][1],mass[0][2]), radius=0.06, color=color.green)
        m_plots.append(m_temp)

    s_plots = []
    
    for spring in springs:
        m1_pos = masses[int(spring[0][0])][0]
        m2_pos = masses[int(spring[0][1])][0]
        s_temp = cylinder(pos=vector(m1_pos[0], m1_pos[1], m1_pos[2]), axis=vector(m2_pos[0]-m1_pos[0], m2_pos[1]-m1_pos[1], m2_pos[2]-m1_pos[2]), length=get_spring_l(masses,spring),color=color.red, radius=0.01)
        s_plots.append(s_temp)
    COM_pos = get_COM(masses)
    COM = sphere(pos=vector(COM_pos[0],COM_pos[1],COM_pos[2]), radius=0.06, color=color.yellow)

    return(m_plots,s_plots,COM)

def update_scene(masses, springs, m_plots, s_plots, COM):
    for idx,m in enumerate(m_plots):
        m.pos = vector(masses[idx][0][0], masses[idx][0][1], masses[idx][0][2])
    
    for idx,s in enumerate(s_plots):
        m1_pos = masses[int(springs[idx][0][0])][0]
        m2_pos = masses[int(springs[idx][0][1])][0]
        s.pos = vector(m1_pos[0], m1_pos[1], m1_pos[2])
        s.axis = vector(m2_pos[0]-m1_pos[0], m2_pos[1]-m1_pos[1], m2_pos[2]-m1_pos[2])
        s.length= get_spring_l(masses,springs[idx])
    COM_pos = get_COM(masses)
    COM.pos = vector(COM_pos[0],COM_pos[1],COM_pos[2])

def run_simulation(final_T,increment, mu_static, mu_kinetic,floor,plot_energies=False):
    masses = initialize_masses()
    springs = initialize_springs(masses)
    masses[0][3] = np.array([3000,60000,4000])
    m_obs, s_obs, COM = initialize_scene(masses, springs,z=floor-0.06)
    
    time.sleep(3)
    
    times = np.arange(0, final_T, 0.02)
    if plot_energies:
        energies = {"springs": [], "masses": []}
    for t in np.arange(0, final_T, increment):
        interact(springs,masses,t,increment, mu_static,mu_kinetic,floor)
        
        if t in times:
            rate(50)
            update_scene(masses, springs,m_obs, s_obs, COM)
    if plot_energies:
        plot_energies(energies)

def get_COM(masses):
    M = sum([0.1 for m in masses])
    COMx = sum([0.1*m[0][0] for m in masses])/M
    COMy = sum([0.1*m[0][1] for m in masses])/M
    COMz = sum([0.1*m[0][2] for m in masses])/M
    return(np.array([COMx, COMy, COMz]))

def eval_springs(springs_in,final_T=2,increment=0.0002, render = False):
    start = time.time()
    masses = initialize_masses()
    springs = initialize_springs(masses)

    print("call to eval")
    for idx in range(len(springs)):
        springs[idx][1][1] = springs_in[idx][0]
        springs[idx][1][2] = springs_in[idx][1]
        springs[idx][1][3] = springs_in[idx][2]
    if render:
        run_simulation()
    else:
        p1 = get_COM(masses)
        for t in np.arange(0, final_T, increment):
            interact(springs,masses,t,increment,0.9,0.8,floor=-4)
        p2 = get_COM(masses)
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
    pop_size = 10
    mut_rate = 0.8
    masses = initialize_masses(0.1)
    springs = initialize_springs(masses)
    population_pool = initialize_population(pop_size)
    
    
    fits = [eval_springs(i) for i in population_pool]
    population_pool = [population_pool[i] for i in np.argsort(fits)]
    population_pool = list(reversed(population_pool))
    next_gen = []
    best_fits = []
    # eval_springs(population_pool[0],True)
    # raise ValueError


    for i in range(n_gen):
        print("Began generation ", i, "...")
        population_pool = population_pool[:int(len(population_pool)/2)]
        while len(next_gen) < pop_size:
            
            parents = ranked_selection(population_pool, 0.2)
            children = breed(parents)
            children = [mutate_individual(mut_rate,c) for c in children]
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
    pop_size = 5
    masses = initialize_masses()
    springs = initialize_springs(masses)
    population_pool = initialize_population(pop_size,springs)
    # eval_springs(population_pool[0],True)
    # raise ValueError
    
    fits = [eval_springs(i) for i in population_pool]
    return(True)
    print("succeeded first set of evals")
    
    population_pool = [population_pool[i] for i in np.argsort(fits)]
    population_pool = list(reversed(population_pool))
    next_gen = []
    best_fits = []

    for i in range(n_gen):
        print("Starting generation ", str(i+1),"...")
        next_gen = set()
        while len(next_gen) < pop_size:
            #print('len =', len(next_gen))
            parents = ranked_selection(population_pool, 0.2)
            children = breed(parents)
            det_crowd(next_gen,parents,children,population_pool)
            next_gen.add(tuple(spawn_individual(springs)))
            next_gen.add(tuple(spawn_individual(springs)))

        
        population_pool = [list(i) for i in next_gen] 
        fits = [eval_springs(i) for i in population_pool]
        population_pool = [population_pool[i] for i in np.argsort(fits)]
        population_pool = list(reversed(population_pool))
        print("Gen best: ",max(fits))
        best_fits.append(max(fits))
    input("Render best solution?")
    eval_springs(population_pool[0], render=True)
    
    return(population_pool[0],best_fits)





def det_crowd(next_pop,parents,children,population_pool):
    p1_f = eval_springs(parents[0])
    p2_f = eval_springs(parents[1])
    c1_f = eval_springs(children[0]) 
    c2_f = eval_springs(children[1])
    if distance([parents[0], children[0]]) + distance([parents[1], children[1]]) < distance([parents[0], children[1]]) + distance([parents[1], children[0]]):
        if c2_f > p2_f:
            mut = mutate_individual(children[1])
            if eval_springs(mut) > c2_f:
                next_pop.add(tuple(mut))
            else:
                next_pop.add(tuple(children[1])) 
        else:
            next_pop.add(tuple(parents[1]))
        if c1_f > p1_f:
            mut = mutate_individual(children[0])
            if eval_springs(mut) > c1_f:
                next_pop.add(tuple(mut))
            else:
                next_pop.add(tuple(children[0]))
        else:
            next_pop.add(tuple(parents[0]))
    else:
        if c2_f > p1_f:
            mut = mutate_individual(children[1])
            if eval_springs(mut) > c2_f:
                next_pop.add(tuple(mut))
            else:
                next_pop.add(tuple(children[1])) 
        else:
            next_pop.add(tuple(parents[0]))
        if c1_f > p2_f:
            mut = mutate_individual(children[0])
            if eval_springs(mut) > c1_f:
                next_pop.add(tuple(mut))
            else:
                next_pop.add(tuple(children[0])) 
        else:
            next_pop.add(tuple(parents[1]))


def spawn_individual(springs):
    individual = []
    num_springs = len(springs)
    for s in range(num_springs):
            # random.random() *(upper - lower) + lower generates a random rational number in the range (lower,upper)
            b = random.random() *(1 - 0) + 0
            c = random.random() *(2*pi - 0) + 0
            k = random.random() *(100000 - 0) + 0
            individual.append((b,c,k))
    return(individual)

def initialize_population(population_size,springs):
    pool = []
    for i in range(population_size):
        pool.append(spawn_individual(springs))
    return(pool)

def mutate_individual(mutation_rate, individual):
    
    if random.choices([True, False], weights=[mutation_rate, 1-mutation_rate], k=1)[0]:
        indiv = copy.deepcopy(individual)
        chosen_s_idx = secrets.choice(range(len(indiv)))
        newb = indiv[chosen_s_idx][0]+random.random() *(indiv[chosen_s_idx][0]*0.2 + indiv[chosen_s_idx][0]*0.2) - indiv[chosen_s_idx][0]*0.2
        newc = indiv[chosen_s_idx][1]+random.random() *(indiv[chosen_s_idx][1]*0.2 + indiv[chosen_s_idx][1]*0.2) - indiv[chosen_s_idx][1]*0.2
        newk = indiv[chosen_s_idx][2]+random.random() *(indiv[chosen_s_idx][2]*0.2 + indiv[chosen_s_idx][2]*0.2) - indiv[chosen_s_idx][2]*0.2
        indiv[chosen_s_idx] = (newb, newc, newk)
        return(indiv)
    else:
        return(individual)

def breed(parents):
    n1 = secrets.choice(range(len(parents[0])))
    n2 = secrets.choice(range(len(parents[0])))
    if n1 == n2:
        n2 = secrets.choice(range(len(parents[0])))
    n1, n2 = sorted([n1,n2])
    child1 = parents[0][0:n1] + parents[1][n1:n2] + parents[0][n2:]
    child2 = parents[1][0:n1] + parents[0][n1:n2] + parents[1][n2:]
    return(child1, child2)

def ranked_selection( population_pool,p_c):
    probabilities = np.array([((1-p_c)**((i+1)-1))*p_c for i in range(len(population_pool)-1)] + [(1-p_c)**(len(population_pool))])
    probabilities /= probabilities.sum()
    indices = list(range(len(population_pool)))
    chosen_ones = [population_pool[c] for c in np.random.choice(indices,size=2, p=probabilities, replace=False)]
    return(chosen_ones)

def distance(individuals):
    b_tot = sum([abs(individuals[0][i][0] - individuals[1][i][0]) for i in range(len(individuals[0]))])
    c_tot = sum([abs(individuals[0][i][1] - individuals[1][i][1])/(2*pi) for i in range(len(individuals[0]))])
    k_tot = sum([abs(individuals[0][i][2] - individuals[1][i][2])/100000 for i in range(len(individuals[0]))])
    return(sum([b_tot,c_tot,k_tot]))

    
    


            











if __name__ == "__main__":
    # run_simulation(5,0.0002,0.9,0.7,-4)
    # masses = initialize_masses()
    # print("Masses: ", masses)
    # print("Springs: ", initialize_springs(masses))

    # start = time.time()
    # [initialize_masses() for i in range(100000)]
    # print("time: ", time.time()-start)
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
    cProfile.run('evolve_robot2(1)')

    # best_pal, best_fits = evolve_robot2(5)


    # with open('best_indiv.pkl', 'wb') as f:
    #     pickle.dump(best_pal, f)

    # with open('all_fits.pkl', 'wb') as f:
    #     pickle.dump(best_fits, f)


    # print(best_pal, '\n')
    # print("======","\n")
    # print(best_fits)
    


    




