import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from vpython import *
import secrets
import random
import copy
import pickle
from numba import njit, vectorize, jit
import cProfile, pstats
import re
import sys
import os
sys.path.append(os.getcwd())


def evolve_robot(n_gen,gcp=False):
    # params
    pop_size = 42
    mut_rate = .6

    # initialize ledgers
    next_gen = []
    next_masses = []
    next_springs = []
    best_fits = []

    # initialialize starting population
    masses = initialize_masses()
    springs = initialize_springs(masses)
    population_pool = initialize_population(pop_size,springs)
    pool_springs = [springs for idx in range(len(population_pool))]
    pool_masses = [masses for idx in range(len(population_pool))] 

    # mutate every individual of the starting population
    for idx in range(len(population_pool)):
        m,s,c = mutate_morphology(pool_masses[idx], pool_springs[idx], population_pool[idx])
        pool_masses[idx] = m
        pool_springs[idx] = s
        population_pool[idx] = c

    # fitness-sort population
    fits = [eval_springs(pool_masses[i],pool_springs[i],population_pool[i]) for i in range(len(population_pool))]
    arginds = np.argsort(fits)
    population_pool = [population_pool[i] for i in arginds]
    pool_masses = [pool_masses[i] for i in arginds]
    pool_springs = [pool_springs[i] for i in arginds]
    population_pool = list(reversed(population_pool))
    pool_masses = list(reversed(pool_masses))
    pool_springs = list(reversed(pool_springs))

    for i in range(n_gen):
        print("Began generation ", i, "...\n")
        while len(next_gen) < pop_size:
            parents,parent_indices = ranked_selection(population_pool, 0.15) 
            children = breed_v3(parents,parent_indices, pool_masses, pool_springs)
            children = [mutate_individual(mut_rate,c) for c in children]
            
            # add parents to next generation
            [next_gen.append(p) for p in parents] 
            [next_masses.append(pool_masses[parent_indices[i]]) for i in range(2)]
            [next_springs.append(pool_springs[parent_indices[i]]) for i in range(2)]

            # add children to next generation
            [next_gen.append(c) for c in children]
            [next_masses.append(pool_masses[parent_indices[i]]) for i in range(2)]
            [next_springs.append(pool_springs[parent_indices[i]]) for i in range(2)]

        print("Done making next generation.\n")
        # next generation becomes current population
        population_pool = [np.copy(i) for i in next_gen]
        pool_masses = [np.copy(j) for j in next_masses]
        pool_springs = [np.copy(k) for k in next_springs]
        
        # fitness-sort new current population
        fits = [eval_springs(pool_masses[i],pool_springs[i],population_pool[i]) for i in range(len(population_pool))]
        arginds = np.argsort(fits)
        population_pool = [population_pool[i] for i in arginds] 
        pool_masses = [pool_masses[i] for i in arginds]
        pool_springs = [pool_springs[i] for i in arginds]
        population_pool = list(reversed(population_pool))
        pool_masses = list(reversed(pool_masses))
        pool_springs = list(reversed(pool_springs))

        # keep generation's best
        gen_best = max(fits)
        best_fits.append(gen_best)
        print("Longest distance in generation was ",gen_best,"\n")
        print("Size of fittest robot was ", len(population_pool[0]),"\n")
        
        # reset ledgers
        next_gen = []
        next_springs = []
        next_masses = []

        # mutate morphology on the bottom 50% 
        for idx in range(5,len(population_pool)):
            m,s,c = mutate_morphology(pool_masses[idx], pool_springs[idx], population_pool[idx])
            pool_masses[idx] = m
            pool_springs[idx] = s
            population_pool[idx] = c
        
    if not gcp:
        input("Render best solution? \n")
        eval_springs(pool_masses[0], pool_springs[0],population_pool[0], render=True) 
    
    return(best_fits, population_pool[0],pool_masses[0],pool_springs[0])

def breed_v3(parents,parent_indices, pool_masses, pool_springs):
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



def ranked_selection( population_pool,p_c):
    probabilities = np.array([((1-p_c)**((i+1)-1))*p_c for i in range(len(population_pool)-1)] + [(1-p_c)**(len(population_pool))])
    probabilities /= probabilities.sum()
    indices = list(range(len(population_pool)))
    indices = np.random.choice(indices,size=2, p=probabilities, replace=False)
    chosen_ones = [population_pool[c] for c in indices]
    return(chosen_ones,indices)


def append_point(point,masses,springs,spring_constants):
    mass = np.array([point,np.array([0.0,0.0,0.0]),np.array([0.0,0.0,0.0]),np.array([0.0,0.0,0.0]),point])
    masses = np.append(masses,mass).reshape(len(masses)+1,5,3)
    for mass_idx1 in range(len(masses)-1):
        m1_p = masses[mass_idx1][0] #get pos
        length = np.linalg.norm(m1_p-point)
        spring = np.array([[[mass_idx1, len(masses)-1, 0, 0],[length, 0,0,5000]]])
        springs = np.concatenate((springs,spring))
        spring_constants = np.concatenate((spring_constants, np.array([spawn_spring()])))
    return(masses, springs,spring_constants)



def fatten_cube(masses,springs,spring_constants):
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
        masses2,springs2,spring_constants2 = append_point(point,masses,springs,spring_constants)
    return(masses2,springs2,spring_constants2)

def slim_cube(masses,springs,spring_constants):
    chosen_mass_idx = secrets.choice(range(len(masses)))
    masses2 = np.delete(masses,chosen_mass_idx,axis=0)
    springs_idxs = springs[:,0,0:2]
    springs_to_remove = np.argwhere(springs_idxs == float(chosen_mass_idx))[:,0]
    springs2 = np.delete(springs, springs_to_remove,axis=0)
    spring_constants2 = np.delete(spring_constants, springs_to_remove,axis=0)
    springs2[:,0,0:2] = np.where(springs2[:,0,0:2] > float(chosen_mass_idx),springs2[:,0,0:2] -1,springs2[:,0,0:2])
    return(masses2,springs2,spring_constants2)

def mutate_morphology(masses,springs,spring_constants):
    operation = np.random.choice(["Fatten", "Slim"],p=[0.75,0.25])
    # operation = "Fatten"
    # print(len(masses))
    if len(masses) < 10 or operation == "Fatten":
    # if operation == "Fatten":
        masses2,springs2,spring_constants2 = fatten_cube(masses,springs,spring_constants)
    else:
        masses2,springs2,spring_constants2 = slim_cube(masses,springs,spring_constants) #this seems to work fine
    
    return(masses2,springs2,spring_constants2)






def spawn_individual(springs):
    individual = []
    num_springs = len(springs)
    individual = np.zeros(3*num_springs).reshape(num_springs,3)
    for s in range(num_springs):
            b = random.random() *(1 - 0) + 0
            c = random.random() *(2*np.pi - 0) + 0
            k = random.random() *(10000 - 0) + 0
            individual[s] = np.array([b,c,k])
    return(individual)

def spawn_spring():
    b = random.random() *(1 - 0) + 0
    c = random.random() *(2*np.pi - 0) + 0
    k = random.random() *(10000 - 0) + 0
    individual = np.array([b,c,k])
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
        newb = indiv[chosen_s_idx][0]+random.random() *(indiv[chosen_s_idx][0]*0.35 + indiv[chosen_s_idx][0]*0.35) - indiv[chosen_s_idx][0]*0.35
        newc = indiv[chosen_s_idx][1]+random.random() *(indiv[chosen_s_idx][1]*0.35 + indiv[chosen_s_idx][1]*0.35) - indiv[chosen_s_idx][1]*0.35
        newk = indiv[chosen_s_idx][2]+random.random() *(indiv[chosen_s_idx][2]*0.35 + indiv[chosen_s_idx][2]*0.35) - indiv[chosen_s_idx][2]*0.35
        indiv[chosen_s_idx] = (newb, newc, newk)
        return(indiv)
    else:
        return(individual)
    
