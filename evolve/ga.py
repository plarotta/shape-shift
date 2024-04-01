import numpy as np
from vpython import *
import secrets
import random
import copy
import sys
import os
sys.path.append(os.getcwd())
from evolve.evolution import Evolution


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
    
    return(best_fits, population_pool[0],pool_masses[0],pool_springs[0])

if __name__ == "__main__":
    pass