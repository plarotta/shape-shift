import numpy as np
from vpython import *
import sys
import os
sys.path.append(os.getcwd())
from evolve.evolution import Evolution
from simulate.simulation import Simulation
from visualize.v_wrappers import *


def evolve(n_gen):
    # params
    pop_size = 24
    mut_rate = .6

    # initialize ledgers
    next_gen = []
    next_masses = []
    next_springs = []
    best_fits = []

    # initialialize starting population
    evo = Evolution()
    evo.population = evo.initialize_population(pop_size, evo.springs)
    # population_pool = initialize_population(pop_size,springs)
    pool_springs = [evo.springs for idx in range(len(evo.population))]
    pool_masses = [evo.masses for idx in range(len(evo.population))] 

    # mutate every individual of the starting population
    for idx in range(len(evo.population)):
        m,s,c = evo.mutate_morphology(pool_masses[idx], pool_springs[idx], evo.population[idx])
        pool_masses[idx] = m
        pool_springs[idx] = s
        evo.population[idx] = c

    # fitness-sort population
    fits = [evo.eval_springs(pool_masses[i],pool_springs[i],evo.population[i]) for i in range(len(evo.population))]
    arginds = np.argsort(fits)
    evo.population = [evo.population[i] for i in arginds]
    pool_masses = [pool_masses[i] for i in arginds]
    pool_springs = [pool_springs[i] for i in arginds]
    evo.population = list(reversed(evo.population))
    pool_masses = list(reversed(pool_masses))
    pool_springs = list(reversed(pool_springs))


    for i in range(n_gen):
        while len(next_gen) < pop_size:
            parents,parent_indices = evo.ranked_selection(evo.population, 0.15) 
            children = evo.breed_v3(parents,parent_indices, pool_masses, pool_springs)
            children = [evo.mutate_individual(mut_rate,c) for c in children]
            
            # add parents to next generation
            [next_gen.append(p) for p in parents] 
            [next_masses.append(pool_masses[parent_indices[i]]) for i in range(2)]
            [next_springs.append(pool_springs[parent_indices[i]]) for i in range(2)]

            # add children to next generation
            [next_gen.append(c) for c in children]
            [next_masses.append(pool_masses[parent_indices[i]]) for i in range(2)]
            [next_springs.append(pool_springs[parent_indices[i]]) for i in range(2)]

        # next generation becomes current population
        evo.population = [np.copy(i) for i in next_gen]
        pool_masses = [np.copy(j) for j in next_masses]
        pool_springs = [np.copy(k) for k in next_springs]
        
        # fitness-sort new current population
        fits = [evo.eval_springs(pool_masses[i],pool_springs[i],evo.population[i]) for i in range(len(evo.population))]
        arginds = np.argsort(fits)
        evo.population = [evo.population[i] for i in arginds] 
        pool_masses = [pool_masses[i] for i in arginds]
        pool_springs = [pool_springs[i] for i in arginds]
        evo.population = list(reversed(evo.population))
        pool_masses = list(reversed(pool_masses))
        pool_springs = list(reversed(pool_springs))

        # keep generation's best
        gen_best = max(fits)
        best_fits.append(gen_best)
        print(f"Gen {i} | Longest distance in generation was {gen_best:.2f} | Size of fittest robot was {len(evo.population[0])}")
        
        # reset ledgers
        next_gen = []
        next_springs = []
        next_masses = []

        # mutate morphology on the bottom 50% 
        for idx in range(5,len(evo.population)):
            m,s,c = evo.mutate_morphology(pool_masses[idx], pool_springs[idx], evo.population[idx])
            pool_masses[idx] = m
            pool_springs[idx] = s
            evo.population[idx] = c
    
    return(best_fits, evo.population[0],pool_masses[0],pool_springs[0])

if __name__ == "__main__":
    _,_, masses, springs = evolve(30)
    a = Simulation(masses=masses, springs=springs)
    sim_path = a.run_simulation(sim_length=20, time_step=0.001, log_k=5, mu_s=0.5, mu_k=0.3, floor_z_position=0, save=True)
    visualize_simulation(sim_path, floor=0)