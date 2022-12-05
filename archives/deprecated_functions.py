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




def breed(parents):
    if len(parents[0]) == len(parents[1]):
        n1 = secrets.choice(range(len(parents[0])))
        n2 = secrets.choice(range(len(parents[0])))
        if n1 == n2:
            n2 = secrets.choice(range(len(parents[0])))
        n1, n2 = sorted([n1,n2])
        child1 = np.concatenate( (parents[0][0:n1], parents[1][n1:n2], parents[0][n2:]) )
        child2 = np.concatenate( (parents[1][0:n1], parents[0][n1:n2], parents[1][n2:]) ) 
    elif len(parents[0]) > len(parents[1]):
        end = len(parents[1])
        n1 = secrets.choice(range(end))
        n2 = secrets.choice(range(end))
        if n1 == n2:
            n2 = secrets.choice(range(end))
        n1, n2 = sorted([n1,n2])
        child1 = np.concatenate( (parents[0][0:n1], parents[1][n1:n2], parents[0][n2:end]) )
        child2 = np.concatenate( (parents[1][0:n1], parents[0][n1:n2], parents[1][n2:end]) ) 

    else:
        end = len(parents[0])
        n1 = secrets.choice(range(end))
        n2 = secrets.choice(range(end))
        if n1 == n2:
            n2 = secrets.choice(range(end))
        n1, n2 = sorted([n1,n2])
        child1 = np.concatenate( (parents[0][0:n1], parents[1][n1:n2], parents[0][n2:end]) )
        child2 = np.concatenate( (parents[1][0:n1], parents[0][n1:n2], parents[1][n2:end]) ) 

    return(child1, child2)

def breed_v2(parents,parent_indices, pool_masses, pool_springs):
    '''this func creates two children via two-point crossover of the 2 parents. if the parents are of
    different sizes, one child inherits the size of 1 of the parent and the other child inherits 
    its size from the other parent'''
    child1_size = len(parents[0])
    child2_size = len(parents[1])
    print("p1: ", len(parents[0]))
    print("p2: ", len(parents[1]))

    # first child
    n1 = secrets.choice(range(child1_size))
    n2 = secrets.choice([i for i in range(child1_size) if i != n1]) #this just makes sure that we dont pick the same n1 and n2
    n1,n2 = sorted([n1,n2])
    child1A = np.concatenate((parents[0][:n1,:] , parents[1][n1:n2,:] , parents[0][n2:child1_size,:]))
    child1B = np.concatenate((parents[1][:n1,:] , parents[0][n1:n2,:] , parents[1][n2:child1_size,:]))
    print(len(child1A),len(child1B))
    print("Ns: ", [n1,n2])
    children1 = [child1A,child1B]
    try:
        fits = [eval_springs(pool_masses[parent_indices[0]],pool_springs[parent_indices[0]], child) for child in children1]
    except IndexError:
        print("parent: ", parents[0])
        print("child A: ", child1A)
        print("child B: ", child1B)
        raise ValueError
    child1 = children1[fits.index(max(fits))]

    # for the second child
    n1 = secrets.choice(range(child2_size))
    n2 = secrets.choice([i for i in range(child2_size) if i != n1]) #this just makes sure that we dont pick the same n1 and n2
    n1,n2 = sorted([n1,n2])
    print(len(child1A),len(child1B))
    print("Ns: ", [n1,n2])
    child2A = np.concatenate((parents[0][:n1,:] , parents[1][n1:n2,:] , parents[0][n2:child2_size,:]))
    child2B = np.concatenate((parents[1][:n1,:] , parents[0][n1:n2,:] , parents[1][n2:child2_size,:]))
    children2 = [child2A,child2B]
    try:
        fits = [eval_springs(pool_masses[parent_indices[1]],pool_springs[parent_indices[1]], child) for child in children2]
    except IndexError:
        print("parent: ", parents[1])
        print("child A: ", child2A)
        print("child B: ", child2B)
        raise ValueError
    child2 = children2[fits.index(max(fits))]

    print("c1: ", len(child1))
    print("c2: ", len(child2))

    return(child1,child2)


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
    # return(True)
    print("succeeded first set of evals")
    
    population_pool = [population_pool[i] for i in np.argsort(fits)]
    # print(population_pool)
    population_pool = list(reversed(population_pool))
    next_gen = []
    best_fits = []

    for i in range(n_gen):
        print("Starting generation ", str(i+1),"...")
        next_gen = set()
        while len(next_gen) < pop_size:
            #print('len =', len(next_gen))
            parents = ranked_selection(population_pool, 0.2)
            print(parents)
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


def distance(individuals):
    b_tot = sum([abs(individuals[0][i][0] - individuals[1][i][0]) for i in range(len(individuals[0]))])
    c_tot = sum([abs(individuals[0][i][1] - individuals[1][i][1])/(2*pi) for i in range(len(individuals[0]))])
    k_tot = sum([abs(individuals[0][i][2] - individuals[1][i][2])/100000 for i in range(len(individuals[0]))])
    return(sum([b_tot,c_tot,k_tot]))

def spawn_individual(springs):
    individual = []
    num_springs = len(springs)
    individual = np.zeros(3*num_springs).reshape(num_springs,3)
    # material_dict = {1:np.array([0.0,0.0,1000.0]),2:np.array([0.0,0.0,20000.0]),3:np.array([0.25,0.0,5000.0]),4:np.array([0.25,np.pi,5000.0])}
    for s in range(num_springs):
            # random.random() *(upper - lower) + lower generates a random rational number in the range (lower,upper)
            b = random.random() *(1 - 0) + 0
            c = random.random() *(2*np.pi - 0) + 0
            k = random.random() *(10000 - 0) + 0
            # k = 1000.0
            # material = secrets.choice([1,2,3,4])
            # print(np.array([b,c,k]))
            # input("continue?\n")
            # individual[s] = np.copy(material_dict[material])
            individual[s] = np.array([b,c,k])
    return(individual)

def spawn_spring():
    # random.random() *(upper - lower) + lower generates a random rational number in the range (lower,upper)
    # material_dict = {1:np.array([0.0,0.0,1000.0]),2:np.array([0.0,0.0,20000.0]),3:np.array([0.25,0.0,5000.0]),4:np.array([0.25,np.pi,5000.0])}
    b = random.random() *(1 - 0) + 0
    c = random.random() *(2*np.pi - 0) + 0
    k = random.random() *(10000 - 0) + 0
    # k = 1000.0
    # material = secrets.choice([1,2,3,4])
    
    individual = np.array([b,c,k])
    # return(np.copy(material_dict[material]))
    return(individual)

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