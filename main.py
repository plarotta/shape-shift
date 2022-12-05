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

# mass is going to become a 5x3 np array
# mass = np.array(
#                   np.array([0,0,0]), {current position}
#                   np.array([0,0,0]), {velocity}
#                   np.array([0,0,0]), {acceleration}
#                   np.array([0,0,0]),  {net force}
#                   np.array([0,0,0]), {initial position}
# )

# spring is going to become a 3x2 np array
# spring = np.array(
#                   np.array([m1_idx, m2_idx, 0,0]),
#                   np.array([a,b,c,k])  {a = rest length, b = sinusoid amplitude, c = sinusoid phase shift, k = spring constant}
# )

def initialize_masses():
    masses = np.zeros(120).reshape(8,5,3)
    n = 0
    for x in [0,1]:
        for y in [0,1]:
            for z in [0,1]:
                masses[n][0] = np.array([float(x),float(y),float(z)])
                masses[n][4] = np.array([float(x),float(y),float(z)])
                n+=1
    return(masses)

def initialize_springs(masses):
    first = True
    for mass_idx1 in range(len(masses)):
        for mass_idx2 in range(mass_idx1+1,len(masses)):
            m1_p = masses[mass_idx1][0] #get pos
            m2_p = masses[mass_idx2][0] #get pos
            length = np.linalg.norm(m2_p-m1_p)
            if first == True:
                springs = np.array([[[mass_idx1, mass_idx2, 0, 0],[length, 0,0,5000]]])
                first = False
            else:
                springs = np.concatenate((springs, np.array([[[mass_idx1, mass_idx2, 0, 0],[length, 0,0,5000]]])))
    return(springs)
    
@njit()
def interact_fast(springs,masses,t,increment,mu_static,mu_kinetic,floor=-4,breath=False):
    for s in springs:
        l = get_spring_l(masses,s)
        l_vect = masses[int(s[0][1])][0] - masses[int(s[0][0])][0]  
        L0 = s[1][0] + s[1][1] * np.sin(4*t + s[1][2])
        f_k = s[1][3] * (l - L0)
        f_dir = f_k/np.sqrt(l_vect[0]**2 + l_vect[1]**2 + l_vect[2]**2)
        f_full = f_dir * l_vect
        masses[int(s[0][0])][3] = masses[int(s[0][0])][3] + f_full
        masses[int(s[0][1])][3] = masses[int(s[0][1])][3] - f_full
    for m in masses:   
        m[3] = m[3] + np.array([0,-9.81,0]) #gravity
        if m[0][1] <= floor:
            f_net = m[3]
            Fp = np.array([f_net[0], 0.0, f_net[2]])
            Fp_norm = np.sqrt(Fp[0]**2 + Fp[2]**2)
            Fn = np.array([0.0, f_net[1], 0.0])
            Fn_norm = f_net[1]
            if Fp_norm < abs(Fn_norm * mu_static): #friction
                m[3] = m[3] - np.array([f_net[0],0.0,f_net[2]])                
            else:
                dirFn = mu_kinetic*Fn_norm*np.array([f_net[0], 0.0, f_net[2]])/Fp_norm #friction
                m[3] = m[3] - np.array([dirFn[0],0.0,dirFn[2]])

            if m[0][1] < floor: #floor reaction force
                ry = 10000*(abs(m[0][1] - floor))
                m[3] = m[3] + np.array([0,ry,0])
        integrate(m,increment)

#useful unjitted interact function for debugging
def interact(springs,masses,t,increment,mu_static,mu_kinetic,floor=-4,breath=False):
    for s in springs:
        l = get_spring_l(masses,s)
        l_vect = masses[int(s[0][1])][0] - masses[int(s[0][0])][0]  
        L0 = s[1][0] + s[1][1] * np.sin(4*t + s[1][2])
        f_k = s[1][3] * (l - L0)
        f_dir = f_k/np.sqrt(l_vect[0]**2 + l_vect[1]**2 + l_vect[2]**2)
        f_full = f_dir * l_vect
        masses[int(s[0][0])][3] = masses[int(s[0][0])][3] + f_full
        masses[int(s[0][1])][3] = masses[int(s[0][1])][3] - f_full
    for m in masses:   
        m[3] = m[3] + np.array([0,-9.81,0]) #gravity
        if m[0][1] <= floor:
            f_net = m[3]
            Fp = np.array([f_net[0], 0.0, f_net[2]])
            Fp_norm = np.sqrt(Fp[0]**2 + Fp[2]**2)
            Fn = np.array([0.0, f_net[1], 0.0])
            Fn_norm = f_net[1]
            if Fp_norm < Fn_norm * mu_static: #friction
                m[3] = m[3] - np.array([f_net[0],0.0,f_net[2]])                
            else:
                dirFn = mu_kinetic*Fn_norm*np.array([f_net[0], 0.0, f_net[2]])/Fp_norm #friction
                m[3] = m[3] - np.array([dirFn[0],0.0,dirFn[2]])

            if m[0][1] < floor:
                ry = 10000*(abs(m[0][1] - floor))
                m[3] = m[3] + np.array([0,ry,0])
        integrate(m,increment)

@njit()
def integrate(mass,increment):
    mass[2] = mass[3]/0.1
    mass[1] = mass[1] + mass[2]*increment
    mass[1] = mass[1] * 0.9985
    mass[0] = mass[0] + mass[1]*increment
    mass[3] = np.array([0.0,0.0,0.0])

    
@njit()
def get_spring_l(masses, spring):
    m1_idx1 = spring[0][0]
    m2_idx2 = spring[0][1]
    m1_p = masses[int(m1_idx1)][0]
    m2_p = masses[int(m2_idx2)][0]
    diff = m2_p-m1_p
    return(np.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2))
        
def initialize_scene(masses,springs,z, breath=False):
    # walls
      
    scene = canvas(width=1200,height=750,background=color.black)
    # scene.camera.pos = vector(5,25,5)
    v = wtext(text="0 m/s")  
    
    if not breath:
        floor = box(pos=vector(0.5,z-0.06,0), height=0.02, length=100, width=100,color=color.white)
        for x in range(-50,53,3):
            gridx = curve(pos=[vector(x,z-0.06,50),vector(x,z-0.06,-50)],color=color.black)
        for y in range(-50,53,3):
            gridx = curve(pos=[vector(50,z-0.06,y),vector(-50,z-0.06,y)],color=color.black)
   
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
    scene.camera.follow(COM)
    return(m_plots,s_plots,COM,v)

def update_scene(masses, springs, m_plots, s_plots, COM,v,speed):
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
    v.text = str(str(round(speed,3))+" m/s")

def run_simulation(masses, springs,final_T,increment, mu_static, mu_kinetic,floor,plot_energies=False, debug_mode = False):
    if debug_mode:
        masses = initialize_masses()
        springs = initialize_springs(masses)
    # masses[0][3] = np.array([300000,600000,400000])
    m_obs, s_obs, COM,v = initialize_scene(masses, springs,z=floor-0.06)
    
    time.sleep(3)
    
    times = np.arange(0, final_T, 0.02)
    if plot_energies:
        energies = {"springs": [], "masses": []}
    for t in np.arange(0, final_T, increment):
        if t in times:
            p1 = get_COM(masses)
            interact(springs,masses,t,increment, mu_static,mu_kinetic,floor)
            p2 = get_COM(masses)
            disp = np.linalg.norm(np.array([p2[0],0,p2[2]])-np.array([p1[0],0,p1[2]]))
            speed = disp/increment
            rate(50)
            update_scene(masses, springs,m_obs, s_obs, COM,v,speed)
        else:
            interact(springs,masses,t,increment, mu_static,mu_kinetic,floor)
    if plot_energies:
        plot_energies(energies)
# @njit()
def get_COM(masses):
    M = sum([0.1 for m in masses])
    masses[:,0,0]
    COMx = np.sum(masses[:,0,0]*0.1)/M
    COMy = np.sum(masses[:,0,1]*0.1)/M
    COMz = np.sum(masses[:,0,2]*0.1)/M
    return(np.array([COMx, COMy, COMz]))

def eval_springs(masses,springs,spring_constants,final_T=4,increment=0.0002, render = False):
    start = time.time()
    print("call to eval")
    for idx in range(len(springs)):
        springs[idx][1][1] = spring_constants[idx][0]
        springs[idx][1][2] = spring_constants[idx][1]
        springs[idx][1][3] = spring_constants[idx][2]
    if render:
        run_simulation(masses, springs,6,0.0002,0.9,0.8,0)
    else:
        p1 = get_COM(masses)
        for t in np.arange(0, final_T, increment):
            interact_fast(springs,masses,t,increment,0.9,0.8,floor=0)
        p2 = get_COM(masses)
        print("eval time: ", time.time()-start)
        rehome(masses) #reset current state of the masses
        return(np.linalg.norm(np.array([p2[0],0,p2[2]])-np.array([p1[0],0,p1[2]])))

def rehome(masses):
    for m in masses:
        pos0 = m[4]
        m[0] = pos0
        m[1] = np.array([0.0, 0.0, 0.0])
        m[2] = np.array([0.0, 0.0, 0.0])
        m[3] = np.array([0.0, 0.0, 0.0])
    
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


if __name__ == "__main__":

    
    # for i in range(20):
    #     best_fits,population_pool,pool_masses,pool_springs = evolve_robot(350,gcp=True)

    #     with open( str("final_fits" + str(i) + ".pkl"), "wb" ) as f:
    #         pickle.dump(best_fits,f)
        
    #     with open( str("final_pop_pools" + str(i) + ".pkl"), "wb" ) as f:
    #         pickle.dump(population_pool,f)
        
    #     with open( str("final_mass_pools" + str(i) + ".pkl"), "wb" ) as f:
    #         pickle.dump(pool_masses,f)
        
    #     with open( str("final_spring_pools" + str(i) + ".pkl"), "wb" ) as f:
    #         pickle.dump(pool_springs,f)
        
    #     print(str("succeeded round " + str(i) + " of full run-through.\n"))
        





    # masses = initialize_masses()
    # springs = initialize_springs(masses)

    # initialize_scene(masses,springs,0, breath=False)
    
    # best_fits,population_pool,pool_masses,pool_springs = evolve_robot(3)


    # file_n = "3"

    # file = open('final3_num'+file_n+'/final3_mass_pools' +file_n+'.pkl', 'rb')
    # # dump information to that file
    # masses = pickle.load(file)
    # # close the file
    # file.close()
    # file = open('final3_num' +file_n+'/final3_spring_pools'+file_n+'.pkl', 'rb')
    # # dump information to that file
    # springs = pickle.load(file)
    # # close the file
    # file.close()
    # file = open('final3_num'+file_n+'/final3_pop_pools'+file_n+'.pkl', 'rb')
    # # dump information to that file
    # constants = pickle.load(file)
    # # close the file
    # file.close()
    # initialize_scene(masses,springs,0, breath=False)
    # eval_springs(masses, springs, constants,render=True)


    # file_n = "2"
    # file = open('final5_good_'+file_n+'/final5good_mass_pools' +file_n+'.pkl', 'rb')
    # # dump information to that file
    # masses = pickle.load(file)
    # # close the file
    # file.close()
    # file = open('final5_good_' +file_n+'/final5good_spring_pools'+file_n+'.pkl', 'rb')
    # # dump information to that file
    # springs = pickle.load(file)
    # # close the file
    # file.close()
    # file = open('final5_good_'+file_n+'/final5good_pop_pools'+file_n+'.pkl', 'rb')
    # # dump information to that file
    # constants = pickle.load(file)
    # # close the file
    # file.close()


    # # # run_simulation(masses, springs,6,0.0002,0.9,0.8,floor=-4)
    # initialize_scene(masses,springs,0, breath=False)

    # eval_springs(masses, springs, constants,render=True)

    file_n="2"
    file = open('final4_good_' +file_n+'/final4good_fits'+file_n+'.pkl', 'rb')
    # dump information to that file
    fits = pickle.load(file)


    plt.plot(fits)
    plt.xlabel("Generation # (44 evaluations per generation)")
    plt.ylabel("Fitness (total COM XY displacement in meters)")
    plt.title("Learning Curve")
    plt.show()
