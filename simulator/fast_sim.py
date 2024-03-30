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
    masses = np.zeros((8,5,3))
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
            interact_fast(springs,masses,t,increment, mu_static,mu_kinetic,floor)
            p2 = get_COM(masses)
            disp = np.linalg.norm(np.array([p2[0],0,p2[2]])-np.array([p1[0],0,p1[2]]))
            speed = disp/increment
            rate(5)
            update_scene(masses, springs,m_obs, s_obs, COM,v,speed)
        else:
            interact_fast(springs,masses,t,increment, mu_static,mu_kinetic,floor)
    if plot_energies:
        plot_energies(energies)


@njit()
def get_COM(masses):
    M = len(masses) * 0.1
    masses[:,0,0]
    COMx = np.sum(masses[:,0,0]*0.1)/M
    COMy = np.sum(masses[:,0,1]*0.1)/M
    COMz = np.sum(masses[:,0,2]*0.1)/M
    return(np.array([COMx, COMy, COMz]))

def eval_springs(masses,springs,spring_constants,final_T=4,increment=0.0002, render = False):
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
        rehome(masses) #reset current state of the masses
        return(np.linalg.norm(np.array([p2[0],0,p2[2]])-np.array([p1[0],0,p1[2]])))


def rehome(masses):
    for m in masses:
        pos0 = m[4]
        m[0] = pos0
        m[1] = np.array([0.0, 0.0, 0.0])
        m[2] = np.array([0.0, 0.0, 0.0])
        m[3] = np.array([0.0, 0.0, 0.0])

if __name__ == "__main__":
    masses = initialize_masses()
    springs = initialize_springs(masses)
    run_simulation(masses, springs,6,0.0002,0.9,0.8,floor=-10)