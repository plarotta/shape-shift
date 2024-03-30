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

def visualize_simulation(masses, springs,final_T,increment, mu_static, mu_kinetic,floor,plot_energies=False, debug_mode = False):
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
            update_scene(masses, springs, m_obs, s_obs, COM,v,speed)
        else:
            interact_fast(springs,masses,t,increment, mu_static,mu_kinetic,floor)
    if plot_energies:
        plot_energies(energies)