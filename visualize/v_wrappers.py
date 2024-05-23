'''wrappers to display simulation data using vpython'''

import time
from vpython import *
import pickle
import sys
import os
sys.path.append(os.getcwd())
from simulate.fast_utils import get_COM_fast, get_spring_l_fast
import pysim
import numpy as np


def initialize_scene(z):
    # walls
    scene = canvas(width=1200,height=750,background=color.black)
    v = wtext(text="0 m/s")  
    floor = box(pos=vector(0.5,z-0.06,0), height=0.02, length=100, width=100,color=color.white)
    for x in range(-50,53,3):
        gridx = curve(pos=[vector(x,z-0.06,50),vector(x,z-0.06,-50)],color=color.black)
    for y in range(-50,53,3):
        gridx = curve(pos=[vector(50,z-0.06,y),vector(-50,z-0.06,y)],color=color.black)

def plot_robot_masses(robot_masses, mass_plots=None, COM_plot=None, offset_x=0,offset_z=0):
    if mass_plots is None:
        mass_plots = []
        for mass in robot_masses:
            m_temp = sphere(pos=vector(mass[0]+offset_x,mass[1],mass[2]+offset_z), 
                            radius=0.06, 
                            color=color.green)
            mass_plots.append(m_temp)
        COM_pos = get_COM_fast(robot_masses)
        COM_plots = sphere(pos=vector(COM_pos[0],COM_pos[1],COM_pos[2]), radius=0.06, color=color.yellow)
    else:
        for idx,m in enumerate(mass_plots):
            m.pos = vector(robot_masses[idx][0], robot_masses[idx][1], robot_masses[idx][2])
        COM_pos = get_COM_fast(robot_masses)
        COM_plot.pos = vector(COM_pos[0],COM_pos[1],COM_pos[2])
    return(mass_plots, COM_plots)

def plot_robot_springs(robot_springs, robot_masses, spring_plots=None, offset_x=0, offset_y=0):
    if spring_plots is None:
        spring_plots = []
        for spring in robot_springs:
            m1_pos = robot_masses[int(spring[0])][0:3]
            m2_pos = robot_masses[int(spring[1])][0:3]
            s_temp = cylinder(pos=vector(m1_pos[0], m1_pos[1], m1_pos[2]), 
                            axis=vector(m2_pos[0]-m1_pos[0], m2_pos[1]-m1_pos[1], m2_pos[2]-m1_pos[2]), 
                            length=get_spring_l_fast(robot_masses,spring),
                            color=color.red, 
                            radius=0.01)
            spring_plots.append(s_temp)
    else:
        for idx,s in enumerate(spring_plots):
            m1_pos = robot_masses[int(robot_springs[idx][0])][0:3]
            m2_pos = robot_masses[int(robot_springs[idx][1])][0:3]
            s.pos = vector(m1_pos[0], m1_pos[1], m1_pos[2])
            s.axis = vector(m2_pos[0]-m1_pos[0], m2_pos[1]-m1_pos[1], m2_pos[2]-m1_pos[2])
            s.length= get_spring_l_fast(robot_masses,robot_springs[idx])
    return(spring_plots)

def update_scene(masses, springs, m_plots, s_plots, COM,v,speed):
    for idx,m in enumerate(m_plots):
        m.pos = vector(masses[idx][0][0], masses[idx][0][1], masses[idx][0][2])
    
    for idx,s in enumerate(s_plots):
        m1_pos = masses[int(springs[idx][0][0])][0]
        m2_pos = masses[int(springs[idx][0][1])][0]
        s.pos = vector(m1_pos[0], m1_pos[1], m1_pos[2])
        s.axis = vector(m2_pos[0]-m1_pos[0], m2_pos[1]-m1_pos[1], m2_pos[2]-m1_pos[2])
        s.length= get_spring_l_fast(masses,springs[idx])
    COM_pos = get_COM_fast(masses)
    COM.pos = vector(COM_pos[0],COM_pos[1],COM_pos[2])
    v.text = str(str(round(speed,3))+" m/s")

def update_viz(sim_masses, sim_springs, mass_plots, spring_plots, COM_plots, velocity_plots, speed_plots):
    if len(sim_masses) == 1:
        mass_plots, COM_plots = plot_robot_masses(sim_masses[0], mass_plots, COM_plots)
        spring_plots = plot_robot_springs(sim_springs[0], sim_masses[0], spring_plots)
    else:
        # TODO: generalize for N robots (need offsets so the robots aren't plotted on top of each other)
        pass


def visualize_simulation(path_to_sim,floor):
    time.sleep(6)
    with open(path_to_sim, 'rb') as handle:
        sim_logger = pickle.load(handle)
    
    timestamps = sorted(list(sim_logger.keys()))
    m_obs, s_obs, COM,v = initialize_scene(sim_logger[timestamps[0]]['masses'], 
                                           sim_logger[timestamps[0]]['springs'],
                                           z=floor-0.06)
    for i,t in enumerate(timestamps[1:]):
        print(f'sim time: {t:.2f}')
        rate(50)
        update_scene(sim_logger[timestamps[i]]['masses'], 
                     sim_logger[timestamps[i]]['springs'], 
                     m_obs, 
                     s_obs, 
                     COM,
                     v,
                     sim_logger[timestamps[i]]['speed'])

if __name__ == "__main__":
    
    simy = pysim.Simulation()

    # set the scene
    initialize_scene(0)
    m_ob, com_ob = plot_robot_masses(simy.get_sim_masses()[0])
    s_ob = plot_robot_springs(simy.get_sim_springs()[0], simy.get_sim_masses()[0])
    sim_length = 1
    time_step = 0.1
    for i,t in enumerate(np.arange(0, sim_length, time_step)):
        rate(50)

    # # if i % log_k == 0:
    # #     # p1 = self.get_COM(self.masses)
    # #     sim.step(t)
    # #     # p2 = self.get_COM(self.masses)
    # #     # displacement = np.linalg.norm(np.array([p2[0],0,p2[2]])-np.array([p1[0],0,p1[2]]))
    # #     # speed = displacement/time_step
    # #     # self.sim_log[t] = {
    # #     #     'masses': np.copy(self.masses),
    # #     #     'springs': np.copy(self.springs),
    # #     #     'speed': speed
    # #     # }
    # # else:
    # #     sim.step(t)
    #     sim.step()
    #     print(f'SIM TIME: {sim.get_sim_t()}')
    #     print(len(sim.get_sim_masses()))
    #     # print(sim.get_sim_springs())
    #     input()

    