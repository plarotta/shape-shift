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
from simulate.fast_utils import get_COM_fast, interact_fast


def eval_springs(masses,springs,spring_constants,final_T=4,increment=0.0002):
    for idx in range(len(springs)):
        springs[idx][1][1] = spring_constants[idx][0]
        springs[idx][1][2] = spring_constants[idx][1]
        springs[idx][1][3] = spring_constants[idx][2]

    p1 = get_COM_fast(masses)
    for t in np.arange(0, final_T, increment):
        interact_fast(springs,masses,t,increment,0.9,0.8,floor=0)
    p2 = get_COM_fast(masses)
    rehome(masses) #reset current state of the masses
    return(np.linalg.norm(np.array([p2[0],0,p2[2]])-np.array([p1[0],0,p1[2]])))

def rehome(masses):
    for m in masses:
        pos0 = m[4]
        m[0] = pos0
        m[1] = np.array([0.0, 0.0, 0.0])
        m[2] = np.array([0.0, 0.0, 0.0])
        m[3] = np.array([0.0, 0.0, 0.0])