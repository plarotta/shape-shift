import numpy as np
from numba import njit 
import scipy # this import is needed for JIT on np.linalg methods

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
  
@njit(nogil=True, fastmath=True)
def interact_fast(springs,masses,t,increment,mu_static,mu_kinetic,floor=-4):
    for s in springs:
        l = get_spring_l_fast(masses,s)
        l_vect = masses[int(s[0][1])][0] - masses[int(s[0][0])][0]  
        L0 = s[1][0] + s[1][1] * np.sin(4*t + s[1][2])
        f_k = s[1][3] * (l - L0)
        f_dir = f_k/np.sqrt(l_vect[0]**2 + l_vect[1]**2 + l_vect[2]**2)
        f_full = f_dir * l_vect
        masses[int(s[0][0])][3] = masses[int(s[0][0])][3] + f_full
        masses[int(s[0][1])][3] = masses[int(s[0][1])][3] - f_full
    for m in masses:   
        g = 9.81
        m[3][1] = m[3][1] - g #gravity
        if m[0][1] < floor:
            f_net = m[3]
            Fp = np.array([f_net[0], 0.0, f_net[2]])
            Fp_norm = np.linalg.norm(Fp)
            Fn_norm = f_net[1]
            
            if Fp_norm < abs(Fn_norm * mu_static): #friction
                m[3] = m[3] - Fp               
            else:
                dirFn = mu_kinetic*Fn_norm*Fp/Fp_norm #friction
                m[3] = m[3] - dirFn

            if m[0][1] < floor: #floor reaction force
                ry = 10000*(abs(m[0][1] - floor))
                m[3][1] = m[3][1] + ry
        integrate_fast(m,increment)

@njit()
def integrate_fast(mass, increment):
    mass[2] = mass[3]/0.1
    mass[1] = mass[1] + mass[2]*increment
    mass[1] = mass[1] * 0.996
    mass[0] = mass[0] + mass[1]*increment
    mass[3] = np.array([0.0,0.0,0.0])

@njit()
def get_spring_l_fast(masses, spring):
    m1_idx1 = spring[0]
    m2_idx2 = spring[1]
    m1_p = masses[int(m1_idx1)][0:3]
    m2_p = masses[int(m2_idx2)][0:3]
    diff = m2_p-m1_p
    return(np.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2))

@njit()
def get_COM_fast(masses):
    M = len(masses) * 0.1
    COMx = np.sum(masses[:,0]*0.1)/M
    COMy = np.sum(masses[:,1]*0.1)/M
    COMz = np.sum(masses[:,2]*0.1)/M
    return(np.array([COMx, COMy, COMz]))

