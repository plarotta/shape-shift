#include "robot.hpp"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cmath>
#include <chrono>

#define GRAVITY 9.81;
#define MASS_WEIGHT 0.1;

using Eigen::MatrixXd;
using Eigen::Vector3f;
using Eigen::Map;
using std::cout;
using std::endl;

void Robot::print_masses() {
    cout << masses << endl;
    
    return;
}

void Robot::print_springs() {
    cout << springs << endl;
    return;
}

void Robot::initialize_masses(int mass_num) {
    // row idx == mass idx, col idx == attribute. attributes are pos, vel, acc, force, init pos. each 3d
    total_masses = mass_num;
    masses = MatrixXd::Zero(mass_num, 15); // Initialize with zeros

    int range = static_cast<int>(std::ceil(std::cbrt(mass_num)));

    int n = 0;
    for (int x = 0; x < range; ++x) {
        for (int y = 0; y < range; ++y) {
            for (int z = 0; z < range; ++z) {
                if (n < mass_num) {
                    masses.row(n).segment(0, 3) << static_cast<float>(x), static_cast<float>(y), static_cast<float>(z);
                    masses.row(n).segment(12, 3) << static_cast<float>(x), static_cast<float>(y), static_cast<float>(z);
                }
                else{
                    break;
                }
                ++n;
            }
        }
    }
}

void Robot::initialize_springs() {
    // row idx == spring idx. col idx == attribute. attributes are m1 idx, m2 idx, and 4 spring constants
    int spring_idx = 0;
    springs = MatrixXd::Zero(total_masses*(total_masses-1)/2, 6);
    // cout<<springs<<endl;
    for (int i = 0; i < total_masses; ++i){
        for (int j = i+1; j < total_masses; ++j){
            Vector3f diff(masses.row(j).segment(0,3).cast<float>()-masses.row(i).segment(0,3).cast<float>());
            springs.row(spring_idx).segment(0, 6) << i, j, diff.norm(), 0, 0, 10000;
            spring_idx += 1;
        }
    }
    total_springs = spring_idx;
}

float Robot::get_spring_length(int s) {
    int m1_idx = springs(s,0);
    int m2_idx = springs(s,1);
    Vector3f diff(masses.row(m2_idx).segment(0,3).cast<float>()-masses.row(m1_idx).segment(0,3).cast<float>());
    return diff.norm();
}

void Robot::compute_spring_forces(float t) {
    #pragma omp parallel for
    for (int i = 0; i < total_springs; i ++) {
        float current_length = get_spring_length(i);
        int m1_idx = springs(i,0);
        int m2_idx = springs(i,1);
        Vector3f current_l_vect( masses.row(m2_idx).segment(0,3).cast<float>()-masses.row(m1_idx).segment(0,3).cast<float>());
        float rest_length = springs(i,2) + springs(i,3) * sin(4*t + springs(i,4));
        float spring_f_mag = springs(i,5) * (current_length - rest_length);
        float spring_f_dir = spring_f_mag/current_l_vect.norm();
        Vector3f spring_f_full = current_l_vect * spring_f_dir;

        // update net force on masses attached to spring
        #pragma omp critical
        {
            masses(m1_idx,9) += spring_f_full(0);
            masses(m1_idx,10) += spring_f_full(1);
            masses(m1_idx,11) += spring_f_full(2);

            masses(m2_idx,9) -= spring_f_full(0);
            masses(m2_idx,10) -= spring_f_full(1);
            masses(m2_idx,11) -= spring_f_full(2);
        }   
    }
}

void Robot::force_integration() {
    #pragma omp parallel for
    for (int i = 0; i < total_masses; i++) {
        masses(i,10) -= GRAVITY;
        if (masses(i, 1) < floor_pos) {
            Vector3f net_force = masses.row(i).segment(9,3).cast<float>();
            Vector3f prop_force(net_force(0), 0.0, net_force(2));
            float prop_force_norm = prop_force.norm();
            float normal_force = net_force(1);

            if (prop_force_norm < abs(normal_force * mu_s)){
                masses(i, 9) -=  prop_force(0);
                masses(i, 11) -=  prop_force(2);
            }
            else{
                Vector3f dir_frict_f_mag = mu_k *normal_force/prop_force_norm * prop_force;
                masses(i,9) -= dir_frict_f_mag(0);
                masses(i,11) -= dir_frict_f_mag(2);
            }
            float ground_rxn_mag = ground_k * abs(masses(i,1)-floor_pos);
            masses(i,10) += ground_rxn_mag;
        }

        // simple euler integration
        // accel
        masses(i,6) = masses(i,9)/MASS_WEIGHT;
        masses(i,7) = masses(i,10)/MASS_WEIGHT;
        masses(i,8) = masses(i,11)/MASS_WEIGHT;

        // vel
        masses(i, 3) += masses(i,6) * dt;
        masses(i, 4) += masses(i,7) * dt;
        masses(i, 5) += masses(i,8) * dt;
        
        masses(i, 3) += damping; // TODO: make default value for damping 1
        masses(i, 4) += damping;
        masses(i, 5) += damping;

        // pos
        masses(i, 0) += masses(i,3) * dt;
        masses(i, 1) += masses(i,4) * dt;
        masses(i, 2) += masses(i,5) * dt;

        // clear mass forces
        masses(i, 9) = 0;
        masses(i, 10) = 0;
        masses(i, 11) = 0;
    }
}

Robot::Robot(int a, float c, float d, float e, float f, float g, float h, bool j) {
    total_masses = a;
    floor_pos = c;
    dt = d;
    mu_s = e;
    mu_k = f;
    ground_k = g;
    damping = h;
    cuda = j;
    initialize_masses(total_masses);
    initialize_springs();
}

// default constructor
Robot::Robot() {
    total_masses = 10;
    floor_pos = -0.01;
    dt = 0.01;
    mu_s = 0.9;
    mu_k = 0.7;
    ground_k = 10000;
    damping = 0.996;
    cuda = false;
    initialize_masses(total_masses);
    initialize_springs();
}


