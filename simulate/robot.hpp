#ifndef ROBOT_H
#define ROBOT_H
#include <eigen3/Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::Matrix;
using Eigen::Dynamic;

class Robot {
    float floor_pos;
    float dt;
    float mu_s;
    float mu_k; 
    float ground_k; 
    float damping;
    void initialize_masses(int m);
    void initialize_springs();

public:
    Robot();
    Robot(int, float, float, float, float, float, float, bool);
    MatrixXd masses;
    MatrixXd springs;
    int total_masses;
    int total_springs;
    void print_masses();
    void print_springs();
    void compute_spring_forces(float time);
    void force_integration();
    float get_spring_length(int spring_idx);
    void compute_spring_forces_fast(float time);
    bool cuda;

};

#endif