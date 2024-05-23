#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>
#include "robot.hpp"


using namespace Eigen;
using namespace std;


class Simulation {
    float dt;
    int num_robots;
    float mu_k;
    float mu_s;
    float t;
    int masses_per_rob;
    float floor_pos;
    float ground_k; 
    float damping;
    bool cuda;
    std::vector<Robot> robots; // Vector to store Robot objects
    

public:
    Simulation (float, int, float, float, int, float, float, float, bool);
    Simulation();
    static Simulation def_const(); //default constructor
    void step();
    void run_simulation(float);  
    float get_sim_t();
    std::vector<MatrixXd> get_sim_masses();
    std::vector<MatrixXd> get_sim_springs();

};