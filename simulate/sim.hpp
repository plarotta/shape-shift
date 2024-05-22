#include <eigen3/Eigen/Dense>
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
    float T;
    float t;
    std::vector<Robot> robots; // Vector to store Robot objects

public:
    Simulation (float, int, float, float, float);
    Simulation();
    void step();
    void run_simulation();  
    float get_sim_t();
    std::vector<MatrixXd> get_sim_masses();
    std::vector<MatrixXd> get_sim_springs();

};