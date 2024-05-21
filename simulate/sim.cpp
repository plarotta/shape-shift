#include <eigen3/Eigen/Dense>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>
#include "sim.hpp"
#include "robot.hpp"
#include <vector>
#include <iomanip>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using namespace Eigen;
using namespace std;

Simulation::Simulation (float a, int b, float c, float d, float e) {
    dt = a;
    num_robots = b;
    mu_k = c;
    mu_s = d;
    T = e;

    // Resize the vector to hold the desired number of robots
    robots.resize(num_robots);

    for (int i = 0; i < num_robots; i++){
        // use default constructor for all the robots for now
        robots[i] = Robot();
    }
}
Simulation::Simulation() {
    dt = 0.05;
    num_robots = 300;
    mu_k = 0.7;
    mu_s = 0.8;
    T = 1;

    // Resize the vector to hold the desired number of robots
    robots.resize(num_robots);

    for (int i = 0; i < num_robots; i++){
        // use default constructor for all the robots for now
        robots[i] = Robot();
    }
}

void Simulation::run_simulation() {
    float sim_time = 0.0;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; sim_time < T; i++){
        
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(current_time - start_time).count();

        cout << "SIM TIME: " << sim_time << " | " << "REAL TIME: " << std::fixed << std::setprecision(2) << elapsed_time << endl;
        // update each robot in the simulation
        for (int r_idx = 0; r_idx < num_robots; r_idx ++){
            robots[r_idx].compute_spring_forces(sim_time);
            robots[r_idx].force_integration();
        }
        sim_time += dt;
    }
}


PYBIND11_MODULE(pysim, m) {
    py::class_<Simulation>(m, "Simulation")
        .def(py::init())
        .def("run_simulation", &Simulation::run_simulation);

    }

