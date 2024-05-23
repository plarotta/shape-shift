// #include <Eigen/Dense>
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
#include <pybind11/eigen.h>
#include <pybind11/stl.h>


namespace py = pybind11;

using namespace Eigen;
using namespace std;

Simulation::Simulation (float a, int b, float c, float d, float e) {
    dt = a;
    num_robots = b;
    mu_k = c;
    mu_s = d;
    T = e;
    t = 0.0;

    // Resize the vector to hold the desired number of robots
    robots.resize(num_robots);

    for (int i = 0; i < num_robots; i++){
        // use default constructor for all the robots for now
        robots[i] = Robot();
    }
}
Simulation::Simulation() {
    dt = 0.05;
    num_robots = 1;
    mu_k = 0.7;
    mu_s = 0.8;
    T = 1;
    t = 0.0;

    // Resize the vector to hold the desired number of robots
    robots.resize(num_robots);

    for (int i = 0; i < num_robots; i++){
        // use default constructor for all the robots for now
        robots[i] = Robot();
    }
}

void Simulation::step() {
    for (int r_idx = 0; r_idx < num_robots; r_idx ++){
            robots[r_idx].compute_spring_forces(t);
            robots[r_idx].force_integration();
    }
    t += dt;
}

float Simulation::get_sim_t() {
    return t;
}

void Simulation::run_simulation() {
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; t < T; i++){
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(current_time - start_time).count();
        cout << "SIM TIME: " << t << " | " << "REAL TIME: " << std::fixed << std::setprecision(2) << elapsed_time << endl;
        // update each robot in the simulation
        step();
        t += dt;
    }
}

std::vector<MatrixXd> Simulation::get_sim_masses() {
    std::vector<MatrixXd> sim_masses;
    sim_masses.resize(num_robots);
    for (int r_idx = 0; r_idx < num_robots; r_idx ++){
            sim_masses[r_idx] = robots[r_idx].masses;
    }
    return sim_masses;
}

std::vector<MatrixXd> Simulation::get_sim_springs() {
    std::vector<MatrixXd> sim_springs;
    sim_springs.resize(num_robots);
    for (int r_idx = 0; r_idx < num_robots; r_idx ++){
            sim_springs[r_idx] = robots[r_idx].springs;
    }
    return sim_springs;
}

PYBIND11_MODULE(pysim, m) {
    py::class_<Simulation>(m, "Simulation")
        .def(py::init())
        .def("step", &Simulation::step)
        .def("run_simulation", &Simulation::run_simulation)
        .def("get_sim_t", &Simulation::get_sim_t)
        .def("get_sim_masses", &Simulation::get_sim_masses)
        .def("get_sim_springs", &Simulation::get_sim_springs);
    }

