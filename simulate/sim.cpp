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

Simulation::Simulation (float a, int b, float c, float d, int e, float f, float g, float h, bool s) {
    dt = a;
    num_robots = b;
    mu_s = c;
    mu_k = d;
    t = 0.0;
    masses_per_rob = e;
    floor_pos = f;
    ground_k = g; 
    damping = h;
    cuda = s; // TODO: add CUDA paralellization. someday...

    // Resize the vector to hold the desired number of robots
    robots.resize(num_robots);

    for (int i = 0; i < num_robots; i++){
        robots[i] = Robot(masses_per_rob, floor_pos, dt, mu_s, mu_k, ground_k, damping, cuda) ;
    }
}

Simulation::Simulation() {
    dt = 0.001;
    num_robots = 1;
    mu_k = 0.7;
    mu_s = 0.9;
    t = 0.0;
    masses_per_rob = 8;
    floor_pos = -0.01;
    ground_k = 10000; 
    damping = 0.996;
    cuda = false; // TODO: CUDA parellization

    // Resize the vector to hold the desired number of robots
    robots.resize(num_robots);

    for (int i = 0; i < num_robots; i++){
        // use default constructor for all the robots for now
        robots[i] = Robot(masses_per_rob, floor_pos, dt, mu_s, mu_k, ground_k, damping, cuda);
    }
}

Simulation Simulation::def_const() { return Simulation(); }

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

void Simulation::run_simulation(float T) {
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
        .def(py::init(&Simulation::def_const))
        .def(py::init([](float a, int b, float c, float d, int e, float f, float g, float h, bool s) {
            return new Simulation(a, b, c, d, e, f, g, h, s);}))
        .def("step", &Simulation::step)
        .def("run_simulation", &Simulation::run_simulation)
        .def("get_sim_t", &Simulation::get_sim_t)
        .def("get_sim_masses", &Simulation::get_sim_masses)
        .def("get_sim_springs", &Simulation::get_sim_springs);
    }

