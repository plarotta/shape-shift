// C++ implementation of the computationally-intensive main.py functions

// TODO
// 1. Mass & Spring classes [done]
// 2. Relevant Simulation class methods
//      a. __init__ [done]
//      b. initialize_springs & initialize_masses 
//      c. interact ***important***
//      d. integrate ***important***
//      e. get_spring_l
//      f. get_COM

#include <iostream>
#include <vector>
#include <map>

using namespace std;


class Mass {
    public:

    double mass;
    double position[3];
    double velocity[3];
    double acceleration[3];
    vector<vector<double> > f_ext; 

    Mass(double m, double pos[], double v[], double acc[], vector<vector<double> > f){
        mass = m;
        position[0] = pos[0];
        position[1] = pos[1];
        position[2] = pos[2];

        velocity[0] = v[0];
        velocity[1] = v[1];
        velocity[2] = v[2];
        acceleration[0] = acc[0];
        acceleration[1] = acc[1];
        acceleration[2] = acc[2];
        f_ext = f;
    }

};

class Spring{
    public:

    int idcs[2];
    map <char,double> constants;
    double rest_length;
    double spring_constant;

    Spring(int id[], map <char,double> c, double l0, double k){
        idcs[0] = id[0];
        constants = c;
        rest_length = l0;
        spring_constant = k;
    }

};

class Simulation{
    public:
    double increment;
    double final_T;
    double spring_k;
    vector<Mass> masses;
    int springs;
    double mu_static;
    double mu_kinetic;

    vector<Mass> initialize_masses();
    vector<Spring> initialize_springs();

    Simulation(double dt, double T, double k, double mu_s, double mu_k){
        increment = dt;
        final_T = T;
        spring_k = k;
        masses = initialize_masses();
        springs = 2; //initialize_springs();
        mu_static = mu_s;
        mu_kinetic = mu_k;
    }   
};

vector<Mass> Simulation::initialize_masses(){
    vector<Mass> output;
    double v[] = {0,0,0};
    double a[] = {0,0,0};
    vector<vector<double> > f;

    for (int x = 0; x < 2; x++){
        for (int y = 0; y < 2; y++){
            for (int z = 0; z < 2; z++){
                double p[] = {x,y,z};
                output.push_back(Mass(0.1,p,v,a,f));
            }
        }
    }
    return(output);
}

// vector<Spring> Simulation::initialize_springs(){
//     vector<Spring> a = 
// }






int main()
{
    Simulation sim(0.2,5,10000, 0.8,0.6);
    vector<Mass> vi = sim.masses;
    for (int i : vi){ 
        cout << vi[i].position[0] <<","<< vi[i].position[1]<< ","<< vi[i].position[2] <<endl;
    }
}