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
#include<cmath>

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
    
    double rest_length;
    double spring_constant;
    map <char,double> constants;
    //constants['a']=rest_length; // { {"a",rest_length},{"b",0},{"c",0},{"k",spring_constant} };
    
    constants.insert(pair<char, double>("a", rest_length));
    
    Spring(int id[], double l0, double k){
        idcs[0] = id[0];
        idcs[1] = id[1];
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
        springs = initialize_springs();
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
                // cout<<p[0]<<","<<p[1]<<","<<p[2] <<endl;
                output.push_back(Mass(0.1,p,v,a,f));
            }
        }
    }
    return output;
}

vector<Spring> Simulation::initialize_springs(){
    vector<Spring> output;
    for (int mass_idx1 =0; mass_idx1<masses.size();mass_idx1++) {
        for (int mass_idx2 =mass_idx1+1; mass_idx2<masses.size();mass_idx2++) {
            cout<<mass_idx1<<","<<mass_idx2<<endl;
            double m1_p[3];
            m1_p[0]=masses[mass_idx1].position[0];
            m1_p[1]=masses[mass_idx1].position[1];
            m1_p[2]=masses[mass_idx1].position[2];
            double m2_p[3];
            m2_p[0]=masses[mass_idx2].position[0];
            m2_p[1]=masses[mass_idx2].position[1];
            m2_p[2]=masses[mass_idx2].position[2];

            double length=get_vect_length(m1_p,m2_p);
            int indcs[2] = {mass_idx1,mass_idx2};

            output.push_back(Spring(indcs,length,spring_k);

        }

    }
}


double get_vect_length(double v1[], double v2[]){
    double res = sqrt(pow(v2[0]-v1[0],2) + pow(v2[1]-v1[1],2) + pow(v2[2]-v1[2],2));
    return res;
}



int main()
{
    Simulation sim(0.2,5,10000, 0.8,0.6);
    
    for (i=0,i<sim.springs.size();i++){
        cout<<"length"<<sim.springs[i].length<<endl;
        cout<<"mass indices ["<<sim.springs[i].idcs[0]<<","<<sim.springs[i].idcs[1]<<"]"<<endl;
    }

    

}