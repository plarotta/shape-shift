# Shape Shift
A full genetic program to evolve shapes that can move.

## Description
Three modules:
1. Simulate: efficient physics simulator written in Python.
2. Visualize: takes data from a simulation and displays it using vpython.
3. Evolve: implements a basic genetic program to evolve shapes capable of translation.


*New: C++ binding*

compile with 
```g++-14 -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup $(python3 -m pybind11 --includes) simulate/sim.cpp simulate/robot.cpp -o pysim$(python3-config --extension-suffix) -fopenmp```


(remove -fopenmp to disable multithreading)
