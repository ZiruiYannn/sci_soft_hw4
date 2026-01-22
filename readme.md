**Use Src/Makefile to generate the executable.**

Practical Information:

**simulation3.cpp**: generate the simulation results of predictions of the observations
**siqrd.hpp**: contains the function and Jacobian for siqrd
**simple_ode**: contains the function and Jacobian for ODEs
**ivp_solver**: contains the class ivp. The class ivp has member functions simu_forward, simu_backward and simu_heun which do the simulations.
**io_siqrd**: contains all the read and write functions I need. 
**bfgs**: contains the class bfgs. The class bfgs has member functions BFGS_forward, BFGS_backward and BFGS_heun.
