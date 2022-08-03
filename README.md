# Hyper-TICA
Hyper TICA Phd Project  

## Project Idea
From theory we know that the TICA method is able to estimate (with a certain precision) the leading slow modes of a complex system.  
If the simulation is shorter than the timescale associated to a transition between different metastable states, then the system remains stuck in the starting macroscopic basin. The TICA method is a data driven approach, hence in this scenario it will be not able to detect the slow modes associated to the macroscopic transitions, but rather it will approximate the *fast* modes of the system (slow modes inside the starting basin) relative to the starting basin.  
We assume that accelerating these *fast* modes the system is able to explore wider the local minima and eventually escape from it in a finite amount of time. When a transition occur the hierarchy of the slow modes changes, and the slowest motion becomes that one associated to the macroscopic transition.  
In pratice we apply in a iterative manner the TICA method, every time to the samples obtained from the previous simulation.     
  
### Folder
There will be different folders:  
-   **1D_Model**: One dimensional model to test Deep-TICA performance
-   **Potential2D**: bi-dimensional Toy Model potential
-   **Toy Model**: M&uuml;ller Potential bi-dimensional Toy Model  
-   **Alanine**: simulations of alanine dipeptide with gromacs
-   **Chignolin**: simulations of chignolin with gromacs
-   **Time Lagged Dataset Cpp**: performance test using Armadillo c++ library to compute the time lagged dataset  
  
  

