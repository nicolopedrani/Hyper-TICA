# Hyper-TICA
Hyper TICA Phd Project  

## Project Idea
From theory we know that the TICA method is able to estimate (with a certain precision) the leading slow modes of a complex system.  
If the simulation is shorter than the timescale associated to a transition between different metastable states, then the system remains stuck in the starting macroscopic basin. The TICA method is a data driven approach, hence in this scenario it will be not able to detect the slow modes associated to the macroscopic transitions, but rather it will approximate the *fast* modes of the system (slow modes inside the starting basin) relative to the starting basin.  
We assume that accelerating these *fast* modes the system is able to explore wider the local minima and eventually escape from it in a finite amount of time. When a transition occur the hierarchy of the slow modes changes, and the slowest motion becomes that one associated to the macroscopic transition.  
In pratice we apply in a iterative manner the TICA method, every time to the samples obtained from the previous simulation.  
**Problem**: at every new iteration we apply to the system a different bias potential, which adds an external force in the direction of the new found cvs from the previous simulation. And at each iteration we obtain a new set of points distributed according to a new boltzmann distribution, different from the unbiased one. This could lead to issues because we want to estimate the1 *unbiased slow modes* of the system, so in principle we must reweight the samples.  
From a biased simulation it is always possible to recover unbiased estimate of static properties, and only rarely it is possible to recover the dynamic properties of the system such as the time correlation functions. Unfortunately TICA method requires the computation of these dynamic properties, in particular it requires the evalution of the time correlation function of the descriptors. So far we have used the algorithm implemented here [https://doi.org/10.1021/acs.jctc.8b00231], which exploits the concept of rescaled time introduced by Voter. When Voter first introduced this quantity he assumed certain hypothesis:  
-   deposited static positive bias
-   bias potential deposited only in the starting basin, and not on the transition state and further.  
These conditions are not met in our scenario, and in fact the algorithm can lead to a wrong estimation of the time correlation functions.  
**What I have done**: Rather than reweight a dynamic property I decide to not reweight at all. Obviously this is not a correct approach a priori because it can lead to artefacts. My strategy is the following:  
-   collect data from previous simulation  
-   TICA (Deep-TICA) Analysis without reweighting
-   OPES method to add the external bias, with a well-tempered distribution as a target. In order to avoid exploration of unphysical regions (less likely to be observed in a unbiased simulation) I set the bias factor as low as possible in order to obtain a balance between fast exploration and exploration of physically important regions only.  
This approach obviously requires more Molecular Dynamics steps to be performed for each iteration. The main issue is that it is not controllable and it leads to artefact easily.  
**Perspective**:  
-   find a more robust method for the evaluation of the time correlation function
-   change iterative approach. For example one can combine the autoencoders with the TICA method.    
  
### Folder
There will be different folders:  
-   **1D_Model**: One dimensional model to test Deep-TICA performance
-   **Potential2D**: bi-dimensional Toy Model potential
-   **Toy Model**: M&uuml;ller Potential bi-dimensional Toy Model  
-   **Alanine**: simulations of alanine dipeptide with gromacs
-   **Chignolin**: simulations of chignolin with gromacs
-   **Time Lagged Dataset Cpp**: performance test using Armadillo c++ library to compute the time lagged dataset  