# On the Fly Estimation of CVs  
The idea is to obtain cvs on the fly  
  
We cannot use all the cumulative data, because the unbias and bias distribution of the data are different.  
Then we have to estimate the correlation function just from few data obtained by "short" simulations (5 ns)  
  
The first attempt is with TICA, a more controllable way to estimate this cvs

# Mixed  
  
The first part of the iterations are made by the use of TICA. Which is faster and more explorative even if less precise. 
The last part will consist in the use of Deep-TICA on simulations that exhibts transitions.  
How to check a transition: 
I tried different methods, the first one the biased partition function. Basically the acceleration factor. But it does not work and depends on the convergence of the bias, which is very slow for very complicated systems. 
Then I try the integral of the PSD. But even in this case it depends too much on the found cv. One must compute the new cv for all the old ones and check its PSD integral.

My proposal now is the following: the other quantity able to identify a transition is the time correlation lenght, even if evaluated from a biased trajectory (I remember that I am going to use a low biasfactor to avoid exploration of unphysical regions). I stop the iteration when the correlation lenghts stop to increase over a certain percentage. 