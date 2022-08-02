# Toy Model Model M&uuml;ller Potential  
  
-   **unbias**: short unbias simulations. Then biased simulations are performed biasing along the first eigenfunctions of either TICA or Deep-TICA operator. 
-   **benchmark**: benchmark simulations to compare estimate eigenfunctions versus the more realistic ones.  
-   **analysis**: inside this folder there are other two folders: **data** and **test_corr**. In the former one are present the results relative to the standard deviation and the mean of the leading eigenfunctions obtained training different NN. In the latter the method used to estimate the time correlation function is compared to the pandas correlation functions method (Pearson correlation function).  
-   **On the fly**: application of Deep-TICA method in an iterative manner, which means "On the fly".   