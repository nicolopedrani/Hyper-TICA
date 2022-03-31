### Theoretical Framework, TICA time-lagged independent component analysis  

From a *Molecular Dynamics* simulation we obtain a set of points $\left\{ \vec {x}(t_i), \vec {p}(t_i) \right\} _{i=1,N}$ which define a trajectory in the Phase Space $\Omega$ of the system, where $N$ is the total number of simulation steps.  
These points are distributed according to a certain probability distribution $p_t(\vec x)$. If the simulation is sufficiently long and the system is ergodic then (in the canonical ensemble NPT) $p(\vec x) = \dfrac{e^{-\beta U(\vec x)}}{Z}$ is the Boltzmann one. Strictly speaking we can say that during the simulation the distribution probability evolves in time and the stationary distribution $\mu (\vec x)$ is the Boltzmann distribution: $\lim_{\tau \rightarrow \infty} p_{t+\tau}(\vec x)  = \mu (\vec x)$  

#### Transfer operator
It is possible to formalize this statement by the definition of the transfer operator $\mathcal{T}(\tau)$: given a $p_t(\vec x)$ then $p_{t+\tau}(\vec x) = \mathcal{T}(\tau) \circ p_t(\vec x)$  
Denoting with $\psi_i(\vec x)$ its eigenfunctions, and with $\lambda_i(\tau) = e^{-\frac{\tau}{t_i}}$ its eigenvalues it is possibile to rewrite the expression above as follows: $$ p_{t+\tau}(\vec x) = \mathcal{T}(\tau) \circ p_t(\vec x) = \sum_i e^{-\frac{\tau}{t_i}} \langle p_t(\vec x) \psi_i(\vec x) \rangle \psi_i(\vec x) $$
As $\tau \rightarrow \infty$ all the contributes vanish but the one relative to $t_i \rightarrow \infty$ which is exactly the Boltzmann distribution $\mu(\vec x)$, the fixed point of $\mathcal{T}(\tau)$, $\mathcal{T}(\tau) \circ \mu(\vec x) = \mu (\vec x)$  
  
#### TICA Analysis  
The aim of TICA analysis is to diagonalize $\mathcal{T}(\tau)$ and express its action on a probability distribution through its eigenfunctions and eigenvalues. In this way by applying infinite times $\mathcal{T}(\tau)$ to $p_t(\vec x)$ one finally obtains the form of $\mu (\vec x)$:
$$ \lim_{k \rightarrow \infty} \left( \prod^k \mathcal{T}(\tau) \right) \circ p_t(\vec x) =\lim_{k \rightarrow \infty} \mathcal{T}(k \tau) p_t(\vec x) = \lim_{k \rightarrow \infty} p_{t+k \tau} (\vec x) = \lim_{k \rightarrow \infty} \sum_i e^{-\frac{k \tau}{t_i}} \langle p_t(\vec x) \psi_i(\vec x) \rangle \psi_i(\vec x) = \mu (\vec x) $$  
Where $ \langle p_t(\vec x) \psi_i(\vec x) \rangle = \int d \vec x p_t(\vec x) \psi_i(\vec x) e^{-\beta U(\vec x)} $  
In order to diagonalize this operator we must consider a subset of the all eigenfunctions, such that we can work in a finite dimensional vector space. Suppose we are interested in the first $m$ slow modes of the system. With this choice we are assuming that all the other slow modes are those with a timescale much smaller than $\tau$: $e^{-\frac{\tau}{t_i}} \rightarrow 0$, for $i>m$.  
Then it is clear that $\tau$ defines which are the slow modes we are interested in. But in principle we do not know the slow modes of the system and their timescales.  

#### Linear approximation and TICA
How can we guess the $\psi_i(\vec x)$ ? 
To answer this question we first recall what TICA does:   
It uses a linear transformation to map an original order parameters $d_j(\vec x)$ set to a new set of order parameters $\psi_i(\vec x)$, the *Independent component*. $\psi_i(\vec x)$ must fullfill two properties:  
1.  they are uncorrelated. This means that $\langle \psi_i(\vec x_t) \psi_j(\vec x_{t} \rangle = \delta_{i,j}$
2.  Their autocovariances at a fixed lag time $\tau$ are maximal  
If $\psi_i(\vec x)$ are the $\mathcal{T}(\tau)$ eigenfunctions, then $$\langle \psi_i(\vec x_t) \psi_i(\vec x_{t+\tau}) \rangle = \lambda_i$$  
  
We consider a basis set formed by the so called *descriptors* $d_j(\vec x)$, $\left\{ d_j(\vec x) \right\}_{j=1,n}$. Then we assume that each $\psi_i(\vec x)$ can be written as a linear combination of $\{d_j(\vec x)\}$: $\psi_i(\vec x) = \sum_j^n b_{ij} d_j(\vec x)$  
With this approximation the problem of finding $\psi_i(\vec x)$ (of maximizing $\lambda_i$) becomes a general eigenvalue problem to find $b_{ij}$: $$C^d(\tau) \cdot \vec b_i = \lambda_i C^d(0) \cdot \vec b_i$$
where $C_{ij}^d(\tau) = \langle d_i(\vec x_t) d_j(\vec x_{t+\tau})  \rangle$ and $C_{ij}^d(0) = \langle d_i(\vec x_t) d_j(\vec x_{t})  \rangle$  
  
Condition 1. requires that there is not overlap between $\lambda_i$ and $\lambda_j$, and that $C^d(\tau)$ is symmetric. Obviously this condition it is not satisfied from the data, which are obtained from an finite simulation, so we must enforce this condition.  
