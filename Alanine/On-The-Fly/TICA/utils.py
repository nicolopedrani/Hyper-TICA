import numpy as np
import pandas as pd
import torch
torch.manual_seed(21)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.rcParams.update({'font.size': 15})
import matplotlib as mpl

from mlcvs.utils.data import create_time_lagged_dataset
from mlcvs.utils.io import load_dataframe
from mlcvs.tica import TICA_CV
from mlcvs.utils.fes import compute_fes

from scipy import integrate
from scipy import signal

#-- to run process from jupyter --#
import subprocess
import os
from pathlib import Path
# execute bash command in the given folder
def execute(command, folder, background=False, print_result=True):
    cmd = subprocess.run(command, cwd=folder, shell=True, capture_output = True, text=True, close_fds=background)
    if cmd.returncode == 0:
        if print_result:
            print(f'Completed: {command}')
    else:
        print(cmd.stderr)

def tica_load(beta=1.0,path="colvar.data",descriptors="^p.",reweighting=True,step=100):
    
    data = load_dataframe(path)[::step]
    
    search_values=descriptors
    search_names=descriptors
    X, names, t = data.filter(regex=search_values).values, data.filter(regex=search_names).columns.values, data['time'].values
    
    if reweighting:
        # Compute logweights for time reweighting
        logweight = data["opes.bias"].to_numpy()
        #-- the logweights are V(x,y)*beta --#
        logweight = (logweight-max(logweight))*beta
    else:
        logweight=None
        print("no weights")

    return data,logweight,t,X,names

def training(beta,path,train_parameters,tprime=None):

    data,logweight,t,X,names = tica_load(beta=beta,path=path,descriptors=train_parameters["descriptors"],
                                                    reweighting=train_parameters['reweighting'],step=train_parameters["step"])
    # DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MODEL
    model = TICA_CV(n_features=X.shape[1])
    model.to(device)

    # TRAIN
    model.fit(X, t, lag=train_parameters["lag_time"], logweights=logweight, tprime=tprime)
    
    return model,data,logweight,X,names

def Boltzmann_product(model0,model1,X,j=0,k=1,logweight=None,normed=False):

    if logweight is not None:
            weights= np.exp(logweight)
    else:
        weights=np.ones(X.shape[0])

    a = np.transpose(model0(torch.Tensor(X)).detach().cpu().numpy())[j]
    b = np.transpose(model1(torch.Tensor(X)).detach().cpu().numpy())[k]
    sumtot = np.multiply(np.multiply(a,weights),b).mean()

    if normed:
        sumcv1 = np.multiply(np.multiply(a,weights),a).mean()
        sumcv2 = np.multiply(np.multiply(b,weights),b).mean()
        return sumtot/np.sqrt(sumcv1*sumcv2)
    else:
        return sumtot

def orthogonal_cv(model,X,logweight=None,j=0,k=1):

    a = np.transpose(model(torch.Tensor(X)).detach().cpu().numpy())[j]
    b = np.transpose(model(torch.Tensor(X)).detach().cpu().numpy())[k]

    prod_ab = Boltzmann_product(model,model,X,logweight=logweight,j=j,k=k)
    prod_aa = Boltzmann_product(model,model,X,logweight=logweight,j=j,k=j) 

    c = b - prod_ab/(prod_aa) * a

    return c

#-- Fes plot with 1D and 2D estimation --#
def gridspec_fes(s,logweight,sim_parameters):
    fig = plt.figure(figsize=(10,10))
    gs = GridSpec(4, 4)#, figure=fig)

    ax_scatter = fig.add_subplot(gs[1:4,0:3])
    ax_hist_x = fig.add_subplot(gs[0,:3])
    ax_hist_y = fig.add_subplot(gs[1:,3])

    #-- 2D plot --#
    fes,grid,bounds,error = compute_fes(s, weights=np.exp(logweight),
                                        temp=sim_parameters["temp"],
                                        kbt=sim_parameters["kbt"],
                                        blocks=sim_parameters["blocks"],
                                        bandwidth=sim_parameters["bandwidth"],scale_by='range')
                                        #,plot=True, plot_max_fes=sim_parameters["plot_max_fes"], ax = ax_scatter)
    bounds = np.arange(0, 60, 5.)
    cmap = plt.cm.get_cmap('fessa',len(bounds))
    colors = list(cmap(np.arange(len(bounds))))
    cmap = mpl.colors.ListedColormap(colors[:-1], "")
    # set over-color to last color of list 
    cmap.set_over("white")
    c = ax_scatter.pcolormesh(grid[0], grid[1], fes, cmap=cmap,shading='auto',alpha=1,
        norm = mpl.colors.BoundaryNorm(bounds, ncolors=len(bounds)-1, clip=False), label="FES [Kj/mol]"
    )
    c = ax_scatter.contourf(grid[0], grid[1], fes, bounds , cmap=cmap,shading='auto',alpha=1, linewidth=10,
        norm = mpl.colors.BoundaryNorm(bounds, ncolors=len(bounds)-1, clip=False), label="FES [Kj/mol]",
    )
    #fig.colorbar(c, ax=ax_scatter,label="FES [KbT]")
    c = ax_scatter.contour(grid[0], grid[1], fes, bounds, linewidths=3,cmap="gray",linestyles="dashed",
        norm = mpl.colors.BoundaryNorm(bounds, ncolors=len(bounds)-1, clip=False), label="FES [Kj/mol]",
    )
    ax_scatter.legend(["FES [Kj/mol]"])
    c.clabel()
    ax_scatter.grid()
    ax_scatter.set_xlabel(r"$\phi$")
    ax_scatter.set_ylabel(r"$\psi$")
    #np.savetxt(folder+"fes.txt",fes,delimiter=" ")
    #np.savetxt(folder+"grid0.txt",grid[0],delimiter=" ")
    #np.savetxt(folder+"grid1.txt",grid[1],delimiter=" ")

    #-- 1D plot --#
    fes,grid,bounds,error = compute_fes(s[:,0], weights=np.exp(logweight),
                                        temp=sim_parameters["temp"],
                                        kbt=sim_parameters["kbt"],
                                        blocks=sim_parameters["blocks"],
                                        bandwidth=sim_parameters["bandwidth"],scale_by='range')
    ax_hist_x.errorbar(grid,fes,yerr=error)
    ax_hist_x.set_ylabel("FES [Kj/mol]")
    ax_hist_x.grid()
                                        #,plot=True, plot_max_fes=sim_parameters["plot_max_fes"], ax = ax_hist_y)
    fes,grid,bounds,error = compute_fes(s[:,1], weights=np.exp(logweight),
                                        temp=sim_parameters["temp"],
                                        kbt=sim_parameters["kbt"],
                                        blocks=sim_parameters["blocks"],
                                        bandwidth=sim_parameters["bandwidth"],scale_by='range')
                                        #,plot=True, plot_max_fes=sim_parameters["plot_max_fes"], ax = ax_hist_x)
    ax_hist_y.errorbar(fes,grid,xerr=error)
    ax_hist_y.set_xlabel("FES [Kj/mol]")
    ax_hist_y.grid()

    plt.tight_layout()
    plt.show()
    
#-- Unbias Autocorrelation --#
# this implementation is the same as *autocorr* method in pandas
# but with this I can choose if either normalize the result or not
def my_autocorrelation_pandaslike(x,lag,normed=True):
    
    mean = np.average(x,weights=None)
    variance = np.cov(x,aweights=None)
    N = len(x)
      
    x_lag = x[lag::]
    x_0 = x[:(N-lag)]

    if normed:
        autocorr =  np.sum( np.multiply( (x_0-mean), (x_lag-mean) ) ) / ( (N-lag) * variance ) 
    else:
        autocorr = np.sum( np.multiply( x_0, x_lag )) / (N-lag)
    
    return autocorr

#-- Velocity Autocorrelation --#
def my_autocorrelation_velocity(v,lag):

    N = len(v)
    v_lag = v[lag::]
    v_0 = v[:(N-lag)]

    autocorr = sum ([ np.dot(v_0[i],v_lag[i]) for i in range(N-lag)] ) / (N-lag)
    
    return autocorr

def create_time_lagged_dataset_cpp(input_file,path="./"):

    #-- remove old files --#
    execute("rm w_lag.txt w_t.txt x_lag.txt x_t.txt",folder=path, print_result=False)

    #-- run create_time_lagged_dataset.exe --#
    execute("./create_time_lagged_dataset.exe "+input_file,folder=path, print_result=False)

    #--load data with pandas, much faster than numpy --#
    cpp_xt = torch.Tensor(pd.read_csv(path+"x_t.txt",delimiter="\t",dtype=float,header=None).to_numpy())
    cpp_xlag = torch.Tensor(pd.read_csv(path+"x_lag.txt",delimiter="\t",dtype=float,header=None).to_numpy())
    cpp_wt = torch.Tensor( np.transpose( pd.read_csv(path+"w_t.txt",delimiter="\t",dtype=float,header=None).to_numpy() )[0] )
    cpp_wlag = torch.Tensor( np.transpose( pd.read_csv(path+"w_lag.txt",delimiter="\t",dtype=float,header=None).to_numpy() )[0] )

    #-- create TensorDataset for training --#
    data = cpp_xt,cpp_xlag,cpp_wt,cpp_wlag
    dataset = torch.utils.data.TensorDataset(*data)

    return dataset

#-- My Autocorrelation --#
# one descriptor at once 
def my_autocorrelation_python(x,lag,weight=None,time=None):
   
    N = len(x)
    if weight is None:
        weight = np.ones(N)
    if time is None:
        time = np.arange(0,N)

    data = create_time_lagged_dataset(x, t = time, lag_time = lag, logweights = np.log(weight))
    x_t,x_lag,w_t,w_lag = np.array(data[:][0]),np.array(data[:][1]),np.array(data[:][2]),np.array(data[:][3])
    Nw = np.sum(w_t)
    mean = np.average(x_t,weights=w_t)
    variance = np.cov(x_t,aweights=w_t)
    autocorr = np.sum( np.multiply( np.multiply( (x_t-mean), (x_lag-mean) ), w_lag ) ) / (Nw*variance)

    return autocorr

#-- index tells me which is the descriptor to analyze --#
#-- one descriptor at once --#
def my_autocorrelation_cpp(inputFile,lag,index,path="./",weight=True):
    
    #-- modify lag time in input file --#
    execute("sed -i 's/lag.*/lag = "+str(lag)+"/g' "+inputFile,folder=path, print_result=False)

    #-- modify if_weight in input file --#
    if not weight:
        execute("sed -i 's/if_weights.*/if_weights = 0/g' "+inputFile,folder=path, print_result=False)
    else:
        execute("sed -i 's/if_weights.*/if_weights = 1/g' "+inputFile,folder=path, print_result=False)

    #-- ricordo che i weights vengono moltiplicati per beta dentro la funzione cpp --#
    data = create_time_lagged_dataset_cpp(inputFile,path=path)
    x_t,x_lag,w_t,w_lag = np.array(data[:][0]).T[index],np.array(data[:][1]).T[index],np.array(data[:][2]),np.array(data[:][3])
    Nw = np.sum(w_t)
    mean = np.average(x_t,weights=w_t)
    variance = np.cov(x_t,aweights=w_t)
    autocorr = np.sum( np.multiply( np.multiply( (x_t-mean), (x_lag-mean) ), w_lag ) ) / (Nw*variance)

    return autocorr

#-- all descriptors --#
def my_autocorrelation_cpp_all(inputFile,lag,path="./",weight=True):
    
    #-- modify lag time in input file --#
    execute("sed -i 's/lag.*/lag = "+str(lag)+"/g' "+inputFile,folder=path, print_result=False)

    #-- modify if_weight in input file --#
    if not weight:
        execute("sed -i 's/if_weights.*/if_weights = 0/g' "+inputFile,folder=path, print_result=False)
    else:
        execute("sed -i 's/if_weights.*/if_weights = 1/g' "+inputFile,folder=path, print_result=False)

    #-- ricordo che i weights vengono moltiplicati per beta dentro la funzione cpp --#
    data = create_time_lagged_dataset_cpp(inputFile,path=path)
    x_t,x_lag,w_t,w_lag = np.array(data[:][0]),np.array(data[:][1]),np.array(data[:][2]),np.array(data[:][3])
    Nw = np.sum(w_t)
    mean = np.array([ np.average(x_t.T[i],weights=w_t) for i in range(x_t.shape[1]) ])
    variance = np.array([ np.cov(x_t.T[i],aweights=w_t) for i in range(x_t.shape[1]) ])
    autocorr = np.array([np.sum( np.multiply(np.multiply((x_t.T[i]-mean[i]),(x_lag.T[i]-mean[i])),w_lag))/(Nw*variance[i]) for i in range(x_t.shape[1])])
    
    return autocorr


