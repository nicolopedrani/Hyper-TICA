import numpy as np
import pandas as pd
import torch
torch.manual_seed(21)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.rcParams.update({'font.size': 15})
import matplotlib as mpl
from scipy.interpolate import griddata
from scipy import integrate

from mlcvs.utils.data import create_time_lagged_dataset, FastTensorDataLoader, tprime_evaluation, find_time_lagged_configurations
from torch.utils.data import Subset,random_split
from mlcvs.utils.io import load_dataframe
from mlcvs.tica import DeepTICA_CV
from mlcvs.utils.fes import compute_fes

from scipy.optimize import curve_fit
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

#-- fitting time auto-correlation function --#
def f(x,l):
    return np.exp(-x/l)

def create_time_lagged_dataset(X, t = None, lag_time = 10, logweights = None, tprime = None, interval = None):
    """
    Create a dataset of time-lagged configurations. If a set of (log)weights is given the search is performed in the accelerated time.

    Parameters
    ----------
    X : array-like
        input descriptors
    t : array-like, optional
        time series, by default np.arange(len(X))
    lag_time: float, optional
        lag between configurations, by default = 10        
    logweights : array-like,optional
        logweights to evaluate rescaled time as dt' = dt*exp(logweights)
    tprime : array-like,optional
        rescaled time estimated from the simulation. If not given 'tprime_evaluation(t,logweights)' is used instead
    """

    # check if dataframe
    if type(X) == pd.core.frame.DataFrame:
        X = X.values
    if type(t) == pd.core.frame.DataFrame:
        t = t.values

    # assert
    assert t.ndim == 1 
    assert len(X) == len(t)

    # define time if not given
    if t is None:
        t = np.arange(0,len(X))

    #define tprime if not given
    if tprime is None:
        tprime = tprime_evaluation(t, logweights)

    # find pairs of configurations separated by lag_time
    data = find_time_lagged_configurations(X, tprime,lag=lag_time)
    
    if interval is not None:
        # covert to a list
        data = list(data)
        # assert dimension of interval
        assert len(interval) == 2
        # modifies the content of data by slicing
        for i in range(len(data)):
            data[i] = data[i][interval[0]:interval[1]]

    #return data
    return torch.utils.data.TensorDataset(*data)

def train_deeptica_load(beta=1.0,lag_time=10,path="colvar.data",descriptors="^p.",trainsize=0.8,reweighting=True,path_cpp=None):
    
    data = load_dataframe(path)
    
    search_values=descriptors
    search_names=descriptors
    X, names, t = data.filter(regex=search_values).values, data.filter(regex=search_names).columns.values, data['time'].values
    n_features = X.shape[1]
    
    if reweighting:
        # Compute logweights for time reweighting
        logweight = data["opes.bias"].to_numpy()*beta
        #-- the logweights are V(x,y)*beta --#
    else:
        logweight=None
        #print("no weights")
 
    if path_cpp is not None:
        print("## Using CPP implementation ##")
        execute("sed -i 's/lag.*/lag = "+str(lag_time)+"/g' input.dat",folder=path_cpp, print_result=False)
        dataset = create_time_lagged_dataset_cpp("input.dat",path=path_cpp)
    else:
        dataset = create_time_lagged_dataset(X,t=t,lag_time=lag_time,logweights=logweight)
    n_train  = int( trainsize * len(dataset) )
    n_valid  = len(dataset) - n_train
    train_data, valid_data = random_split(dataset,[n_train,n_valid])

    # create dataloaders
    train_loader = FastTensorDataLoader(train_data, batch_size=len(train_data))
    valid_loader = FastTensorDataLoader(valid_data, batch_size=len(valid_data))

    #print('Time-lagged pairs:\t',len(dataset))
    #print('Training data:\t\t',len(train_data))
    #print('Validation data:\t',len(valid_data))

    return data,logweight,train_loader,valid_loader,X

def training(beta,path,train_parameters):

    data,logweight,train_loader,valid_loader,X = train_deeptica_load(beta=beta,lag_time=train_parameters["lag_time"],  
                                                                            path=path,descriptors=train_parameters["descriptors"],
                                                                            trainsize=train_parameters['trainsize'],
                                                                            reweighting=train_parameters['reweighting'],
                                                                            path_cpp=train_parameters["path_cpp"])
    # DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MODEL
    model = DeepTICA_CV(train_parameters['nodes'],activation=train_parameters['activ_type'])
    model.to(device)

    # OPTIMIZER (Adam)
    opt = torch.optim.Adam(model.parameters(), lr=train_parameters['lrate'], weight_decay=train_parameters['l2_reg'])
    if train_parameters["lrscheduler"]:
        model.set_LRScheduler(opt)
    model.set_optimizer(opt)
    
    if train_parameters["earlystop"]:
        model.set_earlystopping(patience=train_parameters['es_patience'],
                            min_delta=0.,consecutive=train_parameters['es_consecutive'], save_best_model=True, log=False)

    # TRAIN
    model.fit(train_loader,valid_loader=valid_loader,
        standardize_inputs=train_parameters['standardize_inputs'],
        standardize_outputs=train_parameters['standardize_outputs'],
        loss_type=train_parameters['loss_type'],
        n_eig=train_parameters['n_eig'],
        nepochs=train_parameters['num_epochs'],
        batch_sie=train_parameters['batchsie'],
        info=False, log_every=train_parameters['log_every'])

    return model,data,logweight,X,train_loader,valid_loader

#-- only for unbias simulation --#
def fit_timeacorr(descriptors_names,data,axs=None):

    if axs is None:
        fig,axs = plt.subplots(1,2,figsize=(14,5))#,sharey=True)

    #-- unit of step --#
    last=10
    x = np.linspace(0,last+1,last)
    acorr = np.empty(last)
    corr_length = np.empty(len(descriptors_names))
    k=0

    for desc in descriptors_names:
        print("autocorrelation for ", desc)
        for i in range(last):
            acorr[i] = data[desc].autocorr(i)
        p_opt,p_cov = curve_fit(f,x[:last],acorr[:last],maxfev=2000)
        corr_length[k] = p_opt[0]
        f_fit = f(x[0:last],p_opt[0])
        axs[0].plot(x,acorr)
        axs[0].plot(x,f_fit)
        k+=1

    axs[0].legend([r"$e^{-\frac{\tau}{\xi}}$"])
    
    c_length = pd.DataFrame(descriptors_names,columns=["descriptors"])
    c_length["cor_len"] = corr_length
    c_length.plot(kind="bar",x="descriptors",y="cor_len",rot=35,ax=axs[1],fontsize=15,label=r"$\xi$")

    axs[0].set_xlabel(r'$\tau$')
    axs[0].set_title(r'$C(\tau)$')
    axs[1].set_title(r'$\xi=$ from $e^{-\frac{\tau}{\xi}}$')
    plt.tight_layout()

    plt.show()

#-- plot model loss function --#
def plot_model_lossfunction(model):

    n_eig=len(model.nn[-1].bias)

    fig, axs = plt.subplots(1,2,figsize=(12,5))#,dpi=100)

    loss_train = [x.cpu() for x in model.loss_train]
    loss_valid = [x.cpu() for x in model.loss_valid]

    # Loss function
    ax = axs[0]
    ax.plot(loss_train,'-',label='Train')
    ax.plot(loss_valid,'--',label='Valid')
    ax.set_ylabel('Loss Function')

    # Eigenvalues vs epoch
    ax = axs[1]
    with torch.no_grad():
        evals_train = np.asarray(torch.cat(model.evals_train).cpu())
    for i in range(n_eig):
        ax.plot(evals_train[:,i],label='Eig. '+str(i+1))
    ax.set_ylabel('Eigenvalues')
    
    # Common setup
    for ax in axs:
        if model.earlystopping_.early_stop:
            if model.earlystopping_.best_epoch is not None:
                ax.axvline(model.earlystopping_.best_epoch,ls='dotted',color='grey',alpha=0.5,label='Early Stopping')
                ax.set_xlabel('#Epochs')
                ax.legend(ncol=2)

    plt.tight_layout()

def plot_cvs_isolines(model,limits=((-np.pi,np.pi),(-np.pi,np.pi)),interval=50,scatter=None,path="angles/COLVAR"):

        complete_data = load_dataframe(path)
        search_values="^d[^a-z]"
        X_complete = complete_data.filter(regex=search_values).values
        x = complete_data["phi"].to_numpy()[::interval] 
        y = complete_data["psi"].to_numpy()[::interval] 

        fig,axs = plt.subplots(1,2,figsize=(14,8))

        for k,ax in enumerate(axs):
                
                #-- evaluate cvs --#
                z = np.transpose(model(torch.Tensor(X_complete)).detach().numpy())[k][::interval] 
                #-- define grid --#
                npts = len(z)
                xi = np.linspace(limits[0][0],limits[0][1],int(npts/2.))
                yi = np.linspace(limits[1][0],limits[1][1],int(npts/2.))
                #-- grid the data --#
                zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='nearest')
                #-- bounds to plot isolines --#
                bounds = np.arange(np.min(zi), np.max(zi), 0.1)
                cmap = plt.cm.get_cmap('fessa',len(bounds))
                colors = list(cmap(np.arange(len(bounds))))
                cmap = mpl.colors.ListedColormap(colors[:-1], "")
                #-- plot --#
                CS = ax.contourf(xi,yi,zi,cmap=cmap,shading='auto',alpha=1,zorder=-1,
                norm = mpl.colors.BoundaryNorm(bounds, ncolors=len(bounds)-1, clip=False) )
                fig.colorbar(CS, ax=ax)
                if scatter is not None:
                        ax.scatter(scatter["phi"],scatter["psi"],s=2,c='white',alpha=0.8,zorder=-1,label="scatter")
                CS = ax.contour(xi,yi,zi,shading='auto',alpha=1,zorder=-1,linewidths=3)
                ax.clabel(CS,inline=True,fmt='%1.1f',fontsize=10)
                ax.set_xlim(limits[0][0],limits[0][1])
                ax.set_ylim(limits[1][0],limits[1][1])
                ax.legend()
                ax.set_xlabel(r"$\phi$")
                ax.set_ylabel(r"$\psi$")
                ax.set_title('Deep TICA '+str(k+1))

        plt.tight_layout()
        plt.show()

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

#-- My Autocorrelation, with rescaled time if weights are given --#
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


