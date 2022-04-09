import numpy as np
import pandas as pd
import torch
torch.manual_seed(21)
import matplotlib.pyplot as plt

from mlcvs.utils.data import create_time_lagged_dataset, FastTensorDataLoader
from torch.utils.data import Subset,random_split
from mlcvs.utils.io import load_dataframe
from mlcvs.tica import DeepTICA_CV

from scipy.optimize import curve_fit

#-- fitting time auto-correlation function --#
def f(x,l):
    return np.exp(-x/l)

def tprime_evaluation(X,t=None,logweights=None):

    # define time if not given
    if t is None:
        t = np.arange(0,len(X))

    # rescale time with log-weights if given
    if logweights is not None:
        # compute time increment in simulation time t
        dt = np.round(t[1]-t[0],3)
        # sanitize logweights
        logweights = torch.Tensor(logweights)
        logweights -= torch.max(logweights)
        lognorm = torch.logsumexp(logweights,0)
        logweights /= lognorm
        # compute instantaneus time increment in rescaled time t'
        d_tprime = torch.exp(logweights)*dt
        # calculate cumulative time t'
        tprime = torch.cumsum(d_tprime,0)
    else:
        tprime = t

    return tprime

def train_deeptica_load(temp=1.0,lag_time=10,path="colvar.data",descriptors="^p.",trainsize=0.8,reweighting=True):
    
    data = load_dataframe(path)
    
    search_values=descriptors
    search_names=descriptors
    X, names, t = data.filter(regex=search_values).values, data.filter(regex=search_names).columns.values, data['time'].values
    n_features = X.shape[1]
    
    if reweighting:
        # Compute logweights for time reweighting
        logweight = data["opes.bias"].to_numpy()
        #-- the logweights are V(x,y)/T --#
        logweight = (logweight-max(logweight))/temp
    else:
        print("no weights")
        logweight=None
 
    #tprime = tprime_evaluation(X,t=t,logweights=logweight)
         
    dataset = create_time_lagged_dataset(X,t=t,lag_time=lag_time,logweights=logweight)
    n_train  = int( trainsize * len(dataset) )
    n_valid  = len(dataset) - n_train
    train_data, valid_data = random_split(dataset,[n_train,n_valid])

    # create dataloaders
    train_loader = FastTensorDataLoader(train_data, batch_size=len(train_data))
    valid_loader = FastTensorDataLoader(valid_data, batch_size=len(valid_data))

    print('Time-lagged pairs:\t',len(dataset))
    print('Training data:\t\t',len(train_data))
    print('Validation data:\t',len(valid_data))

    return data,logweight,train_loader,valid_loader,X#,tprime

def training(temp,path,train_parameters):

    data,logweight,train_loader,valid_loader,X = train_deeptica_load(temp=temp,lag_time=train_parameters["lag_time"],  
                                                                            path=path,descriptors=train_parameters["descriptors"],
                                                                            trainsize=train_parameters['trainsize'],
                                                                            reweighting=train_parameters['reweighting'])
    # DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MODEL
    model = DeepTICA_CV(train_parameters['nodes'],activation=train_parameters['activ_type'])
    model.to(device)

    # OPTIMIZER (Adam)
    opt = torch.optim.Adam(model.parameters(), lr=train_parameters['lrate'], weight_decay=train_parameters['l2_reg'])

    # REGULARIZATION
    model.set_optimizer(opt)
    model.set_earlystopping(patience=train_parameters['es_patience'],
                            min_delta=0.,consecutive=train_parameters['es_consecutive'], save_best_model=True, log=False)

    # TRAIN
    model.fit(train_loader,valid_loader,
        standardize_inputs=train_parameters['standardize_inputs'],
        standardize_outputs=train_parameters['standardize_outputs'],
        loss_type=train_parameters['loss_type'],
        n_eig=train_parameters['n_eig'],
        nepochs=train_parameters['num_epochs'],
        info=False, log_every=train_parameters['log_every'])

    return model,data,logweight,X

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
        if model.earlystopping_.best_epoch is not None:
            if model.earlystopping_.early_stop:
                ax.axvline(model.earlystopping_.best_epoch,ls='dotted',color='grey',alpha=0.5,label='Early Stopping')
                ax.set_xlabel('#Epochs')
                ax.legend(ncol=2)

    plt.tight_layout()

def plot_cvs_isolines(model,limits,points=150,n_out=2,scatter=None,axs=None):

    #-- prepare grid --#
    xx,yy = np.meshgrid(np.linspace(limits[0][0],limits[0][1],points),np.linspace(limits[1][0],limits[1][1],points))
    grid = np.transpose(np.array([xx.reshape(points*points),yy.reshape(points*points)]))

    #-- evaluate cvs on the grid --#
    cvs = []
    for i in range(n_out):
        cvs.append(np.transpose(model(torch.Tensor(grid)).detach().cpu().numpy())[i].reshape(points,points))

    #-- plotting --#
    if axs is None:
        fig,axs = plt.subplots(1,n_out,figsize=(12,6))

    for k,ax in enumerate(axs):
        cset = ax.contourf(xx,yy,cvs[k],linewidths=1,cmap="fessa")
        cset = ax.contour(xx,yy,cvs[k],linewidths=3,cmap="gray",linestyles="dashed")
        ax.clabel(cset,inline=True,fmt='%1.1f',fontsize=15)
        ax.set_xlabel("p.x")
        ax.set_ylabel("p.y")
        ax.set_title('Deep-TICA '+str(k+1))
        if scatter is not None:
            ax.scatter(scatter[:,0],scatter[:,1],s=2,c='white',alpha=0.8)
            ax.set_aspect('equal')
        
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

#-- My Autocorrelation --#
def my_autocorrelation(x,lag,weight=None):
    autocorr = np.empty(0)
    N = len(x)
    if weight is None:
        weight = np.ones(N)
    Nw = np.sum(weight)
    x = x.reshape(N)
    mean = np.average(x,weights=weight)
    variance = np.cov(x,aweights=weight)
    #variance = np.var(x[:])
    #mean = np.mean(x[:])   
    sum = 0
    for i in range(N-lag):
        sum += (x[i]-mean)*(x[i+lag]-mean)
    autocorr = np.append(autocorr,sum/(Nw*variance))
    return autocorr