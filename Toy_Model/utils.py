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

