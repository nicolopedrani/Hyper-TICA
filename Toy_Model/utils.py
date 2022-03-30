import numpy as np
import pandas as pd
import torch
torch.manual_seed(21)

from mlcvs.utils.data import create_time_lagged_dataset, FastTensorDataLoader
from torch.utils.data import Subset,random_split
from mlcvs.utils.io import load_dataframe
from mlcvs.tica import DeepTICA_CV

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

#takes the orthogonal component of v in respect of w, with boltzmann product
def make_orthogonal(modelv,modelw,X,j=0,k=1,logweight=None):
    return modelv-boltzmann_product(modelv,modelw,X,j=j,k=k,logweight=logweight)/boltzmann_product(modelw,modelw,X,j=k,k=k,logweight=logweight)*modelw

#the boltzmann scalar product is the integral on the boltzmann sampling X of the product of two functions f and g
def boltzmann_product(model0,model1,X,j=0,k=1,logweight=None):

    if logweight is not None:
        weights= np.exp(logweight)
    else:
        weights=np.ones(X.shape[0])

    i,sumtot,sumcv1,sumcv2=0,0,0,0
    for elem in X:
        el = torch.Tensor(elem)
        a = model0(el)[j]
        b = model1(el)[k]
        sumtot+=a*b*weights[i]
        sumcv1+=a*a*weights[i]
        sumcv2+=b*b*weights[i]
        i+=1
        
    sumtot/=i
    sumcv1/=i
    sumcv2/=i
    sumtot=sumtot.detach().cpu().numpy()
    sumcv1=sumcv1.detach().cpu().numpy()
    sumcv2=sumcv2.detach().cpu().numpy()
    
    return sumtot/np.sqrt(sumcv1*sumcv2)
