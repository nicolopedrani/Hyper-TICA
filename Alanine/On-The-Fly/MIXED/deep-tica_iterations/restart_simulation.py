#-- useful python script for training the DeepTICA cvs --#
from utils import *

#-- to not visualize warnings --#
import warnings
warnings.filterwarnings('ignore')

import time

start = time.time()

iteration = 2# non è arrivato fino al 30. Errore con la Barrier

kb=0.008314
#-- SIMULATION PARAMETERS --#
sim_parameters = {
    'temp':300, 
    'beta': 1./(300*kb),
    'kbt': None,
    #-- parameters to compute the fes --#
    'blocks':2,
    'bandwidth': 0.02,
    'plot_max_fes' :70,
}
#--------------------------------------#

starting_folder = "bias"+str(iteration)+"/"
# load data
data = load_dataframe(starting_folder+"COLVAR")
descriptors_names = data.filter(regex='^d[^a-z]').columns.values

#-- train_datasets and valid_datasets list, it will be filled with new data every iteration
train_datasets = []
valid_datasets = []
# torch seed 
torch.manual_seed(21)

BARRIER = 35 # barrier parameter for OPES  
STRIDE = 100 # stride of the simulation, usually 100 which means 1/5 ps
iterations = 5 # number of iterations, I have not decided yet which criterion to stop the iterations
dt = 0.000002 # time step of simulation, in nanoseconds
Time = 6 # in nanoseconds, time of single simulation
size = (Time/dt)/STRIDE # total sampled points for each simulation
restart = True # if restart simulation
#-- minimum and maximum lag time --#
min_lag,max_lag = 0.2,10 
n = 5 # how many lag times between min and max lag
lags = np.linspace(min_lag,max_lag,n) #-- how many batches for the train and valid set of a single simulation
shuffle = False # if shuffle the data between batches
train_sim = None # number of previous simulations to train the NN

#-- TRAINING PARAMETERS --#
n_output = 2 # 2 non linear combination of the descriptors  
n_input = len(descriptors_names) # can change..
train_parameters = {
              'descriptors': '^d[^a-z]', # can change during simulation
              'nodes':[n_input,30,30,n_output],
              'activ_type': 'tanh',#'relu','selu','tanh'
              'lag_time':10, 
              'loss_type': 'sum', 
              'n_eig': n_output,
              'trainsize':0.7, 
              'lrate':1e-3,
              'l2_reg':0.,
              'num_epochs':1000,
              'batchsize': -1, #---> è da fare sul train loder and valid loader
              'es_patience':100,
              'es_consecutive':True,
              'standardize_outputs':True,
              'standardize_inputs': True,
              'log_every':50,
              }

# how many data in single batch, batchsize
n_train = int( size*train_parameters["trainsize"] )
n_valid = int( size*(1-train_parameters["trainsize"])-int(10*max_lag) )
print("training samples: ",n_train, "\t validation samples", n_valid)

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

t = data['time'].values
X = data[descriptors_names].values

# create time lagged dataset with different lag times
for lag in lags:
    #random split
    # TensorDataset (x_t,x_lag,w_t,w_lag)
    dataset = create_time_lagged_dataset(X,t=t,lag_time=np.round(lag,3),interval=[0,n_train+n_valid])
    train_data, valid_data = random_split(dataset,[n_train,n_valid])
    train_datasets.append(train_data)
    valid_datasets.append(valid_data)

train_loader = FastTensorDataLoader(train_datasets, batch_size=n_train,shuffle=shuffle)
valid_loader = FastTensorDataLoader(valid_datasets, batch_size=n_valid,shuffle=shuffle)

#-- TRAIN --#
# MODEL
model = DeepTICA_CV(train_parameters['nodes'],activation=train_parameters['activ_type'],gaussian_random_initialization=True)
model.to(device)
# OPTIMIZER (Adam)
opt = torch.optim.Adam(model.parameters(), lr=train_parameters['lrate'], weight_decay=train_parameters['l2_reg'])
# lrscheduler
#model.set_LRScheduler(opt,min_lr=5e-5)
model.set_optimizer(opt)
if valid_loader is not None:
    # EarlyStopping
    model.set_earlystopping(patience=train_parameters['es_patience'],
                            min_delta=0.001,consecutive=train_parameters['es_consecutive'], save_best_model=True, log=False) 

# TRAIN
model.fit(train_loader=train_loader,valid_loader=valid_loader,
    standardize_inputs=train_parameters['standardize_inputs'],
    standardize_outputs=train_parameters['standardize_outputs'],
    loss_type=train_parameters['loss_type'],
    n_eig=train_parameters['n_eig'],
    nepochs=train_parameters['num_epochs'],
    info=False, log_every=train_parameters['log_every'])

#-- move the model back to cpu for convenience --#
model.to('cpu')
#-- export checkpoint (for loading the model back to python) and torchscript traced module --#
save_folder = starting_folder+"deeptica/"
try:
    os.mkdir(save_folder)
except:
    print("already exists")
#-- move to cpu before saving results --#
model.to("cpu")
model.export(save_folder)

folder = "deep_bias1/"
Path(folder).mkdir(parents=True, exist_ok=True)

with open(folder+"plumed.dat","w") as file:
    print("""
# vim:ft=plumed

MOLINFO STRUCTURE=input.ala2.pdb
phi: TORSION ATOMS=@phi-2 
psi: TORSION ATOMS=@psi-2
#energy
ene: ENERGY

# include descriptors
INCLUDE FILE=plumed_descriptors.data

# define cv
deep0: PYTORCH_MODEL FILE=../bias"""+str(iteration)+"""/deeptica/model.ptc ARG=d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18,d19,d20,d21,d22,d23,d24,d25,d26,d27,d28,d29,d30,d31,d32,d33,d34,d35,d36,d37,d38,d39,d40,d41,d42,d43,d44,d45

opes: OPES_METAD ARG=deep0.node-0,deep0.node-1 TEMP=300 BIASFACTOR=2 PACE=500 RESTART=NO FILE=KERNELS BARRIER="""+str(BARRIER)+""" STATE_WFILE=RestartKernels STATE_WSTRIDE=500*10  
#opes: OPES_METAD ARG=deep0.node-0 TEMP=300 BIASFACTOR=2 PACE=500 RESTART=NO FILE=KERNELS BARRIER="""+str(BARRIER)+""" STATE_WFILE=RestartKernels STATE_WSTRIDE=500*10  

PRINT FMT=%g STRIDE="""+str(STRIDE)+""" FILE=COLVAR ARG=*

ENDPLUMED
""",file=file)

#-- run gromacs --#
execute("cp script/input.* script/plumed_descriptors.data script/run_gromacs.sh "+folder,folder=".")
execute("sed -i '0,/ns/s/ns.*/ns="+str(Time)+"/' run_gromacs.sh",folder=folder)
#restart simulation
execute("sed -i '0,/cpi_state/s/cpi_state.*/cpi_state=true/' run_gromacs.sh",folder=folder, print_result=False)
execute("cp "+starting_folder+"alanine.part000"+str(iteration+1)+".log "+starting_folder+"alanine.part000"+str(iteration+1)+".xtc "+starting_folder+"alanine.part000"+str(iteration+1)+".edr "+starting_folder+"state.cpt "+folder,folder=".", print_result=False) 

run_gromacs = execute("./run_gromacs.sh",folder=folder)

#training new simulation
#-- TRAIN --#
data = load_dataframe(folder+"COLVAR")
t = data['time'].values
X = data[descriptors_names].values

# create time lagged dataset with different lag times
for lag in lags:
    #random split
    # TensorDataset (x_t,x_lag,w_t,w_lag)
    dataset = create_time_lagged_dataset(X,t=t,lag_time=lag,interval=[0,n_train+n_valid])
    train_data, valid_data = random_split(dataset,[n_train,n_valid])
    train_datasets.append(train_data)
    valid_datasets.append(valid_data)

if train_sim is not None:
    train_loader = FastTensorDataLoader(train_datasets[-train_sim*n:], batch_size=n_train,shuffle=shuffle)
    valid_loader = FastTensorDataLoader(valid_datasets[-train_sim*n:], batch_size=n_valid,shuffle=shuffle)
else:
    train_loader = FastTensorDataLoader(train_datasets, batch_size=n_train,shuffle=shuffle)
    valid_loader = FastTensorDataLoader(valid_datasets, batch_size=n_valid,shuffle=shuffle)

train_loader = FastTensorDataLoader(train_datasets, batch_size=n_train,shuffle=shuffle)
valid_loader = FastTensorDataLoader(valid_datasets, batch_size=n_valid,shuffle=shuffle)

# MODEL
model = DeepTICA_CV(train_parameters['nodes'],activation=train_parameters['activ_type'],gaussian_random_initialization=True)
model.to(device)
# OPTIMIZER (Adam)
opt = torch.optim.Adam(model.parameters(), lr=train_parameters['lrate'], weight_decay=train_parameters['l2_reg'])
# lrscheduler
#model.set_LRScheduler(opt,min_lr=5e-5)
model.set_optimizer(opt)
if valid_loader is not None:
    # EarlyStopping
    model.set_earlystopping(patience=train_parameters['es_patience'],
                            min_delta=0.005,consecutive=train_parameters['es_consecutive'], save_best_model=True, log=False) 
# TRAIN
model.fit(train_loader=train_loader,valid_loader=valid_loader,
    standardize_inputs=train_parameters['standardize_inputs'],
    standardize_outputs=train_parameters['standardize_outputs'],
    loss_type=train_parameters['loss_type'],
    n_eig=train_parameters['n_eig'],
    nepochs=train_parameters['num_epochs'],
    info=False, log_every=train_parameters['log_every'])
#-- move the model back to cpu for convenience --#
model.to('cpu')
#-- export checkpoint (for loading the model back to python) and torchscript traced module --#
save_folder = folder+"deeptica/"
try:
    os.mkdir(save_folder)
except:
    print("already exists")
#-- move to cpu before saving results --#
model.to("cpu")
model.export(save_folder)

#-- Start Iterations --# 
for i in range(1,iterations):

    print("ITERATION NUMBER ", i)

    # start bias simulation    
    folder += "deep_bias"+str(i+1)+"/"
    Path(folder).mkdir(parents=True, exist_ok=True)

    with open(folder+"plumed.dat","w") as file:
        print("""
        # vim:ft=plumed

        MOLINFO STRUCTURE=input.ala2.pdb
        phi: TORSION ATOMS=@phi-2 
        psi: TORSION ATOMS=@psi-2
        #energy
        ene: ENERGY

        # include descriptors
        INCLUDE FILE=plumed_descriptors.data
        
        # define cv
        deep"""+str(i)+""": PYTORCH_MODEL FILE=../deeptica/model.ptc ARG=d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18,d19,d20,d21,d22,d23,d24,d25,d26,d27,d28,d29,d30,d31,d32,d33,d34,d35,d36,d37,d38,d39,d40,d41,d42,d43,d44,d45

        opes: OPES_METAD ARG=deep"""+str(i)+""".node-0,deep"""+str(i)+""".node-1 BIASFACTOR=2 TEMP=300 PACE=500 RESTART=NO FILE=KERNELS BARRIER="""+str(BARRIER)+""" STATE_WFILE=RestartKernels STATE_WSTRIDE=500*10  
        #opes: OPES_METAD ARG=deep"""+str(i)+""".node-0 TEMP=300 BIASFACTOR=4 PACE=500 RESTART=NO FILE=KERNELS BARRIER="""+str(BARRIER)+""" STATE_WFILE=RestartKernels STATE_WSTRIDE=500*10  

        PRINT FMT=%g STRIDE="""+str(STRIDE)+""" FILE=COLVAR ARG=*

        ENDPLUMED
        """,file=file)

    #-- run gromacs --#
    execute("cp script/input.* script/plumed_descriptors.data script/positions.data script/run_gromacs.sh "+folder,folder=".")
    execute("sed -i '0,/ns/s/ns.*/ns="+str(Time)+"/' run_gromacs.sh",folder=folder)
    if restart:
        #restart simulation
        execute("sed -i '0,/cpi_state/s/cpi_state.*/cpi_state=true/' run_gromacs.sh",folder=folder, print_result=False)
        # restart simulation
        n = i+iteration+1
        if n==1:
            execute("cp ../alanine.log ../alanine.xtc ../alanine.edr ../state.cpt .",folder=folder, print_result=False)
        elif n<10:
            execute("cp ../alanine.part000"+str(n)+".log ../alanine.part000"+str(n)+".xtc ../alanine.part000"+str(n)+".edr ../state.cpt .",folder=folder, print_result=False)
        elif n>=10:
            execute("cp ../alanine.part00"+str(n)+".log ../alanine.part00"+str(n)+".xtc ../alanine.part00"+str(n)+".edr ../state.cpt .",folder=folder, print_result=False) 

    run_gromacs = execute("./run_gromacs.sh",folder=folder)

    #-- TRAIN --#
    data = load_dataframe(folder+"COLVAR")
    t = data['time'].values
    X = data[descriptors_names].values
    
    # create time lagged dataset with different lag times
    for lag in lags:
        #random split
        # TensorDataset (x_t,x_lag,w_t,w_lag)
        dataset = create_time_lagged_dataset(X,t=t,lag_time=lag,interval=[0,n_train+n_valid])
        train_data, valid_data = random_split(dataset,[n_train,n_valid])
        train_datasets.append(train_data)
        valid_datasets.append(valid_data)

    if train_sim is not None:
        train_loader = FastTensorDataLoader(train_datasets[-train_sim*n:], batch_size=n_train,shuffle=shuffle)
        valid_loader = FastTensorDataLoader(valid_datasets[-train_sim*n:], batch_size=n_valid,shuffle=shuffle)
    else:
        train_loader = FastTensorDataLoader(train_datasets, batch_size=n_train,shuffle=shuffle)
        valid_loader = FastTensorDataLoader(valid_datasets, batch_size=n_valid,shuffle=shuffle)
    
    train_loader = FastTensorDataLoader(train_datasets, batch_size=n_train,shuffle=shuffle)
    valid_loader = FastTensorDataLoader(valid_datasets, batch_size=n_valid,shuffle=shuffle)

    # MODEL
    model = DeepTICA_CV(train_parameters['nodes'],activation=train_parameters['activ_type'],gaussian_random_initialization=True)
    model.to(device)
    # OPTIMIZER (Adam)
    opt = torch.optim.Adam(model.parameters(), lr=train_parameters['lrate'], weight_decay=train_parameters['l2_reg'])
    # lrscheduler
    #model.set_LRScheduler(opt,min_lr=5e-5)
    model.set_optimizer(opt)
    if valid_loader is not None:
        # EarlyStopping
        model.set_earlystopping(patience=train_parameters['es_patience'],
                                min_delta=0.005,consecutive=train_parameters['es_consecutive'], save_best_model=True, log=False) 
    # TRAIN
    model.fit(train_loader=train_loader,valid_loader=valid_loader,
        standardize_inputs=train_parameters['standardize_inputs'],
        standardize_outputs=train_parameters['standardize_outputs'],
        loss_type=train_parameters['loss_type'],
        n_eig=train_parameters['n_eig'],
        nepochs=train_parameters['num_epochs'],
        info=False, log_every=train_parameters['log_every'])
    #-- move the model back to cpu for convenience --#
    model.to('cpu')
    #-- export checkpoint (for loading the model back to python) and torchscript traced module --#
    save_folder = folder+"deeptica/"
    try:
        os.mkdir(save_folder)
    except:
        print("already exists")
    #-- move to cpu before saving results --#
    model.to("cpu")
    model.export(save_folder)

print("###--- End Simulations ---###")
print("total time of simulations: ", (iterations+1)*Time, " ns")
end = time.time()
print("Run for: ",end - start, " s")

