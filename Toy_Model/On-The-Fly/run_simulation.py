#-- useful python script for training the DeepTICA cvs --#
from utils import *
#-- to not visualize warnings --#
import warnings
warnings.filterwarnings('ignore')

import time

start = time.time()

#-- python script for write correct input files for "ves_md_linearexpansion" plumed module --#
from input_VES import *

#-- to not visualize warnings --#
import warnings
warnings.filterwarnings('ignore')

## SIMULATION PARAMETERS ##

sim_parameters = {
    'nstep':500000, 
    'plumedseed':4525,
    'friction':10,
    'temp':1, #kbt units
    #-- upper basin --#
    #'initial_position':[0.6,0.02],
    #-- middle basin --#
    #'initial_position':[-0.05,0.47],
    #-- lower basin --#
    'initial_position':[-0.55,1.45],
    #-- parameters to compute the fes --#
    'blocks':2,
    'bandwidth': 0.02,
    'plot_max_fes' :16,
}

BARRIER = 5.5 # barrier parameter for OPES  
correction_factor = 0.9 # if the selected barrier is too high it can broke che system and gromacs fails. With this factor I change the
                        # the barrier value, setting it to 0.9 its previous value
STRIDE = 200 # every 1/4 ps 
iterations = 10 # number of iterations, I have not decided yet which criterion to stop the iterations
dt = 0.000005 # time step of simulation, in nanoseconds
Time = (dt*sim_parameters["nstep"]) # in nanoseconds, time of single simulation
size = (sim_parameters["nstep"])/STRIDE # total sampled points for each simulation
restart = True # if restart simulation
#-- minimum and maximum lag time --#
min_lag,max_lag = 1,2 
n = 1 # how many lag times between min and max lag
lags = np.linspace(min_lag,max_lag,n) #-- how many batches for the train and valid set of a single simulation
print(lags)
train_sim = None # number of previous simulations to train the NN
shuffle = False # if shuffle the data between batches

print("###--- PARAMETERS ---###")
print("Barrier for OPES: ", BARRIER)
print("print points every ", STRIDE, " steps")
print("each iteration lasts ", Time, " ns")
print("sample size: ",int(size) )
print("Iterations: ", iterations)
if train_sim is not None:
    print("NN will be trained with the last ", train_sim, " simulation data")
else:
    print("NN will be trained with all the cumulative data")
print("min lag: ", min_lag, "\t max lag: ", max_lag, "\t number of lag times = ", n)
if shuffle:
    print("the data between batches will be shuffled")
#-- train_datasets and valid_datasets list, it will be filled with new data every iteration
train_datasets = []
valid_datasets = []
# torch seed 
torch.manual_seed(21)

# start with unbias simulation
folder = "unbias/"
Path(folder).mkdir(parents=True, exist_ok=True)

with open(folder+"plumed.dat","w") as file:
    print("""
# vim:ft=plumed

# using natural units for Toy Model 
UNITS NATURAL

# compute position for the one particle  
p: POSITION ATOM=1
# adding external potential 
potential: CUSTOM ARG=p.x,p.y FUNC="""+Mullerpot(),"""PERIODIC=NO
ene: BIASVALUE ARG=potential

# Print 
PRINT FMT=%g STRIDE="""+str(STRIDE)+""" FILE=COLVAR ARG=p.x,p.y,ene.bias,

ENDPLUMED
""",file=file)

print("###--- Start Simulations ---###")
#-- run --#
#-- write input files for ves module --#
generate_input_file(name_file=folder+"input",nstep=sim_parameters["nstep"],temp=sim_parameters["temp"],
                    friction=sim_parameters["friction"],random_seed=sim_parameters["plumedseed"],
                    initial_position=sim_parameters["initial_position"])
write_coeff("0",folder+"input")

#-- move necessary files for ves module --#
execute("mv pot_coeffs_input.data "+folder,folder=".")
#-- run plumed --#
execute("plumed ves_md_linearexpansion input",folder=folder)

# load data
data = load_dataframe(folder+"COLVAR")
descriptors_names = data.filter(regex='^p.').columns.values

#-- TRAINING PARAMETERS --#
n_output = 2 # 2 non linear combination of the descriptors  
n_input = len(descriptors_names) # can change..
train_parameters = {
              'descriptors': '^p.', # can change during simulation
              'nodes':[n_input,10,n_output],
              'activ_type': 'tanh',#'relu','selu','tanh'
              'lag_time':10, 
              'loss_type': 'sum', 
              'n_eig': n_output,
              'trainsize':0.7, 
              'lrate':1e-3,
              'l2_reg':0.,
              'num_epochs':200,
              'batchsize': -1, #---> è da fare sul train loder and valid loader
              'es_patience':10,
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
last_positions = np.array(X[-1:])[0][0],np.array(X[-1:])[0][1] 
# create time lagged dataset with different lag times
for lag in lags:
    #random split
    # TensorDataset (x_t,x_lag,w_t,w_lag)
    dataset = create_time_lagged_dataset(X,t=t,lag_time=np.round(lag,3),interval=[0,n_train+n_valid])
    train_data, valid_data = random_split(dataset,[n_train,n_valid])
    train_datasets.append(train_data)
    valid_datasets.append(valid_data)

# to not divide the set, it create a dataset composed by all the found couples with different lag times
#print(n*n_train)
#print(len(ConcatDataset(train_datasets)))

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
save_folder = folder+"deeptica/"
try:
    os.mkdir(save_folder)
except:
    print("already exists")
#-- move to cpu before saving results --#
model.to("cpu")
model.export(save_folder)

for i in range(1,iterations):

    print("ITERATION NUMBER ", i)

    # start bias simulation    
    folder += "bias"+str(i)+"/"
    Path(folder).mkdir(parents=True, exist_ok=True)

    # plumed file
    old_cv = ""
    if i>1:
        for deep in range(i-1):
            old_cv+="deep"+str(deep)+": PYTORCH_MODEL FILE="+"../"*(i-deep)+"deeptica/model.ptc ARG=p.x,p.y\n"

    with open(folder+"plumed.dat","w") as file:
        print("""
        # vim:ft=plumed

        # using natural units for Toy Model 
        UNITS NATURAL

        # compute position for the one particle  
        p: POSITION ATOM=1
        # adding external potential 
        potential: CUSTOM ARG=p.x,p.y FUNC="""+Mullerpot(),"""PERIODIC=NO
        ene: BIASVALUE ARG=potential

        # old cvs
        """
        +old_cv+
        """
        # define cv
        deep"""+str(i-1)+""": PYTORCH_MODEL FILE=../deeptica/model.ptc ARG=p.x,p.y

        # two cvs
        opes: OPES_METAD ARG=deep"""+str(i-1)+""".node-0,deep"""+str(i-1)+""".node-1 TEMP="""+str(sim_parameters["temp"])+""" PACE=500 RESTART=NO CALC_WORK FILE=KERNELS BARRIER="""+str(BARRIER)+""" STATE_WFILE=RestartKernels STATE_WSTRIDE=500*10  

        # one cv
        #opes: OPES_METAD ARG=deep"""+str(i-1)+""".node-0 TEMP="""+str(sim_parameters["temp"])+""" PACE=500 RESTART=NO CALC_WORK FILE=KERNELS BARRIER="""+str(BARRIER)+""" STATE_WFILE=RestartKernels STATE_WSTRIDE=500*10  

        PRINT FMT=%g STRIDE="""+str(STRIDE)+""" FILE=COLVAR ARG=p.x,p.y,deep"""+str(i-1)+""".node-0,deep"""+str(i-1)+""".node-1,ene.bias,opes.*

        ENDPLUMED
        """,file=file)

    #-- run --#
    #-- write input files for ves module --#
    generate_input_file(name_file=folder+"input",nstep=sim_parameters["nstep"],temp=sim_parameters["temp"],
                        friction=sim_parameters["friction"],random_seed=sim_parameters["plumedseed"],
                        initial_position=last_positions)
    write_coeff("0",folder+"input")

    #-- move necessary files for ves module --#
    execute("mv pot_coeffs_input.data "+folder,folder=".")
    #-- run plumed --#
    execute("plumed ves_md_linearexpansion input",folder=folder)

    #-- TRAIN --#
    data = load_dataframe(folder+"COLVAR")
    t = data['time'].values
    X = data[descriptors_names].values
    # alternative method to not modify temperature but only rescale the bias
    logweight = data["opes.bias"].to_numpy()-max(data["opes.bias"].to_numpy())
    logweight /= np.abs(min(logweight))
    logweight /= sim_parameters["temp"]
    dt = t[1]-t[0]
    tprime = dt * np.cumsum(np.exp(logweight))

    # create time lagged dataset with different lag times
    for lag in lags:
        #random split
        # TensorDataset (x_t,x_lag,w_t,w_lag)
        dataset = create_time_lagged_dataset(X,t=t,lag_time=np.round(lag,3),logweights=logweight,tprime=tprime,interval=[0,n_train+n_valid])
        train_data, valid_data = random_split(dataset,[n_train,n_valid])
        train_datasets.append(train_data)
        valid_datasets.append(valid_data)

    if train_sim is not None:
        train_loader = FastTensorDataLoader(train_datasets[-train_sim*n:], batch_size=n_train,shuffle=shuffle)
        valid_loader = FastTensorDataLoader(valid_datasets[-train_sim*n:], batch_size=n_valid,shuffle=shuffle)
    else:
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
    save_folder = folder+"deeptica/"
    try:
        os.mkdir(save_folder)
    except:
        print("already exists")
    #-- move to cpu before saving results --#
    model.to("cpu")
    model.export(save_folder)

    # evaluate the variance of new cvs
    logweight = (data["opes.bias"].to_numpy()-max(data["opes.bias"].to_numpy()) )/sim_parameters["temp"]


print("###--- End Simulations ---###")
print("total time of simulations: ", iterations*Time, " ns")
end = time.time()
print("Run for: ",end - start, " s")