# first attemp to generalize the reinforcement tica

#-- useful python script for training the DeepTICA cvs --#
import re
from utils import *
#-- to not visualize warnings --#
import warnings
warnings.filterwarnings('ignore')

import time

start = time.time()

#-- SIMULATION PARAMETERS --#
kb=0.008314
sim_parameters = {
    'temp':340, 
    'beta': 1./(340*kb),
    'kbt': None,
    #-- parameters to compute the fes --#
    'blocks':2,
    'bandwidth': 0.02,
    'plot_max_fes' :70,
}

BARRIER = 40 # barrier parameter for OPES  
correction_factor = 0.9 # if the selected barrier is too high it can broke che system and gromacs fails. With this factor I change the
                        # the barrier value, setting it to 0.9 its previous value
STRIDE = 100 # stride of the simulation, usually 100 which means 1/5 ps
iterations = 10 # number of iterations, I have not decided yet which criterion to stop the iterations
dt = 0.000002 # time step of simulation, in nanoseconds
Time = 10 # in nanoseconds, time of single simulation
size = (Time/dt)/STRIDE # total sampled points for each simulation
restart = True # if restart simulation
#-- minimum and maximum lag time --#
min_lag,max_lag = 0.2,5 #0.2,20
n = 5 # how many lag times between min and max lag
lags = np.linspace(min_lag,max_lag,n) #-- how many batches for the train and valid set of a single simulation
train_sim = None # number of previous simulations to train the NN
shuffle = False # if shuffle the data between batches

# arguments for deep tica cvs
ARG = ""
count =0
descriptors_file = "plumed-descriptors.dat" # "plumed-descriptors.dat", "plumed-driver.dat"
file = open('script/'+descriptors_file, 'r').readlines() # all descriptors
#file = open('script/plumed-descriptors.dat', 'r').readlines() # most important descriptors from Luigi's article, it is not correct a priori
for line in file:
    if count > 6:
        #print(line)
        if line == "\n":
            break
        ARG+=line.split()[0][:-1]
        ARG+=","
    count+=1

print("###--- PARAMETERS ---###")
print("Barrier for OPES: ", BARRIER)
print("print points every ", STRIDE, " steps")
print("each iteration lasts ", Time, " ns")
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
# write plumed input file for unbias simulation
with open(folder+"plumed.dat","w") as file:
    print("""
# vim:ft=plumed

UNITS LENGTH=nm

MOLINFO MOLTYPE=protein STRUCTURE=chignolin-ref.pdb
WHOLEMOLECULES ENTITY0=1-166

# Define CVs

# Select Calpha
PROTEIN: GROUP ATOMS=1-166
CA: GROUP ATOMS=5,26,47,61,73,88,102,109,123,147
# RMSD
rmsd_ca: RMSD REFERENCE=chignolin-ca.pdb TYPE=OPTIMAL
# END-TO-END DISTANCE
end: DISTANCE ATOMS=5,147
# HBONDS
hbonds: CONTACTMAP ATOMS1=23,145 ATOMS2=45,120 ATOMS3=56,100 ATOMS4=56,107 SWITCH={RATIONAL R_0=0.4 NN=6 MM=8} SUM
ene: ENERGY

INCLUDE FILE="""+descriptors_file+"""

PRINT FMT=%g STRIDE="""+str(STRIDE)+""" FILE=COLVAR ARG=*

ENDPLUMED
""",file=file)

print("###--- Start Simulations ---###")
#-- run gromacs --#
execute("cp script/* "+folder,folder=".")
execute("sed -i '0,/ns/s/ns.*/ns="+str(Time)+"/' run_gromacs.sh",folder=folder)
execute("./run_gromacs.sh",folder=folder)
# load data
data = load_dataframe(folder+"COLVAR")
descriptors_names = data.filter(regex='^dd[^a-z]').columns.values
print(descriptors_names)
print( len(descriptors_names) )

#-- TRAINING PARAMETERS --#
n_output = 5
n_input = len(descriptors_names) # can change..
train_parameters = {
              'descriptors': '^dd[^a-z]', # can change during simulation
              'nodes':[n_input,60,30,n_output],
              'activ_type': 'tanh',#'relu','selu','tanh'
              'lag_time':10, 
              'loss_type': 'sum', 
              'n_eig': n_output,
              'trainsize':0.7, 
              'lrate':1e-3,
              'l2_reg':0.,
              'num_epochs':1000,
              'batchsize': -1, #---> è da fare sul train loder and valid loader
              'es_patience':50,
              'es_consecutive':True,
              'standardize_outputs':True,
              'standardize_inputs': True,
              'log_every':50,
              }

print("layers: ",train_parameters["nodes"])
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
    dataset = create_time_lagged_dataset(X,t=t,lag_time=lag,interval=[0,n_train+n_valid])
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

for i in range(1,iterations):

    print("ITERATION NUMBER ", i)

    # start bias simulation    
    folder += "bias"+str(i)+"/"
    Path(folder).mkdir(parents=True, exist_ok=True)

    # plumed file
    old_cv = ""
    if i>1:
        for deep in range(i-1):
            old_cv+="#deep"+str(deep)+": PYTORCH_MODEL FILE="+"../"*(i-deep)+"deeptica/model.ptc ARG="+ARG+"\n"     

    with open(folder+"plumed.dat","w") as file:
        print("""
        # vim:ft=plumed

        UNITS LENGTH=nm

        MOLINFO MOLTYPE=protein STRUCTURE=chignolin-ref.pdb
        WHOLEMOLECULES ENTITY0=1-166

        # Define CVs
        # Select Calpha
        PROTEIN: GROUP ATOMS=1-166
        CA: GROUP ATOMS=5,26,47,61,73,88,102,109,123,147
        # RMSD
        rmsd_ca: RMSD REFERENCE=chignolin-ca.pdb TYPE=OPTIMAL
        # END-TO-END DISTANCE
        end: DISTANCE ATOMS=5,147
        # HBONDS
        hbonds: CONTACTMAP ATOMS1=23,145 ATOMS2=45,120 ATOMS3=56,100 ATOMS4=56,107 SWITCH={RATIONAL R_0=0.4 NN=6 MM=8} SUM
        ene: ENERGY

        INCLUDE FILE="""+descriptors_file+"""

        # old cvs
        """
        +old_cv+
        """
        # define cv
        deep"""+str(i-1)+""": PYTORCH_MODEL FILE=../deeptica/model.ptc ARG="""+ARG+"""

        opes: OPES_METAD ARG=deep"""+str(i-1)+""".node-0,deep"""+str(i-1)+""".node-1 TEMP=340 PACE=500 BIASFACTOR=4 RESTART=NO FILE=KERNELS BARRIER="""+str(BARRIER)+""" STATE_WFILE=RestartKernels STATE_WSTRIDE=500*10  

        PRINT FMT=%g STRIDE="""+str(STRIDE)+""" FILE=COLVAR ARG=*

        ENDPLUMED
        """,file=file)

    #-- run gromacs --#
    execute("cp script/* "+folder,folder=".")
    execute("sed -i '0,/ns/s/ns.*/ns="+str(Time)+"/' run_gromacs.sh",folder=folder)
    if restart:
        #restart simulation
        execute("sed -i '0,/cpi_state/s/cpi_state.*/cpi_state=true/' run_gromacs.sh",folder=folder, print_result=False)
        # restart simulation
        if i==1:
            execute("cp ../chignolin.log ../chignolin.xtc ../chignolin.edr ../state.cpt .",folder=folder, print_result=False)
        elif i<10:
            execute("cp ../chignolin.part000"+str(i)+".log ../chignolin.part000"+str(i)+".xtc ../chignolin.part000"+str(i)+".edr ../state.cpt .",folder=folder, print_result=False)
        elif i>=10:
            execute("cp ../chignolin.part00"+str(i)+".log ../chignolin.part00"+str(i)+".xtc ../chignolin.part00"+str(i)+".edr ../state.cpt .",folder=folder, print_result=False) 

    run_gromacs = execute("./run_gromacs.sh",folder=folder)

    ''' I am not able yet to solve this issue
    print("run gromacs: ",run_gromacs)
    newBARRIER = BARRIER
    while not run_gromacs:
        print("BARRIER too high, lowering BARRIER by a factor of: ", 1-correction_factor)
        newBARRIER = correction_factor*newBARRIER
        execute("sed -i '0,/BARRIER/s/BARRIER.*/BARRIER="+str(newBARRIER)+"/' plumed.dat",folder=folder, print_result=False)
        run_gromacs = execute("./run_gromacs.sh",folder=folder)
    '''

    #-- TRAIN --#
    data = load_dataframe(folder+"COLVAR")
    t = data['time'].values
    X = data[descriptors_names].values
    '''
    # alternative method to not modify temperature but only rescale the bias
    logweight = data["opes.bias"].to_numpy()-max(data["opes.bias"].to_numpy())
    logweight *= sim_parameters["beta"]
    logweight /= np.abs(min(logweight))
    dt = t[1]-t[0]
    tprime = dt * np.cumsum(np.exp(logweight))
    '''
    
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

    # evaluate the variance of new cvs
    #logweight = (data["opes.bias"].to_numpy()-max(data["opes.bias"].to_numpy()) )*sim_parameters["beta"]

print("###--- End Simulations ---###")
print("total time of simulations: ", iterations*Time, " ns")
end = time.time()
print("Run for: ",end - start, " s")

