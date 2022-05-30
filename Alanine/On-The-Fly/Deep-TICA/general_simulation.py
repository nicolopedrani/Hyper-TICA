# first attemp to generalize the reinforcement tica

#-- useful python script for training the DeepTICA cvs --#
from utils import *
#-- to not visualize warnings --#
import warnings
warnings.filterwarnings('ignore')

#-- SIMULATION PARAMETERS --#
kb=0.008314
sim_parameters = {
    'temp':300, 
    'beta': 1./(300*kb),
    'kbt': None,
    #-- parameters to compute the fes --#
    'blocks':2,
    'bandwidth': 0.02,
    'plot_max_fes' :70,
}
BARRIER = 30 # barrier parameter for OPES  
STRIDE = 100 # stride of the simulation

#-- TRAINING PARAMETERS --#
n_output = 2 # 2 non linear combination of the descriptors  
n_input = 45 # can change..
train_parameters = {
              'descriptors': '^d[^a-z]', # can change during simulation
              'nodes':[n_input,30,30,n_output],
              'activ_type': 'tanh',#'relu','selu','tanh'
              'lag_time':10, 
              'loss_type': 'sum', 
              'n_eig': n_output,
              'trainsize':0.75, 
              'lrate':1e-3,
              'l2_reg':0.,
              'num_epochs':300,
              'batchsize': -1, #---> è da fare sul train loder and valid loader
              'es_patience':10,
              'es_consecutive':True,
              'standardize_outputs':False,
              'standardize_inputs': True,
              'log_every':50,
              }
# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

iterations = 30 # number of iterations, I have not decided yet which criterion to stop the iterations
dt = 0.000002 # time step of simulation, in nanoseconds
time = 1 # in nanoseconds
size = time/dt # total sampled points
restart = False # if restart simulation
#-- minimum and maximum lag time --#
min_lag,max_lag = 0.2,5
# how many data in single batch, batchsize
n_train = int( size*train_parameters["trainsize"] )
n_valid = int( size*(1-train_parameters["trainsize"])-int(10*max_lag) )
lags = np.linspace(min_lag,max_lag,5) #-- how many batches for the train and valid set of a single simulation
#-- train_datasets and valid_datasets list, it will be filled with new data every iteration
train_datasets = []
valid_datasets = []

# start with unbias simulation
folder = "unbias_A/"
Path(folder).mkdir(parents=True, exist_ok=True)
# write plumed input file for unbias simulation
with open(folder+"plumed.dat","w") as file:
    print("""
# vim:ft=plumed

MOLINFO STRUCTURE=input.ala2.pdb
phi: TORSION ATOMS=@phi-2
psi: TORSION ATOMS=@psi-2
ene: ENERGY

INCLUDE FILE=plumed_descriptors.data

PRINT FMT=%g STRIDE="""+str(STRIDE)+""" FILE=COLVAR ARG=*

ENDPLUMED
""",file=file)

#-- run gromacs --#
execute("cp script/input.* script/plumed_descriptors.data script/run_gromacs.sh "+folder,folder=".")
execute("./run_gromacs.sh",folder=folder)



for i in range(iterations):
    
    folder += "bias"+str(i)+"/"
    Path(folder).mkdir(parents=True, exist_ok=True)

    data = load_dataframe(folder+"COLVAR")
    descriptors_names = data.filter(regex='^d[^a-z]').columns.values
    
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



