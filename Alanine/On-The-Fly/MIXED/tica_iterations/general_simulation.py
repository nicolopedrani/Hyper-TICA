# first attemp to generalize the reinforcement tica

#-- useful python script for training the DeepTICA cvs --#
from utils import *
#-- to not visualize warnings --#
import warnings
warnings.filterwarnings('ignore')

import time

start = time.time()

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

torch.manual_seed(21)
BARRIER = 40 # barrier parameter for OPES  
STRIDE = 100 # stride of the simulation, usually 100 which means 1/5 ps
iterations = 6 # number of iterations, I have not decided yet which criterion to stop the iterations
dt = 0.000002 # time step of simulation, in nanoseconds
Time = 6 # in nanoseconds, time of single simulation
size = (Time/dt)/STRIDE # total sampled points for each simulation
restart = True # if restart simulation
#-- minimum and maximum lag time --#
lag_time = 3

#-- stop condition for iteration --#
# when it increase of a certain factor, with the respect to the first one,
# then the iterations stop, and the data relative to the last iteration (with apparently a 
# transition) will be analysized with the Deep-TICA
increase_factor = 5
thershold = 200
#-- NOTA: this analysis is performed with the descriptors correlation length,
# but it can be done also with the new find cvs. This operation would be less costly --#

print("###--- PARAMETERS ---###")
print("Barrier for OPES: ", BARRIER)
print("print points every ", STRIDE, " steps")
print("each iteration lasts ", Time, " ns")
print("Iterations: ", iterations)
print("NN will be trained with all the cumulative data")
#-- train_datasets and valid_datasets list, it will be filled with new data every iteration
train_datasets = []
valid_datasets = []
# torch seed 
torch.manual_seed(21)

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

print("###--- Start Simulations ---###")
#-- run gromacs --#
execute("cp script/input.* script/plumed_descriptors.data script/run_gromacs.sh "+folder,folder=".")
execute("sed -i '0,/ns/s/ns.*/ns="+str(Time)+"/' run_gromacs.sh",folder=folder)
execute("./run_gromacs.sh",folder=folder)

# load data
data = load_dataframe(folder+"COLVAR")

descriptors_names = data.filter(regex='^d[^a-z]').columns.values

#-- TRAINING PARAMETERS --#
train_parameters = {
              'descriptors': '^d[^a-z]',
              'lag_time':1,
              'standardize_outputs':True,
              'standardize_inputs': True,
              #if reweight the timescale
              "reweighting": False,
              "step": 1 #
              }

last=int(len(data)/2-1)
'''
#-- autocorrelation length --#
last=int(len(data)/2-1)
x = np.linspace(0,last+1,last)
acorr = np.empty(last)
timescale = np.empty(len(descriptors_names))
for k,desc in enumerate(descriptors_names):
    for j in range(last):
        acorr[j] = data[desc].autocorr(j)
    timescale[k] = integrate.trapz(acorr[:last],x[:last])
times = pd.DataFrame(descriptors_names,columns=["descriptors"])
times["timescale"] = timescale
correlation_length = np.max(timescale)
print("iteration 0, max correlation length ", correlation_length)
'''

#subset of data if necessary
data = data[::train_parameters["step"]]
# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

names = data.filter(regex='^d[^a-z]').columns.values
t = data['time'].values
X = data[descriptors_names].values

# MODEL
model = TICA_CV(n_features=X.shape[1])
model.to(device)

# TRAIN
model.fit(X, t, lag=lag_time)

#-- move the model back to cpu for convenience --# 
model.to('cpu')

#-- print some useful results --#
#print("timescales: ",model.tica.timescales(train_parameters["lag_time"]).detach().cpu().numpy()) 
print("eigenvalues: ",model.tica.evals_.detach().cpu().numpy())
#print("gap: ", model.tica.evals_.detach().cpu().numpy()[0]-model.tica.evals_.detach().cpu().numpy()[1])

model.set_params({"feature_names": names})
#print( model.plumed_input().splitlines()[:2][8:] )


data["cv1"] = np.transpose(model(torch.Tensor(X)).detach().cpu().numpy())[0]
data["cv2"] = np.transpose(model(torch.Tensor(X)).detach().cpu().numpy())[1]

#-- autocorrelation length of new cvs --#
x = np.linspace(0,last+1,last)
acorr = np.empty(last)
timescale = np.empty(len(["cv1","cv2"]))
for k,desc in enumerate(["cv1","cv2"]):
    for j in range(last):
        acorr[j] = data[desc].autocorr(j)
    timescale[k] = integrate.trapz(acorr[:last],x[:last])
times = pd.DataFrame(["cv1","cv2"],columns=["descriptors"])
times["timescale"] = timescale
correlation_length_cv = np.max(timescale)
print("iteration 0, correlation length of cvs ", correlation_length_cv)


#-- start biased iterations --#
for i in range(1,iterations):

    print("ITERATION NUMBER ", i)

    # start bias simulation    
    folder += "bias"+str(i)+"/"
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
        tica_cv1_"""
        +str(i-1)+model.plumed_input().splitlines()[0][8:]+"""\ntica_cv2_"""+str(i-1)+model.plumed_input().splitlines()[1][8:]
        +"""
        
        #opes: OPES_METAD ARG=tica_cv1_"""+str(i-1)+""" TEMP=300 BIASFACTOR=3 PACE=500 RESTART=NO FILE=KERNELS BARRIER="""+str(BARRIER)+""" STATE_WFILE=RestartKernels STATE_WSTRIDE=500*10  
        opes: OPES_METAD ARG=tica_cv1_"""+str(i-1)+""",tica_cv2_"""+str(i-1)+""" TEMP=300 BIASFACTOR=3 PACE=500 RESTART=NO FILE=KERNELS BARRIER="""+str(BARRIER)+""" STATE_WFILE=RestartKernels STATE_WSTRIDE=500*10  

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
        if i==1:
            execute("cp ../alanine.log ../alanine.xtc ../alanine.edr ../state.cpt .",folder=folder, print_result=False)
        elif i<10:
            execute("cp ../alanine.part000"+str(i)+".log ../alanine.part000"+str(i)+".xtc ../alanine.part000"+str(i)+".edr ../state.cpt .",folder=folder, print_result=False)
        elif i>=10:
            execute("cp ../alanine.part00"+str(i)+".log ../alanine.part00"+str(i)+".xtc ../alanine.part00"+str(i)+".edr ../state.cpt .",folder=folder, print_result=False) 

    run_gromacs = execute("./run_gromacs.sh",folder=folder)

    #-- TRAIN --#
    data = load_dataframe(folder+"COLVAR")

    '''
    #-- autocorrelation length --#
    x = np.linspace(0,last+1,last)
    acorr = np.empty(last)
    timescale = np.empty(len(descriptors_names))
    for k,desc in enumerate(descriptors_names):
        for j in range(last):
            acorr[j] = data[desc].autocorr(j)
        timescale[k] = integrate.trapz(acorr[:last],x[:last])
    times = pd.DataFrame(descriptors_names,columns=["descriptors"])
    times["timescale"] = timescale
    new_correlation_length = np.max(timescale)
    print("iteration ",i , ",max correlation length ", new_correlation_length)
    #if increase_factor*correlation_length < new_correlation_length:
    #if thershold < new_correlation_length:
    #    print("### transition ? ###", new_correlation_length, "\t", thershold)
    #    break
    #else:
    #    correlation_length = new_correlation_length
    '''

    #subset of data if necessary
    data = data[::train_parameters["step"]]

    t = data['time'].values
    X = data[descriptors_names].values
    
    # MODEL
    model = TICA_CV(n_features=X.shape[1])
    model.to(device)

    # TRAIN
    model.fit(X, t, lag=lag_time)

    #-- move the model back to cpu for convenience --# 
    model.to('cpu')

    #-- print some useful results --#
    #print("timescales: ",model.tica.timescales(train_parameters["lag_time"]).detach().cpu().numpy()) 
    print("eigenvalues: ",model.tica.evals_.detach().cpu().numpy())
    #print("gap: ", model.tica.evals_.detach().cpu().numpy()[0]-model.tica.evals_.detach().cpu().numpy()[1])

    model.set_params({"feature_names": names})
    #print( model.plumed_input().splitlines()[:2][8:] )

    
    data["cv1"] = np.transpose(model(torch.Tensor(X)).detach().cpu().numpy())[0]
    data["cv2"] = np.transpose(model(torch.Tensor(X)).detach().cpu().numpy())[1]

    #-- autocorrelation length of new cvs --#
    x = np.linspace(0,last+1,last)
    acorr = np.empty(last)
    timescale = np.empty(len(["cv1","cv2"]))
    for k,desc in enumerate(["cv1","cv2"]):
        for j in range(last):
            acorr[j] = data[desc].autocorr(j)
        timescale[k] = integrate.trapz(acorr[:last],x[:last])
    times = pd.DataFrame(["cv1","cv2"],columns=["descriptors"])
    times["timescale"] = timescale
    new_correlation_length_cv = np.max(timescale)
    print("iteration: ",i,", correlation length of cvs ", new_correlation_length_cv)
    #if increase_factor*correlation_length < new_correlation_length:
    if thershold < new_correlation_length_cv:
        print("### transition ? ###", new_correlation_length_cv, "\t", thershold)
        break
    else:
        correlation_length_cv = new_correlation_length_cv
    

print("###--- End Simulations ---###")
print("total time of simulations: ", iterations*Time, " ns")
end = time.time()
print("Run for: ",end - start, " s")

