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
    'temp':340, 
    'beta': 1./(340*kb),
    'kbt': None,
    #-- parameters to compute the fes --#
    'blocks':2,
    'bandwidth': 0.02,
    'plot_max_fes' :70,
}

BARRIER = 40 # barrier parameter for OPES  
STRIDE = 100 # stride of the simulation, usually 100 which means 1/5 ps
iterations = 10 # number of iterations, I have not decided yet which criterion to stop the iterations
dt = 0.000002 # time step of simulation, in nanoseconds
Time = 10 # in nanoseconds, time of single simulation
size = (Time/dt)/STRIDE # total sampled points for each simulation
restart = True # if restart simulation
#-- minimum and maximum lag time --#
lag_time = 1 

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

#-- TRAINING PARAMETERS --#
train_parameters = {
              'descriptors': '^dd[^a-z]',
              'lag_time':1,
              'standardize_outputs':True,
              'standardize_inputs': True,
              #if reweight the timescale
              "reweighting": False,
              "step": 1 #
              }

#subset of data if necessary
data = data[::train_parameters["step"]]
# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

names = data.filter(regex='^dd[^a-z]').columns.values
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

for i in range(1,iterations):

    print("ITERATION NUMBER ", i)

    # start bias simulation    
    folder += "bias"+str(i)+"/"
    Path(folder).mkdir(parents=True, exist_ok=True)

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
        
        # define cv
        tica_cv1_"""
        +str(i-1)+model.plumed_input().splitlines()[0][8:]+"""\ntica_cv2_"""+str(i-1)+model.plumed_input().splitlines()[1][8:]
        +"""
        
        opes: OPES_METAD ARG=tica_cv1_"""+str(i-1)+""",tica_cv2_"""+str(i-1)+""" TEMP=340 BIASFACTOR=4 PACE=500 RESTART=NO FILE=KERNELS BARRIER="""+str(BARRIER)+""" STATE_WFILE=RestartKernels STATE_WSTRIDE=500*10  

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

    #-- TRAIN --#
    data = load_dataframe(folder+"COLVAR")
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

print("###--- End Simulations ---###")
print("total time of simulations: ", iterations*Time, " ns")
end = time.time()
print("Run for: ",end - start, " s")

