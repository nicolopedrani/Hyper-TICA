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
    'nstep':1000000000, 
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

STRIDE = 10000 # every 50 ps
dt = 0.000005 # time step of simulation, in nanoseconds
Time = (dt*sim_parameters["nstep"]) # in nanoseconds, time of single simulation
size = (sim_parameters["nstep"])/STRIDE # total sampled points for each simulation

print("###--- PARAMETERS ---###")
print("print points every ", STRIDE, " steps")
print("each iteration lasts ", Time, " ns")
print("sample size: ",int(size) )

# start with unbias simulation
folder = "long_unbias/"
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

print("###--- End Simulations ---###")
print("total time of simulations: ", Time, " ns")
end = time.time()
print("Run for: ",end - start, " s")