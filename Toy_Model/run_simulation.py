#-- to run process from jupyter --#
import subprocess
import os
from pathlib import Path
# execute bash command in the given folder
def execute(command, folder, background=False, print_result=True):
    cmd = subprocess.run(command, cwd=folder, shell=True, capture_output = True, text=True, close_fds=background)
    if cmd.returncode == 0:
        if print_result:
            print(f'Completed: {command}')
    else:
        print(cmd.stderr)

#-- python script for write correct input files for "ves_md_linearexpansion" plumed module --#
from input_VES import *

#-- to not visualize warnings --#
import warnings
warnings.filterwarnings('ignore')

sim_parameters = {
    'nstep':100000000, 
    'plumedseed':4525,
    'friction':10,
    'temp':0.5, #kbt units
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

import sys
if len(sys.argv)!=3:
    print("please insert the lag time and the seed")
    sys.exit("error")

#lag -> sys.argv[1]
#seed -> sys.argv[2]

folder = "test_different_seeds/lag"+sys.argv[1]+"/bias1_seed"+sys.argv[2]+"/"

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

# definition of Deep-TICA cvs 
deep: PYTORCH_MODEL FILE=../deeptica_seed"""+sys.argv[2]+"""/model.ptc ARG=p.x,p.y

# Bias 
opes: OPES_METAD ARG=deep.node-0,deep.node-1 TEMP=0.5 PACE=500 FILE=KERNELS BARRIER=7.5 STATE_WFILE=RestartKernels STATE_WSTRIDE=500*10

# Print 
# STRIDE=200 so that the printed time is in 1 ps
PRINT FMT=%g STRIDE=200 FILE=COLVAR ARG=deep.node-0,deep.node-1,p.x,p.y,ene.bias,opes.*

ENDPLUMED
""",file=file)

#-- write input files for ves module --#
generate_input_file(name_file=folder+"input",nstep=sim_parameters["nstep"],temp=sim_parameters["temp"],
                    friction=sim_parameters["friction"],random_seed=sim_parameters["plumedseed"],
                    initial_position=sim_parameters["initial_position"])
write_coeff("0",folder+"input")

#-- move necessary files for ves module --#
execute("mv pot_coeffs_input.data "+folder,folder=".")
#-- run plumed --#
execute("plumed ves_md_linearexpansion input",folder=folder)