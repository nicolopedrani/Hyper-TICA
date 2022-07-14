import numpy as np

#------ Prepare Input Files for Toy Model -----------#

#it works with BF_POWERS
def generate_input_file(name_file="input",nstep=10000,temp=1.0,friction=10,random_seed=4245,output_potential_grid=150,dim=2,ORDER=[4,4],Range=[[-3.0,3.0],[-3.0,3.0]],initial_position=[-1.174,+1.477]):
    with open(name_file,"w") as f:
        print("nstep\t"+str(nstep), file=f)
        print("tstep\t0.005", file=f)
        print("temperature\t"+str(temp), file=f)
        print("friction\t"+str(friction), file=f)
        print("random_seed\t"+str(random_seed), file=f)
        print("plumed_input\tplumed.dat", file=f)
        print("dimension\t"+str(dim),file=f)
        print("replicas\t1",file=f)
        for i in range(1,dim+1):
            print("basis_functions_%d BF_POWERS ORDER=" % (i) +str(ORDER[i-1])+" MINIMUM="+str(Range[i-1][0])+" MAXIMUM="+str(Range[i-1][1]),file=f)
        print("input_coeffs\tpot_coeffs_input.data",file=f)
        print("initial_position\t",end="",file=f)
        print(*initial_position, sep = ", ",file=f)
        print("output_potential\tpotential.data",file=f)
        print("output_potential_grid\t"+str(output_potential_grid),file=f)
        print("output_histogram\thistogram.data",file=f)
                
#it needs the path to the input file, ex: "input"
#returns - dimension of the system, 
#        - order of each coordinate, 
#        - name of the output file on to write coeff values
def initialization(input_file):
    dim=0
    order=[]
    out_file=""
    with open(input_file,"r") as f:
        for line in f:
            words = line.split()
            if words==[]:
                continue
            else:
                if words[0]=="dimension":
                    dim=int(words[1])
                for i in range(1,dim+1):
                    if words[0]=="basis_functions_"+str(i):
                        #adding 0th order to the all dimensions
                        order.append(int(words[2][words[2].find("=")+1:])+1)
                if words[0]=="input_coeffs":
                    out_file=words[1]

    return dim,order,out_file

#Muller Potential 
def Mullerpot():
    potential="("
    A=["-200","-100","-170","+15"]
    b=["+0","+0","+11","+0.6"]
    x0=["-1","-0","+0.5","+1"]
    a=["-1","-1","-6.5","0.7"]
    c=["-10","-10","-6.5","+0.7"]
    y0=["-0","-0.5","-1.5","-1"]
    for k in range(4):
        potential+= A[k]+"*exp("+a[k]+"*(x"+x0[k]+")^2"+b[k]+"*(x"+x0[k]+")*(y"+y0[k]+")"+c[k]+"*(y"+y0[k]+")^2)"
    potential+=")/10"
    return potential

#Muller potential function
def Mullerfunction(x,y):
    a = -200*np.exp(-1*(x-1)**2+0*(x-1)*(y-0)-10*(y-0)**2)-100*np.exp(-1*(x-0)**2+0*(x-0)*(y-0.5)-10*(y-0.5)**2)-170*np.exp(-6.5*(x+0.5)**2+11*(x+0.5)*(y-1.5)-6.5*(y-1.5)**2)+15*np.exp(0.7*(x+1)**2+0.6*(x+1)*(y-1)+0.7*(y-1)**2)
    return a/10

#Wolfe-Quapp potential
def WQ():
    return "x^4+y^4-2*x^2-4*y^2+x*y+0.3*x+y"

#Modified Wolfe-Quapp potential
def WQ_MODIFIED():
    return "1.34549*x^4+1.34549*y^4+1.90211*x^3*y+3.92705*x^2*y^2-6.44246*x^2-1.90211*x*y^3-5.55754*y^2+5.58721*x*y+1.33481*x+0.904586*y+18.5598"

#Modified Wolfe-Quapp potential function
def WQMfunction(x,y):
    return 1.34549*x**4+1.34549*y**4+1.90211*x**3*y+3.92705*x**2*y**2-6.44246*x**2-1.90211*x*y**3-5.55754*y**2+5.58721*x*y+1.33481*x+0.904586*y+18.5598

#1 dimensional potential
def potential1d():
    return "3.5*(4*(x*x*x-1.5*x)^2-x*x*x+x)"

def potential1d_function(x):
    return 3.5*(4*(x*x*x-1.5*x)*(x*x*x-1.5*x)-x*x*x+x)

#-- example plumed input file --#

'''
write_coeff("0","input")
generate_input_file(nstep=nstep,temp=temp,friction=friction,random_seed=plumedseed,initial_position=[+0.5,+0])

with open('unbias/plumed.dat', 'w') as f:
    print("p: POSITION ATOM=1",file=f)
    print("potential: CUSTOM ARG=p.x,p.y FUNC="+Mullerpot()+" PERIODIC=NO",file=f)
    print("newene: BIASVALUE ARG=potential",file=f)
    print("ene: CUSTOM ARG=newene.bias FUNC=x",file=f)
    print("PRINT ARG=p.x,p.y,ene STRIDE="+STRIDE+" FILE=colvar.data FMT=%8.4f",file=f)
'''

#it needs the form of the potential with the form above
def evaluate_coeff(U,dim,order):
    Pot = np.array(U.split()[::-1])
    s = set()
    if dim==2:
        #X powers
        for x in range(order[0]):
            indx = np.where(Pot=="x^"+str(x))[0]
            if len(indx)>0:
                varx = "{:.16e}".format(float(Pot[indx+1][0]))
                index = x
                s.add(str(x)+"\t0\t"+varx+"\t"+str(index)+"\ts^"+str(x)+"*1")
            #XY and Y powers    
            for y in range(order[1]):
                #Y powers
                indy = np.where(Pot=="y^"+str(y))[0]
                if len(indy)>0:
                    vary = "{:.16e}".format(float(Pot[indy+1][0]))
                    index = (y)*order[0]
                    s.add("0\t"+str(y)+"\t"+vary+"\t"+str(index)+"\t1*s^"+str(y))
                #XY powers
                indxy = np.where(Pot=="x^"+str(x)+"y^"+str(y))[0]
                if len(indxy)>0:
                    varxy = "{:.16e}".format(float(Pot[indxy+1][0]))
                    index = x+y*order[1]
                    s.add(str(x)+"\t"+str(y)+"\t"+varxy+"\t"+str(index)+"\ts^"+str(x)+"s^"+str(y))
    else:
        #X powers
        for x in range(order[0]):
            indx = np.where(Pot=="x^"+str(x))[0]
            if len(indx)>0:
                varx = "{:.16e}".format(float(Pot[indx+1][0]))
                index = x
                s.add(str(x)+"\t0\t"+varx+"\t"+str(index)+"\ts^"+str(x)+"*1")
    s = list(s)
    return s;

def write_coeff(U,input_file):
    dim,order,out_file=initialization(input_file)
    s=evaluate_coeff(U,dim,order)
    with open(out_file,"w") as f:
        #header
        print("#! FIELDS ",end="", file=f)
        for i in range(1,dim+1):
            print("idx_dim%d " % (i),end="",file=f)
        print("pot.coeffs index description\n",end="", file=f)
        print("#! SET type LinearBasisSet\n#! SET ndimensions  "+str(dim),file=f)
        print("#! SET ncoeffs_total  "+str(np.prod(order)),file=f)
        for i in range(1,dim+1):
            print("#! SET shape_dim%d %d" % (i,order[i-1]),file=f)
        #coefficients
        for el in s:
            print(el,file=f)
        #last line
        print("#!-------------------",file=f)

#------ Prepare Input Files for Toy Model -----------#

#write_coeff("0","input")
#WQ
#write_coeff("+1.0 x^4 +1.0 y^4 -2.0 x^2 -4.0 y^2 +1 x^1y^1 +0.3 x^1 +0.1 y^1 +0.0 x^0y^0","input")
#WQ modified
#write_coeff("+1.34549 x^4 +1.34549 y^4 +1.90211 x^3y^1 +3.92705 x^2y^2 -6.44246 x^2 -1.90211 x^1y^3 -5.55754 y^2 +5.58721 x^1y^1 +1.33481 x^1 +0.904586 y^1 +18.5598 x^0y^0"
#            ,"input")

#generate_input_file(nstep=nstep,temp=temp,friction=friction,random_seed=plumedseed,initial_position=[0.8,0])