from utils import *

#-- to not visualize warnings --#
import warnings
warnings.filterwarnings('ignore')

#-- SIMULATION PARAMETERS --#
temp= 0.5, #kbt units

#-- TRAINING PARAMETERS --#
n_output = 2
n_input = 2
train_parameters = {
              'descriptors': '^p.',
              'nodes':[n_input,10,n_output], 
              'activ_type': 'tanh',
              'lag_time':1,
              'loss_type': 'sum', 
              'n_eig': n_output,
              'trainsize':0.7,
              'lrate':1e-3,
              'l2_reg':0.,
              'num_epochs':1000,
              'earlystop':True,
              'es_patience':100,
              'es_consecutive':False,
              'standardize_outputs':True,
              'standardize_inputs': True,
              'log_every':100,
              #if reweight the timescale
              "reweighting": False,
              }

#-- prepare grid --#
points = 150
limits=((-1.8,1.2),(-0.3,2.1))
xx,yy = np.meshgrid(np.linspace(limits[0][0],limits[0][1],points),np.linspace(limits[1][0],limits[1][1],points))
grid = np.transpose(np.array([xx.reshape(points*points),yy.reshape(points*points)]))

#-- to evaluate mean and variance, cv1 and cv2 --#
sum1 = np.zeros((points,points))
sum2 = np.zeros((points,points))
sum1square = np.zeros((points,points))
sum2square = np.zeros((points,points))

#-- different seeds --#
n_seeds = 50
np.random.seed(13)
seeds = np.random.randint(5000, size=n_seeds)

for seed in seeds:

    print("training with seed: ",seed)
    #fix seed (random split)
    torch.manual_seed(seed)
    model,data,logweight,X = training(temp,"unbias/COLVAR",train_parameters)
    model.to("cpu")

    #-- evaluate cvs on the grid --#
    cvs = []
    for i in range(n_output):
        cvs.append(np.transpose(model(torch.Tensor(grid)).detach().cpu().numpy())[i].reshape(points,points))
    
    #fixing cv signum: cv1 Negative at the bottom right corner
    if model( torch.Tensor([0.5,0.0]) )[0] > 0:
        cvs[0]-=cvs[0]
    # cv2 Positive at the bottom left corner
    if model( torch.Tensor([-1.5,0.0]) )[1] < 0:
        cvs[1]-=cvs[1]
        
    #sum and sum square for the mean and the variance calculation
    sum1+=cvs[0]
    sum2+=cvs[1]
    sum1square+=cvs[0]*cvs[0]
    sum2square+=cvs[1]*cvs[1]

# evaluation of the mean <x>
mean1 = sum1/n_seeds
mean2 = sum2/n_seeds   
# evaluation of the variance <x*x> - <x><x> and standard deviation 
var1 = sum1square/n_seeds - mean1*mean1
var2 = sum2square/n_seeds - mean2*mean2
std1 = np.sqrt( var1 )
std2 = np.sqrt( var2 )

# save results
np.savetxt("mean1.txt",mean1)
np.savetxt("mean2.txt",mean2)
np.savetxt("std1.txt",std1)
np.savetxt("std2.txt",std2)
np.savetxt("var1.txt",var1)
np.savetxt("var2.txt",var2)
