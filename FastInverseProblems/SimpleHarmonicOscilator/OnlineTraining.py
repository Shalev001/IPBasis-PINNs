from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # use a backend that doesn't need a display
import matplotlib.pyplot as plt
import math

from scipy.integrate import solve_ivp

import random

from time import perf_counter
from contextlib import contextmanager

from fomoh.hyperdual import HyperTensor as htorch
from fomoh.nn import DenseModel, nll_loss
from fomoh.nn_models_torch import DenseModel_Torch

import copy

import wandb

#Code taken and modified from tutorial
class MLPWithoutOutput(nn.Module):
    """
     A simple feedforward neural network with multiple hidden layers.
    """
    def __init__(self, input_size, hidden_size, output_size, n_hidden_layers):
        super(MLPWithoutOutput, self).__init__()
        self.input_layer = nn.Sequential(nn.Linear(input_size, hidden_size,dtype=torch.float32), nn.Tanh())
        self.hidden_layers = self._make_hidden_layers(n_hidden_layers, hidden_size)

    def _make_hidden_layers(self, n_hidden_layers, hidden_size):
        layers = []
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(hidden_size, hidden_size,dtype=torch.float32), nn.Tanh()]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        return x

class MLPOutput(nn.Module):
    """
     A simple feedforward neural network with multiple hidden layers.
    """
    def __init__(self, hidden_size, output_size):
        super(MLPOutput, self).__init__()
        self.output_layer = nn.Linear(hidden_size, output_size,dtype=torch.float32)
        #initializing a matrix to store the predicted parameters and forcing terms
        self.parameterSet = nn.Parameter(torch.randn(3, output_size))

    def forward(self, x):
        x = self.output_layer(x)
        return x

def newPdeResidualLossForInverseProblem(ResOut, ResFirstDeriv, ResSecondDeriv, outmodel):
    W = outmodel.output_layer.weight  # shape: [D, H]
    b = outmodel.output_layer.bias    # shape: [D]

    FirstOrderCoeffs = outmodel.parameterSet[0,:] # shape: [1, D]
    ZeroOrderCoeffs = outmodel.parameterSet[1,:] # shape: [1, D]
    Forcings = outmodel.parameterSet[2,:] # shape: [1, D]

    u = (ResOut @  W.T) + b # shape [N, D]
    #u += b

    # du/dt = W @ dR/dt
    du = (W @ ResFirstDeriv.T).T  # shape [N, D]

    # d^2u/dt^2 = W @ d^2R/dt^2
    d2u = (W @ ResSecondDeriv.T).T  # shape [N, D]

    residual = d2u + FirstOrderCoeffs * du + ZeroOrderCoeffs * u + Forcings # shape [N, D]
    return torch.mean(residual.pow(2))

#Method to evaluate network output derivatives using forward mode auto-differentiation 
def computeDerivatives(TorchReservoir,HyperTensReservoir,evalPts):

    HyperTensReservoir.nn_module_to_htorch_model(TorchReservoir,verbose=False)

    #initializing tangent vector
    v = torch.ones_like(evalPts)
    #initializing network input
    x_h = htorch(evalPts,v,v)
    #computing output
    y_h = HyperTensReservoir(x_h,None,requires_grad=True)
    #extracting the calculated first and second derivatives
    resOut = y_h.real
    firstDer = y_h.eps1
    secondDer = y_h.eps1eps2

    return (resOut,firstDer,secondDer)

def trainOutput(Reservoir,HyperTensReservoir,outmodel,data,dataTimes,ICs,colocationPoints,ODEWeight,numEpochs,loss_fn,lr,averageLossOverTime,device,verbose = False):
    #initialising opdimise
    outputOptimizer = optim.Adam(outmodel.parameters(), lr=lr)

    scheduler = scheduler = lr_scheduler.LinearLR(outputOptimizer, start_factor=1.0, end_factor=0.1, total_iters=1)
    
    loss = torch.ones(1)*99999999

    zero = torch.zeros((1,1),dtype=torch.float32,requires_grad=True)

    #We compute the output of the reservoir ahead of time since it stays constant throughout training and would thus be a waste to recompute every epoch
    resZero = Reservoir(zero).detach()

    
    ResOutOverDataPoints = Reservoir(dataTimes).detach()
    
    '''
    ResOutOverEvaluationPoints = Reservoir(colocationPoints)
    firstDerivatives = []
    secondDerivatives = []

    for i in range(ResOutOverEvaluationPoints.shape[1]):
        output = ResOutOverEvaluationPoints[:, i]  # shape: [N]
        grad1 = torch.autograd.grad(
            output, colocationPoints,
            grad_outputs=torch.ones_like(output),
            create_graph=True, retain_graph=True
        )[0]  # [N, input_dim]

        grad2 = torch.autograd.grad(
            grad1, colocationPoints,
            grad_outputs=torch.ones_like(grad1),
            create_graph=True, retain_graph=True
        )[0]

        firstDerivatives.append(grad1)
        secondDerivatives.append(grad2) 

    ReservoirFirstDerivative = torch.cat(firstDerivatives, dim=1) # [N, D]
    ReservoirSecondDerivative = torch.cat(secondDerivatives, dim=1)  # [N, D]
    '''

    ResOutOverEvaluationPoints, ReservoirFirstDerivative, ReservoirSecondDerivative = computeDerivatives(Reservoir,HyperTensReservoir,colocationPoints)

    ResOutOverEvaluationPoints = ResOutOverEvaluationPoints.detach()
    ReservoirFirstDerivative = ReservoirFirstDerivative.detach()
    ReservoirSecondDerivative = ReservoirSecondDerivative.detach()

    W = outmodel.output_layer.weight

    previouseParams = 0

    for epoch in range(numEpochs):

        outputOptimizer.zero_grad()

        #using evaluation points as colocation points
        ODEloss = newPdeResidualLossForInverseProblem(ResOutOverEvaluationPoints, ReservoirFirstDerivative,ReservoirSecondDerivative,outmodel)

        output = outmodel(ResOutOverDataPoints)
        
        dataloss = loss_fn(output,data)

        loss = ODEWeight*ODEloss + dataloss 

        averageLossOverTime.append(loss.item())

        if verbose:
            if epoch % 100 == 0:
                print(f"-------------------------------\nEpoch {epoch}")
                print(f"Average Loss: {loss.item()}")
                print(f"ODEL = {ODEloss}")
                print(f"DataL = {dataloss}")
                
        #using backpropogation to get the derivitive of the parameters in the network with respect to the loss
        loss.backward()
        #taking a step in the direction of negative slope according to our optimiser
        outputOptimizer.step() 
        if epoch == 5000:
            scheduler.step()

        wandb.log({"Log Loss": math.log(loss.item()),"Log ODE Loss": math.log(ODEloss),"Log Data Loss": math.log(dataloss)})

    print(f"Average Loss: {loss.item()}")

    return (outmodel, averageLossOverTime)

@contextmanager
def timer(name="Block"):
    start = perf_counter()
    yield
    end = perf_counter()
    print(f"[{name}] Elapsed time: {end - start:.6f} seconds")

def fprime(t,x,a1,a2,a3):
  return [x[1],-a2*x[0] - a1*x[1] - a3]


torch.manual_seed(43)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resWidth = 40

#Reservoir = MLPWithoutOutput(1,resWidth,1,4).to(device)
Reservoir = DenseModel_Torch(layers=[1,resWidth,resWidth,resWidth,resWidth],activation=nn.Tanh()).to(device)
HyperTensReservoir = DenseModel(layers=[1,resWidth,resWidth,resWidth,resWidth])
HyperTensReservoir.to(device)

Reservoir.load_state_dict(torch.load("Reservoir50.pt",weights_only=True))
Reservoir.eval()

#initilizing loss function
loss_fn = nn.MSELoss().to(device)

nummodels = 1000

diameter = 80
center = 0
ICs = torch.rand((nummodels,2),dtype=torch.float32)*diameter - (diameter/2) + center

coefficients1 = torch.rand((1, nummodels)) * 1.5
coefficients2 = torch.rand((1, nummodels)) * 1.5
forcing = torch.rand((1, nummodels)) * 3 - 1.5

'''coefficients1[0,0] = 1.4621
coefficients2[0,0] = 0.1607
forcing[0,0] = 0.4844'''

#dateching so that the computational graph does not include these matricies giving us an error for using them in multiple calls to .backwards()
coefficients1 = coefficients1.detach()
coefficients2 = coefficients2.detach()
forcing = forcing.detach()

final = 3
initial = 0
numevals = 100
evalpts = np.linspace(initial,final,numevals)

solutions = []

for i in range(nummodels):
    solution_i = solve_ivp(fprime,(initial,final),[ICs[i,0],ICs[i,1]],t_eval=evalpts,args=(coefficients1[0,i],coefficients2[0,i],forcing[0,i]))
    solutions.append(torch.tensor(solution_i.y[0],dtype=torch.float32).reshape(-1,1))

data = torch.cat(solutions,dim=1)

evalpts = torch.tensor(evalpts,dtype=torch.float32,requires_grad=True).reshape(-1,1)

final = 3
initial = 0
numevals = 30
colocationPoints = torch.linspace(initial,final,numevals,dtype=torch.float32,requires_grad=True).reshape(-1,1).to(device)

#setting the number of training epochs
trainingEpochs = 15000
trainlr = 5e-2

averageLossOverTime = []


ODEWeight = 0.001

wandb.init(
      # Set the project where this run will be logged
      project="Damped Harmonic Oscilator (Online Training)",
      # Track hyperparameters and run metadata
      config={
      "learning_rate": trainlr,
      "epochs": trainingEpochs,
      "ODE_weight": ODEWeight,
      "resWidth": resWidth,
      "number of output models": nummodels
      })

outmodel = MLPOutput(resWidth,nummodels).to(device)

print("online training Start!")
with timer("Online Training"):
    outmodel, averageLossOverTime = trainOutput(Reservoir,HyperTensReservoir,outmodel,data,evalpts,ICs,colocationPoints,ODEWeight,trainingEpochs,loss_fn,trainlr,averageLossOverTime,device,verbose = False)
wandb.finish()

print(f"mean squared error on 1st parameters = {torch.mean(torch.square(coefficients1 - outmodel.parameterSet[0,:]))}")
print(f"mean error on 1st parameters = {torch.mean(torch.abs(coefficients1 - outmodel.parameterSet[0,:]))}")
print(f"mean squared error on 2nd parameters = {torch.mean(torch.square(coefficients2 - outmodel.parameterSet[1,:]))}")
print(f"mean error on 2nd parameters = {torch.mean(torch.abs(coefficients2 - outmodel.parameterSet[1,:]))}")

#plt.plot(averageLossOverTime)
logloss = torch.log(torch.tensor(averageLossOverTime))
plt.plot(logloss.detach().numpy())
plt.xlabel('Iteration')
plt.ylabel('loss')
plt.title('Online Training Log Average Loss Of The 30 Readout Layers')
plt.grid(True)
plt.savefig('Online Training Log Average Loss Of The 30 Readout Layers.png')

plt.close()

final = 3
initial = 0
numevals = 1000
colocationPoints = torch.linspace(initial,final,numevals,dtype=torch.float32,requires_grad=True).reshape(-1,1)

modelOut = outmodel(Reservoir(colocationPoints)).detach().numpy()
for i in range(nummodels):
    plt.plot(colocationPoints.detach(), modelOut[:,i], label=f'modelOut(t) {i}')
#plt.plot(solution.t, modelOut[:,0], label='modelOut(t)')
#plt.plot(solution.t, solution.y[0], label='X(t)')
plt.xlabel('Time')
plt.ylabel('Values')

plt.title('Online Training Solutions of the Differential Equations')
plt.legend()
plt.grid(True)
plt.savefig('Online Training Solutions of the Differential Equations.png')

plt.close()

final = 10
initial = 0
numevals = 100
evalpts = np.linspace(initial,final,numevals)
torchEval = torch.tensor(evalpts,dtype=torch.float32).reshape(-1,1)

modelOut = outmodel(Reservoir(torchEval)).detach().numpy()
for i in range(min(nummodels,5)):
    solution_i = solve_ivp(fprime,(initial,final),[ICs[i,0],ICs[i,1]],t_eval=evalpts,args=(coefficients1[0,i],coefficients2[0,i],forcing[0,i]))
    plt.plot(solution_i.t,solution_i.y[0],label=f"Numerical Solution {i}")
    plt.plot(solution_i.t, modelOut[:,i],label=f"Predicted Solution {i}")
    
#plt.plot(solution.t, modelOut[:,0], label='modelOut(t)')
#plt.plot(solution.t, solution.y[0], label='X(t)')
plt.xlabel('Time')
plt.ylabel('Values')

plt.title('Coparisons of Solution of the Differential Equation')
plt.legend()
plt.grid(True)
plt.savefig('Coparisons of Solution of the Differential Equation.png')

plt.close()