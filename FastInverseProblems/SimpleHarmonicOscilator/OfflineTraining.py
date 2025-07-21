#For code with everything in a single file, see FastInverseProblems.py
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
import copy

from scipy.integrate import solve_ivp

import random

from time import perf_counter
from contextlib import contextmanager


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
        self.parameterSet1 = nn.Parameter(torch.randn(1, output_size))
        self.parameterSet2 = nn.Parameter(torch.randn(1, output_size))

    def forward(self, x):
        x = self.output_layer(x)
        return x

def newPdeResidualLoss(ResOut, ResFirstDeriv, ResSecondDeriv,FirstOrderCoeffs,ZeroOrderCoeffs,Forcings, outmodel):
    W = outmodel.output_layer.weight  # shape: [D, H]
    b = outmodel.output_layer.bias    # shape: [D]

    u = (ResOut @  W.T) + b # shape [N, D]
    #u += b

    # du/dt = W @ dR/dt
    du = (W @ ResFirstDeriv.T).T  # shape [N, D]

    # d^2u/dt^2 = W @ d^2R/dt^2
    d2u = (W @ ResSecondDeriv.T).T  # shape [N, D]

    residual = d2u + FirstOrderCoeffs * du + ZeroOrderCoeffs * u - Forcings # shape [N, D]
    return torch.mean(residual.pow(2))

def trainFullNetworkWithPrecomputing(Reservoir,outmodel,numoutputs,ICs,colocationPoints,ODEWeight,numEpochs,loss_fn,lr,averageLossOverTime,device,verbose = False):

    coefficients1 = (torch.rand((1, numoutputs)) * 1.5).to(device)
    coefficients2 = (torch.rand((1, numoutputs)) * 1.5).to(device)
    forcing = (torch.rand((1, numoutputs)) * 3 - 1.5).to(device)

    '''
    forcingIndecies = torch.randint(0,3,(1,numoutputs))

    forcingsOvertime = torch.zeros((colocationPoints.size()[0],numoutputs))
    for i in range(colocationPoints.size()[0]):
        for j in range(numoutputs):
            forcingsOvertime[i,j] = indexToForcing(forcingIndecies[0,j],colocationPoints[i,0])
    '''
    
    #dateching so that the computational graph does not include these matricies giving us an error for using them in multiple calls to .backwards()
    coefficients1 = coefficients1.detach()
    coefficients2 = coefficients2.detach()
    forcing = forcing.detach()

    #initialising optimizer
    FullPINNOptimizer = optim.Adam(list(Reservoir.parameters()) + list(outmodel.parameters()), lr=lr)

    #scheduler = scheduler = lr_scheduler.LinearLR(FullPINNOptimizer, start_factor=1.0, end_factor=0.05, total_iters=1000)
    
    loss = (torch.ones(1)*99999999).to(device)

    zero = torch.zeros((1,1),dtype=torch.float32,requires_grad=True).to(device)

    bestRes = copy.deepcopy(Reservoir)
    bestLoss = loss.item()

    for epoch in range(numEpochs):

        FullPINNOptimizer.zero_grad()

        zeroOut = outmodel(Reservoir(zero))

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

        #using evaluation points as colocation points
        ODEloss = newPdeResidualLoss(ResOutOverEvaluationPoints, ReservoirFirstDerivative,ReservoirSecondDerivative,coefficients1,coefficients2,forcing,outmodel)

        ICloss = loss_fn(zeroOut,ICs[:,0].reshape(1,-1))

        #du0/dt (assuming the first input is t = 0)
        du0 = ((outmodel.output_layer.weight) @ ReservoirFirstDerivative[0,:].reshape(-1,1)).T
        
        ICloss += loss_fn(du0,ICs[:,1].reshape(1, -1))

        loss = ODEWeight*ODEloss + ICloss

        averageLossOverTime.append(loss.item())
                
        #using backpropogation to get the derivitive of the parameters in the network with respect to the loss
        loss.backward()
        #taking a step in the direction of negative slope according to our optimiser
        FullPINNOptimizer.step() 
        #scheduler.step()

        if (loss.item() < bestLoss):
            bestRes = copy.deepcopy(Reservoir)
            bestLoss = loss.item()

        if verbose:
            if epoch % 1000 == 0:
                print(f"-------------------------------\nEpoch {epoch}")
                print(f"Loss: {loss.item()}")
                print(f"ODEL = {ODEloss}")
                print(f"ICL = {ICloss}")

    print(f"Best Loss: {bestLoss}")

    return (bestRes, averageLossOverTime)

@contextmanager
def timer(name="Block"):
    start = perf_counter()
    yield
    end = perf_counter()
    print(f"[{name}] Elapsed time: {end - start:.6f} seconds")

print("Offline training Start!")
with timer("Training Loop"):
    torch.manual_seed(42)
    #assigning device as done in tutorial
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resWidth = 40

    nummodels = 30


    diameter = 10
    center = 0
    ICs = (torch.rand((nummodels,2),dtype=torch.float32)*diameter - (diameter/2) + center).to(device)

    final = 3
    initial = 0
    numevals = 30
    colocationPoints = torch.linspace(initial,final,numevals,dtype=torch.float32,requires_grad=True).reshape(-1,1).to(device)

    #input and hidden layers
    Reservoir = MLPWithoutOutput(1,resWidth,1,4).to(device)

    #initilizing loss function
    loss_fn = nn.MSELoss().to(device)

    #setting the number of training epochs
    trainingEpochs = 20000
    trainlr = 1e-3

    averageLossOverTime = []

    ODEWeight = 1

    outmodel = MLPOutput(resWidth,nummodels).to(device)
    Reservoir, averageLossOverTime = trainFullNetworkWithPrecomputing(Reservoir,outmodel,nummodels,ICs,colocationPoints,ODEWeight,trainingEpochs,loss_fn,trainlr,averageLossOverTime,device,verbose=True)

logloss = torch.log(torch.tensor(averageLossOverTime))
plt.plot(logloss.detach().numpy(),label="Log Train Loss")
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('loss')
plt.title('Training Log Average Loss Of The 30 Readout Layers')
plt.grid(True)
plt.savefig('Training Log Average Loss Of The 30 Readout Layers.png')

plt.close()

final = 5
initial = 0
numevals = 1000
colocationPoints = torch.linspace(initial,final,numevals,dtype=torch.float32,requires_grad=True).reshape(-1,1)

modelOut = outmodel(Reservoir(colocationPoints)).detach().numpy()
for i in range(nummodels):
    plt.plot(colocationPoints.detach(), modelOut[:,i], label='modelOut(t)')
#plt.plot(solution.t, modelOut[:,0], label='modelOut(t)')
#plt.plot(solution.t, solution.y[0], label='X(t)')
plt.xlabel('Time')
plt.ylabel('Values')

plt.title('Training Solutions of the Differential Equations')
#plt.legend()
plt.grid(True)
plt.savefig('Training Solutions of the Differential Equations.png')

plt.close()

torch.save(Reservoir.state_dict(), "Reservoir2.pt")


