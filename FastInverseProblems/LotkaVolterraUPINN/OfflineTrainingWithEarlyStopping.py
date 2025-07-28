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
import math
import copy

from scipy.integrate import solve_ivp

import random

from time import perf_counter
from contextlib import contextmanager

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

    def forward(self, x):
        x = self.output_layer(x)
        return x

def fprime(t,x,Coeffs):
  return [Coeffs[0]*x[0]-Coeffs[1]*x[0]*x[1],-Coeffs[2]*x[1] + Coeffs[3]*x[0]*x[1]]

def LotkaVolterraResidualLoss(ResOut, ResFirstDeriv,Coeffs, outmodel,evenmask,oddmask):
    W = outmodel.output_layer.weight  # shape: [2D, H]
    b = outmodel.output_layer.bias    # shape: [2D]

    u = (ResOut @  W.T) + b # shape [N, 2D]
    #u += b

    # du/dt = W @ dR/dt
    du = (W @ ResFirstDeriv.T).T  # shape [N, 2D]

    x = u[:,evenmask] # shape [N, D]
    dx = du[:,evenmask] # shape [N, D]

    y = u[:,oddmask] # shape [N, D]
    dy = du[:,oddmask] # shape [N, D] 

    #residual = torch.mean(torch.square((Coeffs[0,:]*x - Coeffs[1,:]*x*y - dx)/(Coeffs[0,:]*x - Coeffs[1,:]*x*y + epsilon)) + torch.square((-Coeffs[2,:]*y + Coeffs[3,:]*x*y - dy)/(-Coeffs[2,:]*y + Coeffs[3,:]*x*y + epsilon)))
    residual = torch.mean(torch.square(Coeffs[0,:]*x - Coeffs[1,:]*x*y - dx) + torch.square(-Coeffs[2,:]*y + Coeffs[3,:]*x*y - dy))
    return residual

def trainFullNetworkWithPrecomputing(Reservoir,outmodel,numoutputs,ICs,colocationPoints,ODEWeight,numEpochs,loss_fn,lr,averageLossOverTime,device,verbose = False):

    evenmask = torch.tensor([(i % 2 == 0) for i in range(2*numoutputs)])
    oddmask = torch.tensor([(i % 2 != 0) for i in range(2*numoutputs)])

    coefficients = 0.5 + torch.rand((4, numoutputs))
    
    #dateching so that the computational graph does not include these matricies giving us an error for using them in multiple calls to .backwards()
    coefficients = coefficients.detach()

    solutions = []

    for i in range(nummodels):
        solution_i = solve_ivp(fprime,(initial,final),[ICs[i,0],ICs[i,1]],method='RK45',t_eval=colocationPoints.detach().reshape(-1),args=(coefficients[:,i],))
        solutions.append(torch.tensor(solution_i.y.T,dtype=torch.float32))

    data = torch.cat(solutions,dim=1)

    #initialising opdimise
    FullPINNOptimizer = optim.Adam(list(Reservoir.parameters()) + list(outmodel.parameters()), lr=lr)

    #scheduler = scheduler = lr_scheduler.LinearLR(FullPINNOptimizer, start_factor=1.0, end_factor=0.05, total_iters=1000)
    
    loss = torch.ones(1)*99999999

    zero = torch.zeros((1,1),dtype=torch.float32,requires_grad=True)

    bestRes = copy.deepcopy(Reservoir)
    bestLoss = loss.item()
    BLODE = loss.item()
    BLData = loss.item()
    BLIC = loss.item()

    for epoch in range(numEpochs):

        FullPINNOptimizer.zero_grad()

        zeroOut = outmodel(Reservoir(zero))

        ResOutOverEvaluationPoints = Reservoir(colocationPoints)
        output = outmodel(ResOutOverEvaluationPoints)

        firstDerivatives = []

        for i in range(ResOutOverEvaluationPoints.shape[1]):
            resOut = ResOutOverEvaluationPoints[:, i]  # shape: [N]
            grad1 = torch.autograd.grad(
                resOut, colocationPoints,
                grad_outputs=torch.ones_like(resOut),
                create_graph=True, retain_graph=True
            )[0]  # [N, input_dim]

            firstDerivatives.append(grad1)

        ReservoirFirstDerivative = torch.cat(firstDerivatives, dim=1) # [N, D]

        #using evaluation points as colocation points
        ODEloss = LotkaVolterraResidualLoss(ResOutOverEvaluationPoints, ReservoirFirstDerivative,coefficients, outmodel,evenmask,oddmask)

        DataLoss = loss_fn(output,data)

        x0 = zeroOut[evenmask.reshape((1,-1))]
        y0 = zeroOut[oddmask.reshape((1,-1))]
        ICloss = loss_fn(x0,ICs[:,0])
        ICloss += loss_fn(y0,ICs[:,1])

        loss = ODEWeight*ODEloss + ICloss + DataLoss

        wandb.log({"Log Loss": math.log(loss.item()),"Log ODE Loss": math.log(ODEloss),"Log IC Loss": math.log(ICloss),"Log Data Loss": math.log(DataLoss)})

        averageLossOverTime.append(loss.item())

        if (loss.item() < bestLoss):
            bestRes = copy.deepcopy(Reservoir)
            bestLoss = loss.item()
            BLODE = ODEloss.item()
            BLData = DataLoss.item()
            BLIC = ICloss.item()

        if verbose:
            if epoch % 1000 == 0:
                print(f"-------------------------------\nEpoch {epoch}")
                print(f"Average Loss: {loss.item()}")
                print(f"ODEL = {ODEloss}")
                print(f"ICL = {ICloss}")
                print(f"DataL = {DataLoss}")
                
        #using backpropogation to get the derivitive of the parameters in the network with respect to the loss
        loss.backward()
        #taking a step in the direction of negative slope according to our optimiser
        FullPINNOptimizer.step() 
        #scheduler.step()

    print(f"Best Loss: {bestLoss}")
    print(f"Best Loss ODE Loss: {BLODE}")
    print(f"Best Loss Data Loss: {BLData}")
    print(f"Best Loss IC Loss: {BLIC}")

    return (bestRes, averageLossOverTime)

@contextmanager
def timer(name="Block"):
    start = perf_counter()
    yield
    end = perf_counter()
    print(f"[{name}] Elapsed time: {end - start:.6f} seconds")

print("Offline training Start!")
with timer("Training Loop"):
    torch.manual_seed(43)
    #assigning device as done in tutorial
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resWidth = 64

    nummodels = 100

    diameter = 2
    center = 1
    ICs = torch.rand((nummodels,2),dtype=torch.float32)*diameter - (diameter/2) + center

    final = 10
    initial = 0
    numevals = 1000
    colocationPoints = torch.linspace(initial,final,numevals,dtype=torch.float32,requires_grad=True).reshape(-1,1)

    #input and hidden layers
    Reservoir = MLPWithoutOutput(1,resWidth,2,4).to(device)

    #initilizing loss function
    loss_fn = nn.MSELoss().to(device)

    #setting the number of training epochs
    trainingEpochs = 40000
    trainlr = 3e-4

    averageLossOverTime = []

    ODEWeight = 1

    wandb.init(
      # Set the project where this run will be logged
      project="LotkaVolterraUPINN(Offline Training)",
      # Track hyperparameters and run metadata
      config={
      "learning_rate": trainlr,
      "epochs": trainingEpochs,
      "ODE_weight": ODEWeight,
      "resWidth": resWidth,
      })

    outmodel = MLPOutput(resWidth,2*nummodels).to(device)
    Reservoir, averageLossOverTime = trainFullNetworkWithPrecomputing(Reservoir,outmodel,nummodels,ICs,colocationPoints,ODEWeight,trainingEpochs,loss_fn,trainlr,averageLossOverTime,device,verbose=False)

    wandb.finish()

logloss = torch.log(torch.tensor(averageLossOverTime))
plt.plot(logloss.detach().numpy())
plt.xlabel('Iteration')
plt.ylabel('loss')
plt.title('Training Log Average Loss Of The 30 Readout Layers')
plt.grid(True)
plt.savefig('Training Log Average Loss Of The 30 Readout Layers.png')
#plt.show()

plt.close()

final = 10
initial = 0
numevals = 1000
colocationPoints = torch.linspace(initial,final,numevals,dtype=torch.float32,requires_grad=False).reshape(-1,1)

modelOut = outmodel(Reservoir(colocationPoints)).detach().numpy()
for i in range(0,2*nummodels,2):
    plt.plot(modelOut[:,i], modelOut[:,i+1], label='modelOut(t)')
#plt.plot(solution.t, modelOut[:,0], label='modelOut(t)')
#plt.plot(solution.t, solution.y[0], label='X(t)')
plt.xlabel('Time')
plt.ylabel('Values')

plt.title('Training Solutions of the Differential Equations')
#plt.legend()
plt.grid(True)
plt.savefig('Training Solutions of the Differential Equations.png')
#plt.show()

plt.close()

torch.save(Reservoir.state_dict(), "Reservoir3.pt")