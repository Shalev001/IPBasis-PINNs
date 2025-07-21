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

    coefficients1 = torch.rand((1, numoutputs)) * 1.5
    coefficients2 = torch.rand((1, numoutputs)) * 1.5
    forcing = torch.rand((1, numoutputs)) * 3 - 1.5

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

    #initialising opdimise
    FullPINNOptimizer = optim.Adam(list(Reservoir.parameters()) + list(outmodel.parameters()), lr=lr)

    #scheduler = scheduler = lr_scheduler.LinearLR(FullPINNOptimizer, start_factor=1.0, end_factor=0.05, total_iters=1000)
    
    loss = torch.ones(1)*99999999

    zero = torch.zeros((1,1),dtype=torch.float32,requires_grad=True)

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

        if verbose:
            if epoch % 1000 == 0:
                print(f"-------------------------------\nEpoch {epoch}")
                print(f"Average Loss: {loss.item()}")
                print(f"ODEL = {ODEloss}")
                print(f"ICL = {ICloss}")
                
        #using backpropogation to get the derivitive of the parameters in the network with respect to the loss
        loss.backward()
        #taking a step in the direction of negative slope according to our optimiser
        FullPINNOptimizer.step() 
        #scheduler.step()

    print(f"Average Loss: {loss.item()}")

    return (Reservoir, averageLossOverTime)

def newPdeResidualLossForInverseProblem(ResOut, ResFirstDeriv, ResSecondDeriv,Forcings, outmodel):
    W = outmodel.output_layer.weight  # shape: [D, H]
    b = outmodel.output_layer.bias    # shape: [D]

    FirstOrderCoeffs = outmodel.parameterSet1 # shape: [1, D]
    ZeroOrderCoeffs = outmodel.parameterSet2 # shape: [1, D]

    u = (ResOut @  W.T) + b # shape [N, D]
    #u += b

    # du/dt = W @ dR/dt
    du = (W @ ResFirstDeriv.T).T  # shape [N, D]

    # d^2u/dt^2 = W @ d^2R/dt^2
    d2u = (W @ ResSecondDeriv.T).T  # shape [N, D]

    residual = d2u + FirstOrderCoeffs * du + ZeroOrderCoeffs * u - Forcings # shape [N, D]
    return torch.mean(residual.pow(2))

def trainOutput(Reservoir,outmodel,data,dataTimes,forcings,ICs,colocationPoints,ODEWeight,numEpochs,loss_fn,lr,averageLossOverTime,device,verbose = False):
    #initialising opdimise
    outputOptimizer = optim.Adam(outmodel.parameters(), lr=lr)

    #scheduler = scheduler = lr_scheduler.LinearLR(FullPINNOptimizer, start_factor=1.0, end_factor=0.05, total_iters=1000)
    
    loss = torch.ones(1)*99999999

    zero = torch.zeros((1,1),dtype=torch.float32,requires_grad=True)

    #We compute the output of the reservoir ahead of time since it stays constant throughout training and would thus be a waste to recompute every epoch
    resZero = Reservoir(zero).detach()

    ResOutOverEvaluationPoints = Reservoir(colocationPoints)
    ResOutOverDataPoints = Reservoir(dataTimes).detach()
        
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

    ResOutOverEvaluationPoints = ResOutOverEvaluationPoints.detach()
    ReservoirFirstDerivative = ReservoirFirstDerivative.detach()
    ReservoirSecondDerivative = ReservoirSecondDerivative.detach()

    W = outmodel.output_layer.weight

    for epoch in range(numEpochs):

        outputOptimizer.zero_grad()

        #using evaluation points as colocation points
        ODEloss = newPdeResidualLossForInverseProblem(ResOutOverEvaluationPoints, ReservoirFirstDerivative,ReservoirSecondDerivative,forcings,outmodel)

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
        #scheduler.step()

    print(f"Average Loss: {loss.item()}")

    return (outmodel, averageLossOverTime)

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

    nummodels = 10

    diameter = 10
    center = 0
    ICs = torch.rand((nummodels,2),dtype=torch.float32)*diameter - (diameter/2) + center

    final = 3
    initial = 0
    numevals = 30
    colocationPoints = torch.linspace(initial,final,numevals,dtype=torch.float32,requires_grad=True).reshape(-1,1)

    #input and hidden layers
    Reservoir = MLPWithoutOutput(1,resWidth,1,4).to(device)

    #initilizing loss function
    loss_fn = nn.MSELoss().to(device)

    #setting the number of training epochs
    trainingEpochs = 10000
    trainlr = 1e-4


    averageLossOverTime = []

    target_error = 0.01
    uniform_error = 999999999

    i = -1

    ODEWeight = 1

    outmodel = MLPOutput(resWidth,nummodels).to(device)
    Reservoir, averageLossOverTime = trainFullNetworkWithPrecomputing(Reservoir,outmodel,nummodels,ICs,colocationPoints,ODEWeight,trainingEpochs,loss_fn,trainlr,averageLossOverTime,device,verbose=True)

logloss = torch.log(torch.tensor(averageLossOverTime))
plt.plot(logloss.detach().numpy())
plt.xlabel('Iteration')
plt.ylabel('loss')
plt.title('Training Log Average Loss Of The 30 Readout Layers')
plt.grid(True)
plt.savefig('Training Log Average Loss Of The 30 Readout Layers.png')

plt.close()

final = 3
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

torch.save(Reservoir.state_dict(), "Reservoir.pt")



def fprime(t,x,a1,a2,a3):
  return [x[1],-a2*x[0] - a1*x[1] - a3]

print("online training Start!")
with timer("Online Training"):
    torch.manual_seed(43)

    resWidth = 40

    Reservoir = MLPWithoutOutput(1,resWidth,1,4).to(device)

    Reservoir.load_state_dict(torch.load("Reservoir.pt",weights_only=True))
    Reservoir.eval()

    nummodels = 100

    diameter = 10
    center = 0
    ICs = torch.rand((nummodels,2),dtype=torch.float32)*diameter - (diameter/2) + center

    coefficients1 = torch.rand((1, nummodels)) * 1.5
    coefficients2 = torch.rand((1, nummodels)) * 1.5
    forcing = torch.rand((1, nummodels)) * 3 - 1.5
    
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
    trainingEpochs = 2000
    trainlr = 1e-2

    averageLossOverTime = []

    ODEWeight = 1

    outmodel = MLPOutput(resWidth,nummodels).to(device)

    outmodel, averageLossOverTime = trainOutput(Reservoir,outmodel,data,evalpts,forcing,ICs,colocationPoints,ODEWeight,trainingEpochs,loss_fn,trainlr,averageLossOverTime,device,verbose = True)
    print(f"mean squared error on 1st parameters = {torch.mean(torch.square(coefficients1 - outmodel.parameterSet1))}")
    print(f"mean squared error on 2nd parameters = {torch.mean(torch.square(coefficients2 - outmodel.parameterSet2))}")
    print(f"mean squared error on 1st parameters = {torch.mean(torch.square(coefficients2 - outmodel.parameterSet1))}")
    print(f"mean squared error on 2nd parameters = {torch.mean(torch.square(coefficients1 - outmodel.parameterSet2))}")

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
    plt.plot(colocationPoints.detach(), modelOut[:,i], label='modelOut(t)')
#plt.plot(solution.t, modelOut[:,0], label='modelOut(t)')
#plt.plot(solution.t, solution.y[0], label='X(t)')
plt.xlabel('Time')
plt.ylabel('Values')

plt.title('Online Training Solutions of the Differential Equations')
#plt.legend()
plt.grid(True)
plt.savefig('Online Training Solutions of the Differential Equations.png')

plt.close()

final = 3
initial = 0
numevals = 100
evalpts = np.linspace(initial,final,numevals)
torchEval = torch.tensor(evalpts,dtype=torch.float32).reshape(-1,1)

modelOut = outmodel(Reservoir(torchEval)).detach().numpy()
for i in range(3):
    solution_i = solve_ivp(fprime,(initial,final),[ICs[i,0],ICs[i,1]],t_eval=evalpts,args=(coefficients1[0,i],coefficients2[0,i],forcing[0,i]))
    plt.plot(solution_i.t,solution_i.y[0])
    plt.plot(solution_i.t, modelOut[:,i])
    
#plt.plot(solution.t, modelOut[:,0], label='modelOut(t)')
#plt.plot(solution.t, solution.y[0], label='X(t)')
plt.xlabel('Time')
plt.ylabel('Values')

plt.title('Coparisons of Solution of the Differential Equation')
#plt.legend()
plt.grid(True)
plt.savefig('Coparisons of Solution of the Differential Equation.png')

plt.close()