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

import wandb

import copy

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

class MLP(nn.Module):
    """
     A simple feedforward neural network with multiple hidden layers.
    """
    def __init__(self, input_size, hidden_size, output_size, n_hidden_layers):
        super(MLP, self).__init__()
        self.input_layer = nn.Sequential(nn.Linear(input_size, hidden_size,dtype=torch.float32), nn.Tanh())
        self.hidden_layers = self._make_hidden_layers(n_hidden_layers, hidden_size)
        self.output_layer = nn.Linear(hidden_size,output_size)

    def _make_hidden_layers(self, n_hidden_layers, hidden_size):
        layers = []
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(hidden_size, hidden_size,dtype=torch.float32), nn.Tanh()]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

def LotkaVolterraResidualLossForInverseProblem(ResOut, ResFirstDeriv,Coeffs, outmodel,evenmask,oddmask,unknownTerms):
    W = outmodel.output_layer.weight  # shape: [2D, H]
    b = outmodel.output_layer.bias    # shape: [2D]

    u = (ResOut @  W.T) + b # shape [N, 2D]

    # du/dt = W @ dR/dt
    du = (W @ ResFirstDeriv.T).T  # shape [N, 2D]

    x = u[:,evenmask] # shape [N,D]
    dx = du[:,evenmask] # shape [N,D]

    y = u[:,oddmask] # shape [N,D]
    dy = du[:,oddmask] # shape [N,D]

    #can this be optimised by somehow removing the for loop? ************************************************************************************
    unknownOuts= []
    for i in range(len(unknownTerms)):
        #using i:i+1 makes each vector take a shape of [N,1] instead of [N] that would be returned by simply using i
        #so that the concatination can be completed
        inp = torch.cat([x[:,i:i+1],y[:,i:i+1]],dim=1) # shape [N,2]
        unknownOuts.append(unknownTerms[i](inp)) # shape [N,2] since each unknown term has two outputs
    
    unknownOuts = torch.cat(unknownOuts,dim=1) # shape [N,2D] 
    unknownOut1 = unknownOuts[:,evenmask] # shape [N,D]
    unknownOut2 = unknownOuts[:,oddmask] # shape [N,D]

    residual = torch.mean(torch.square(Coeffs[0,:]*x - unknownOut1 - dx) + torch.square(-Coeffs[2,:]*y + unknownOut2 - dy))
    return residual

#Method to evaluate network output derivative using forward mode auto-differentiation 
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

    return resOut, firstDer

def trainOutput(Reservoir,HyperTensReservoir,outmodel,data,dataTimes,Coeffs,ICs,colocationPoints,ODEWeight,numEpochs,loss_fn,lr,averageLossOverTime,device,verbose = False):
    #initialising opdimise
    outputOptimizer = optim.Adam(outmodel.parameters(), lr=lr)
    
    loss = torch.ones(1)*99999999

    zero = torch.zeros((1,1),dtype=torch.float32,requires_grad=True)

    #We compute the output of the reservoir ahead of time since it stays constant throughout training and would thus be a waste to recompute every epoch
    resZero = Reservoir(zero).detach()

    ResOutOverDataPoints = Reservoir(dataTimes).detach()
    
    '''
    ResOutOverEvaluationPoints = Reservoir(colocationPoints)
    firstDerivatives = []

    for i in range(ResOutOverEvaluationPoints.shape[1]):
        output = ResOutOverEvaluationPoints[:, i]  # shape: [N]
        grad1 = torch.autograd.grad(
            output, colocationPoints,
            grad_outputs=torch.ones_like(output),
            create_graph=True, retain_graph=True
        )[0]  # [N, input_dim]

        firstDerivatives.append(grad1)

    ReservoirFirstDerivative = torch.cat(firstDerivatives, dim=1) # [N, D]
    '''

    ResOutOverEvaluationPoints, ReservoirFirstDerivative = computeDerivatives(Reservoir,HyperTensReservoir,colocationPoints)
    

    ResOutOverEvaluationPoints = ResOutOverEvaluationPoints.detach()
    ReservoirFirstDerivative = ReservoirFirstDerivative.detach()

    W = outmodel.output_layer.weight

    #initializing the terms representing the unknown components to be used in the loss function
    unknownTerms = []
    parameterList = []
    for i in range((int)(W.shape[0]/2)):
        newModel = MLP(2, 20, 2, 2).to(device)
        unknownTerms.append(newModel)
        parameterList += list(newModel.parameters())

    outputOptimizer = optim.Adam(list(outmodel.parameters()) + parameterList, lr=lr)

    bestUnknownTerms = copy.deepcopy(unknownTerms)
    bestOutmodel = copy.deepcopy(outmodel)
    bestLoss = 9999999

    evenmask = torch.tensor([(i % 2 == 0) for i in range(W.shape[0])])
    oddmask = torch.tensor([(i % 2 != 0) for i in range(W.shape[0])])

    for epoch in range(numEpochs):

        outputOptimizer.zero_grad()

        #using evaluation points as colocation points
        ODEloss = LotkaVolterraResidualLossForInverseProblem(ResOutOverEvaluationPoints, ReservoirFirstDerivative,Coeffs, outmodel,evenmask,oddmask,unknownTerms)

        output = outmodel(ResOutOverDataPoints)
        
        dataloss = loss_fn(output,data)

        loss = ODEWeight*ODEloss + dataloss

        averageLossOverTime.append(loss.item())

        wandb.log({"Log Loss": math.log(loss.item()),"Log ODE Loss": math.log(ODEloss),"Log Data Loss": math.log(dataloss)})

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
        '''if (epoch) == 9000:
            scheduler = lr_scheduler.LinearLR(outputOptimizer, start_factor=1.0, end_factor=0.8, total_iters=1)
            scheduler.step()
        elif epoch > 9000 and scheduler is not None:
            scheduler.step()'''
        if loss.item() < bestLoss:
            bestUnknownTerms = copy.deepcopy(unknownTerms)
            bestOutmodel = copy.deepcopy(outmodel)
            bestLoss = loss.item()

    print(f"Average Loss: {loss.item()}")

    return (bestOutmodel, bestUnknownTerms, averageLossOverTime)

@contextmanager
def timer(name="Block"):
    start = perf_counter()
    yield
    end = perf_counter()
    print(f"[{name}] Elapsed time: {end - start:.6f} seconds")

def fprime(t,x,Coeffs):
  return [Coeffs[0]*x[0]-Coeffs[1]*x[0]*x[1],-Coeffs[2]*x[1] + Coeffs[3]*x[0]*x[1]]

print("online training Start!")
with timer("Online Training"):
    torch.manual_seed(43)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resWidth = 64

    nummodels = 100

    diameter = 2
    center = 1.1
    ICs = torch.rand((nummodels,2),dtype=torch.float32)*diameter - (diameter/2) + center

    coefficients = 0.5 + torch.rand((4, nummodels))
    
    #dateching so that the computational graph does not include these matricies giving us an error for using them in multiple calls to .backwards()
    coefficients = coefficients.detach()

    #Reservoir = MLPWithoutOutput(1,resWidth,1,4).to(device)
    Reservoir = DenseModel_Torch(layers=[1,resWidth,resWidth,resWidth,resWidth],activation=nn.Tanh()).to(device)
    HyperTensReservoir = DenseModel(layers=[1,resWidth,resWidth,resWidth,resWidth])
    HyperTensReservoir.to(device)

    Reservoir.load_state_dict(torch.load("Reservoir4.pt",weights_only=True))
    Reservoir.eval()

    #initilizing loss function
    loss_fn = nn.MSELoss().to(device)

    final = 10
    initial = 0
    numevals = 100
    evalpts = np.linspace(initial,final,numevals)

    solutions = []

    for i in range(nummodels):
        solution_i = solve_ivp(fprime,(initial,final),[ICs[i,0],ICs[i,1]],t_eval=evalpts,args=(coefficients[:,i],))
        solutions.append(torch.tensor(solution_i.y.T,dtype=torch.float32))

    data = torch.cat(solutions,dim=1)

    evalpts = torch.tensor(evalpts,dtype=torch.float32,requires_grad=True).reshape(-1,1)

    final = 10
    initial = 0
    numevals = 100
    colocationPoints = torch.linspace(initial,final,numevals,dtype=torch.float32,requires_grad=True).reshape(-1,1).to(device)

    #setting the number of training epochs
    trainingEpochs = 10000
    trainlr = 3e-3

    averageLossOverTime = []

    ODEWeight = 0.1

    wandb.init(
      # Set the project where this run will be logged
      project="LotkaVolterraUPINN(Online Training)",
      # Track hyperparameters and run metadata
      config={
      "learning_rate": trainlr,
      "epochs": trainingEpochs,
      "ODE_weight": ODEWeight,
      "resWidth": resWidth,
      "number of output models": nummodels
      })

    outmodel = MLPOutput(resWidth,2*nummodels).to(device)

    outmodel, unknownTerms, averageLossOverTime = trainOutput(Reservoir,HyperTensReservoir,outmodel,data,evalpts,coefficients,ICs,colocationPoints,ODEWeight,trainingEpochs,loss_fn,trainlr,averageLossOverTime,device,verbose = False)
    
    wandb.finish()

lossSum = 0
MAESum = 0
#use point from the trajectory instead!
for i in range(len(solutions)):
    pred = unknownTerms[i](solutions[i])
    pred1 = pred[:,0]
    pred2 = pred[:,1]
    trueval1 = coefficients[1,i]*solutions[i][:,0]*solutions[i][:,1]
    trueval2 = coefficients[3,i]*solutions[i][:,0]*solutions[i][:,1]
    loss1 = loss_fn(pred1,trueval1)
    MAE1 = torch.mean(torch.abs(trueval1 - pred1))
    loss2 = loss_fn(pred2, trueval2)
    MAE2 = torch.mean(torch.abs(trueval2 - pred2))
    lossSum += loss1 + loss2
    MAESum += MAE1 + MAE2
    #print(f"Unknown term [{i}][0] loss = {loss1},Unknown term [{i}][1] loss = {loss2} ")
averageUnknownTermLoss = lossSum/(2*len(unknownTerms))
unknownTermMAE = MAESum/(2*len(unknownTerms))

print(f"Average loss of unknown terms over observed data points: {averageUnknownTermLoss}")
print(f"Mean Absolute Diffrence of unknown terms over observed data points: {unknownTermMAE}")

#plt.plot(averageLossOverTime)
logloss = torch.log(torch.tensor(averageLossOverTime))
plt.plot(logloss.detach().numpy())
plt.xlabel('Iteration')
plt.ylabel('loss')
plt.title('Online Training Log Average Loss Of The 30 Readout Layers')
plt.grid(True)
plt.savefig('Online Training Log Average Loss Of The 30 Readout Layers.png')

plt.close()

final = 10
initial = 0
numevals = 1000
evalpts = np.linspace(initial,final,numevals)
torchEval = torch.tensor(evalpts,dtype=torch.float32).reshape(-1,1)

modelOut = outmodel(Reservoir(torchEval)).detach().numpy()
for i in range(min(50,nummodels)):
    solution_i = solve_ivp(fprime,(initial,final),[ICs[i,0],ICs[i,1]],t_eval=evalpts,args=(coefficients[:,i],))
    solutions.append(torch.tensor(solution_i.y.T,dtype=torch.float32))
    #plt.plot(solution_i.y[0],solution_i.y[1])
    plt.plot(modelOut[:,2*i], modelOut[:,2*i + 1])
    
#plt.plot(solution.t, modelOut[:,0], label='modelOut(t)')
#plt.plot(solution.t, solution.y[0], label='X(t)')
plt.xlabel('Time')
plt.ylabel('Values')

plt.title('Online Training Coparisons of Solution of the Differential Equation')
#plt.legend()
plt.grid(True)
plt.savefig('Online Training Coparisons of Solution of the Differential Equation.png')

plt.close()