#For code with everything in a single file, see FastInverseProblems.py
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, Dataset
import torch.autograd.forward_ad as fwAD
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

from fomoh.hyperdual import HyperTensor as htorch
from fomoh.nn import DenseModel, nll_loss
from fomoh.nn_models_torch import DenseModel_Torch





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
        self.parameterSet = nn.Parameter(torch.randn(3, output_size))

    def forward(self, x):
        x = self.output_layer(x)
        return x

def PdeResidualLoss(ResOut, ResFirstDeriv, ResSecondDeriv,FirstOrderCoeffs,ZeroOrderCoeffs,Forcings, outmodel):
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

def PdeResidualLossForInverseProblem(ResOut, ResFirstDeriv, ResSecondDeriv, outmodel):
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
    firstDer = y_h.eps1
    secondDer = y_h.eps1eps2

    return (firstDer,secondDer)



def trainFullNetworkWithPrecomputing(Reservoir,HyperTensReservoir,outmodel,numoutputs,ValidationModel,numValidationModels,ICs,valICs,colocationPoints,ODEWeight,numEpochs,loss_fn,lr,averageLossOverTime,averageValidationLossOverTime,device,verbose = False):

    coefficients1 = (torch.rand((1, numoutputs)) * 1.5).to(device)
    coefficients2 = (torch.rand((1, numoutputs)) * 1.5).to(device)
    forcing = (torch.rand((1, numoutputs)) * 3 - 1.5).to(device)
    
    #dateching so that the computational graph does not include these matricies giving us an error for using them in multiple calls to .backwards()
    coefficients1 = coefficients1.detach()
    coefficients2 = coefficients2.detach()
    forcing = forcing.detach()

    valcoefficients1 = (torch.rand((1, numValidationModels)) * 1.5).to(device)
    valcoefficients2 = (torch.rand((1, numValidationModels)) * 1.5).to(device)
    valforcing = (torch.rand((1, numValidationModels)) * 3 - 1.5).to(device)
    
    #dateching so that the computational graph does not include these matricies giving us an error for using them in multiple calls to .backwards()
    valcoefficients1 = valcoefficients1.detach()
    valcoefficients2 = valcoefficients2.detach()
    valforcing = valforcing.detach()

    vallr = 1e-3

    #initialising optimizer
    FullPINNOptimizer = optim.Adam(list(Reservoir.parameters()) + list(outmodel.parameters()), lr=lr)
    #initialising validation set optimizer
    valOptimizer = optim.Adam(ValidationModel.parameters(), lr=vallr)

    #scheduler = scheduler = lr_scheduler.LinearLR(FullPINNOptimizer, start_factor=1.0, end_factor=0.05, total_iters=1000)
    
    loss = (torch.ones(1)*99999999).to(device)
    valloss = (torch.ones(1)*99999999).to(device)

    zero = torch.zeros((1,1),dtype=torch.float32,requires_grad=True).to(device).detach()

    bestRes = copy.deepcopy(Reservoir)
    bestParamMSE = loss.item()
    finalTrainLoss = loss.item()
    finalValLoss = loss.item()
    epochsSinceBesParamMSE = 0

    for epoch in range(numEpochs):

        FullPINNOptimizer.zero_grad()

        zeroOut = outmodel(Reservoir(zero))

        ResOutOverEvaluationPoints = Reservoir(colocationPoints)

        '''
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

        firstDer, secondDer = computeDerivatives(Reservoir,HyperTensReservoir,colocationPoints)
        ReservoirFirstDerivative = firstDer
        ReservoirSecondDerivative = secondDer

        #print(firstDer.requires_grad)

        #print(ReservoirFirstDerivativeNew.shape)
        #print(ReservoirFirstDerivative.shape)
        #print(torch.mean(torch.abs(ReservoirFirstDerivative)))
        #print(torch.mean(torch.abs(firstDer - ReservoirFirstDerivative)))
        
        #using evaluation points as colocation points
        ODEloss = PdeResidualLoss(ResOutOverEvaluationPoints, ReservoirFirstDerivative,ReservoirSecondDerivative,coefficients1,coefficients2,forcing,outmodel)

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

        ReservoirFirstDerivative = ReservoirFirstDerivative.detach() # [N, D]
        ReservoirSecondDerivative = ReservoirSecondDerivative.detach() # [N, D]
        ResOutOverEvaluationPoints = ResOutOverEvaluationPoints.detach()

        valOptimizer.zero_grad()

        valODEloss = PdeResidualLossForInverseProblem(ResOutOverEvaluationPoints, ReservoirFirstDerivative,ReservoirSecondDerivative,ValidationModel)

        valzeroOut = ValidationModel(Reservoir(zero))

        valICloss = loss_fn(valzeroOut,valICs[:,0].reshape(1,-1))

        #du0/dt (assuming the first input is t = 0)
        du0 = ((ValidationModel.output_layer.weight) @ ReservoirFirstDerivative[0,:].reshape(-1,1)).T
        
        valICloss += loss_fn(du0,valICs[:,1].reshape(1, -1))

        valloss = ODEWeight*valODEloss + valICloss

        #averageValidationLossOverTime.append(valloss.item())

        valloss.backward()

        valOptimizer.step()

        paramMSE = loss_fn(ValidationModel.parameterSet[0:1,:], valcoefficients1)
        paramMSE += loss_fn(ValidationModel.parameterSet[1:2,:], valcoefficients2)
        paramMSE = paramMSE/2
        averageValidationLossOverTime.append(paramMSE.item())

        if (paramMSE.item() < bestParamMSE):
            bestRes = copy.deepcopy(Reservoir)
            bestParamMSE = paramMSE.item()
            finalTrainLoss = loss.item()
            finalValLoss = valloss.item()
            epochsSinceBesParamMSE = 0
        else:
            epochsSinceBesParamMSE += 1
            if epochsSinceBesParamMSE >= 1000:
                print(f"Stopped early at epoch {epoch}")
                print(f"Final Train Loss: {finalTrainLoss}")
                print(f"Final Validation Loss: {finalValLoss}")
                print(f"Best Param MSE: {bestParamMSE}")
                return (bestRes, averageLossOverTime,averageValidationLossOverTime)

        if verbose:
            if epoch % 1000 == 0:
                print(f"-------------------------------\nEpoch {epoch}")
                print(f"Loss: {loss.item()}")
                print(f"ODEL = {ODEloss}")
                print(f"ICL = {ICloss}")

    print(f"Final Train Loss: {finalTrainLoss}")
    print(f"Final Validation Loss: {finalValLoss}")
    print(f"Best Param MSE: {bestParamMSE}")
    return (bestRes, averageLossOverTime,averageValidationLossOverTime)

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
    numValidationModels = 10


    diameter = 10
    center = 0
    ICs = (torch.rand((nummodels,2),dtype=torch.float32)*diameter - (diameter/2) + center).to(device)
    valICs = (torch.rand((numValidationModels,2),dtype=torch.float32)*diameter - (diameter/2) + center).to(device)

    final = 3
    initial = 0
    numevals = 30
    colocationPoints = torch.linspace(initial,final,numevals,dtype=torch.float32,requires_grad=True).reshape(-1,1).to(device)

    #input and hidden layers
    #Reservoir = MLPWithoutOutput(1,resWidth,1,4).to(device)
    TorchReservoir = DenseModel_Torch(layers=[1,resWidth,resWidth,resWidth,resWidth],activation=nn.Tanh()).to(device)
    HyperTensReservoir = DenseModel(layers=[1,resWidth,resWidth,resWidth,resWidth])
    HyperTensReservoir.to(device)
    
    #initilizing loss function
    loss_fn = nn.MSELoss().to(device)

    #setting the number of training epochs
    trainingEpochs = 30000
    trainlr = 1e-3

    averageLossOverTime = []
    averageValidationLossOverTime = []

    ODEWeight = 1

    outmodel = MLPOutput(resWidth,nummodels).to(device)
    ValidationModel = MLPOutput(resWidth,numValidationModels).to(device)
    Reservoir, averageLossOverTime, averageValidationLossOverTime = trainFullNetworkWithPrecomputing(TorchReservoir,HyperTensReservoir,outmodel,nummodels,ValidationModel,numValidationModels,ICs,valICs,colocationPoints,ODEWeight,trainingEpochs,loss_fn,trainlr,averageLossOverTime,averageValidationLossOverTime,device,verbose=True)

logloss = torch.log(torch.tensor(averageLossOverTime))
logvalloss = torch.log(torch.tensor(averageValidationLossOverTime))
plt.plot(logloss.detach().numpy(),label="Log Train Loss")
plt.plot(logvalloss.detach().numpy(),label="Log Validation Loss")
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


