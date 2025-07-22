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
    def __init__(self, input_size, hidden_size, n_hidden_layers):
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
        #innitializing output_size/2 parameters since we take outputs in pairs
        self.parameterSet = nn.Parameter(torch.randn(1, (int)(output_size/2)))

    def forward(self, x):
        x = self.output_layer(x)
        return x

#creating a format to feed in points to the network with
class colocPoints(Dataset):

  def __init__(self, X):
    if not torch.is_tensor(X) :
      self.X = torch.tensor(X,requires_grad=True).float()
      #reshaping to a nx2 matrix since that is what our network expects as input
    else:
      self.X = X

  def __len__(self):
      return len(self.X)

  def __getitem__(self, i):
      return {"X": self.X[i]}

def SchrodingerEqnResidualLoss(ResOut, ResFirstTimeDer,ResSecondXDir,xvals, outmodel,evenmask,oddmask):
    xvals = xvals.reshape(-1,1)

    W = outmodel.output_layer.weight  # shape: [2D, H]
    b = outmodel.output_layer.bias    # shape: [2D]

    u = (ResOut @  W.T) + b # shape [N, 2D]

    # du/dt = W @ dR/dt
    duBydt = (W @ ResFirstTimeDer.T).T  # shape [N, 2D]

    # d^2u/dx^2 = W @ d^2R/dx^2
    d2uBydx2 = (W @ ResSecondXDir.T).T  # shape [N, 2D]

    Re = u[:,evenmask] # shape [N, D]
    dReBydt = duBydt[:,evenmask]
    d2ReBydx2 = d2uBydx2[:,evenmask] # shape [N, D]

    Im = u[:,oddmask] # shape [N, D]
    dImBydt = duBydt[:,oddmask]
    d2ImBydx2 = d2uBydx2[:,oddmask] # shape [N, D] 

    #V = Coeffs*(Re**2 + Im**2)
    #V = (Re**2 + Im**2)
    V = (outmodel.parameterSet/2)*(xvals**2)

    #residual = torch.mean(torch.square(-d2ImBydx2/2 - V*Re - dReBydt) + torch.square(d2ReBydx2/2 - V*Im - dImBydt))
    #residual = torch.mean(torch.square(d2ImBydx2/2 - V*Im + dReBydt) + torch.square(d2ReBydx2/2 - V*Re - dImBydt))
    residual = torch.mean(torch.square(dReBydt + d2ImBydx2/2 - V*Im) + torch.square(dImBydt - d2ReBydx2/2 + V*Re))
    return residual

def trainFullNetworkWithPrecomputing(Reservoir,data,temporalNormalization,spacialNormalization,outmodel,numoutputs,ICs,LeftBoundary,rightBoundary,colocationPoints,ODEWeight,ICWeight,BCWeight,DataWeight,numEpochs,loss_fn,lr,averageLossOverTime,device,verbose = False):

    tEqualsZeroMask = torch.isclose(colocationPoints[:, 0], torch.tensor(0.0, dtype=colocationPoints.dtype)).to(device)

    initialODEWeight = ODEWeight
    ODEWeightEpoch = 1000

    #initialising optimise
    OutputOptimizer = optim.Adam(outmodel.parameters(), lr=lr)

    scheduler = scheduler = lr_scheduler.LinearLR(OutputOptimizer, start_factor=1.0, end_factor=0.5, total_iters=1)

    scalingfactor = torch.tensor([[temporalNormalization,spacialNormalization]],requires_grad=False).to(device)

    dataIn = torch.tensor(data[0][:,0:2].real,dtype=torch.float32).to(device)
    dataIn = dataIn * scalingfactor
    expectedRe = []
    expectedIm = []
    for i in range(numoutputs):
        expectedRe.append(torch.tensor(data[i][:,2].real,dtype=torch.float32).to(device))
        expectedIm.append(torch.tensor(data[i][:,2].imag,dtype=torch.float32).to(device))
    
    data = colocPoints(colocationPoints[~tEqualsZeroMask,:])

    #initializing the dataloader
    dataloader = DataLoader(data, batch_size=5000, shuffle=False)

    #precalculating reservoir output for data and colocation points
    resDataOut = Reservoir(dataIn)
    resOutput = []
    firstTDerivatives = []
    firstXDerivatives = []
    secondXDerivatives = []

    for i, batchdata in enumerate(dataloader):

        colocs = batchdata["X"].to(device).detach().clone().requires_grad_(True)

        #scaling network input
        networkInput = colocs*scalingfactor

        ResOutOverEvaluationPoints = Reservoir(networkInput)
        resOutput.append(ResOutOverEvaluationPoints)

        #creating lists to store the derivatives in the batch
        firstTDerivatives_i = []
        firstXDerivatives_i = []
        secondXDerivatives_i = []
        
        # colocationPoints[:,0] = t, colocationPoints[:,1] = x
        for i in range(ResOutOverEvaluationPoints.shape[1]):
            resOut = ResOutOverEvaluationPoints[:, i]  # shape: [N]
            grad1 = torch.autograd.grad(
                resOut, colocs,
                grad_outputs=torch.ones_like(resOut),
                create_graph=True, retain_graph=True
            )[0]  # [N, input_dim]
            grad2 = torch.autograd.grad(
                grad1[:,1], colocs,
                grad_outputs=torch.ones_like(grad1[:,1]),
                create_graph=True, retain_graph=True
            )[0]  # [N, input_dim]

            #this is equivilant to taking grad1[:,0].reshape(-1,1) withough having to call reshape
            firstTDerivatives_i.append(grad1[:,0:1])
            firstXDerivatives_i.append(grad1[:,1:2])
            secondXDerivatives_i.append(grad2[:,1:2])

        #if the first batch has just been completed, initialize the derivatives   
        if len(firstTDerivatives) == 0:
            firstTDerivatives = firstTDerivatives_i
            firstXDerivatives = firstXDerivatives_i
            secondXDerivatives = secondXDerivatives_i
        #otherwise append the new calculated values to the existilg entries
        else:
            for i in range(len(firstTDerivatives)):
                firstTDerivatives[i] = torch.cat((firstTDerivatives[i],firstTDerivatives_i[i]),dim=0)
                firstXDerivatives[i] = torch.cat((firstXDerivatives[i],firstXDerivatives_i[i]),dim=0)
                secondXDerivatives[i] = torch.cat((secondXDerivatives[i],secondXDerivatives_i[i]),dim=0)

    ReservoirFirstTDerivative = torch.cat(firstTDerivatives, dim=1).to(device) # shape: [N, 2D]
    ReservoirFirstXDerivative = torch.cat(firstXDerivatives, dim=1).to(device) # [N, 2D]
    ReservoirSecondXDerivative = torch.cat(secondXDerivatives, dim=1).to(device) # [N, 2D]

    resOutput = torch.cat(resOutput,dim=0)

    ReservoirFirstTDerivative = ReservoirFirstTDerivative.detach()
    ReservoirFirstXDerivative = ReservoirFirstXDerivative.detach()
    ReservoirSecondXDerivative = ReservoirSecondXDerivative.detach()
    resOutput = resOutput.detach()
    resDataOut = resDataOut.detach()

    for epoch in range(numEpochs):
        OutputOptimizer.zero_grad()

        dataOut = outmodel(resDataOut)
        DataLoss = 0
        for i in range(numoutputs):
            DataLoss += loss_fn(dataOut[:,2*i],expectedRe[i])
            DataLoss += loss_fn(dataOut[:,2*i + 1],expectedIm[i])
        #getting the average data loss
        DataLoss /= 2*numoutputs

        ICout = outmodel(Reservoir(colocationPoints[tEqualsZeroMask,:]))
        ICLoss = loss_fn(ICout,ICs)

        #left boundry mask: When applied to the output, it will return the outputs on the left boundry
        LBMask = torch.isclose(colocationPoints[~tEqualsZeroMask, 1], torch.tensor(LeftBoundary, dtype=colocs.dtype)).to(device)
        #right boundry mask: When applied to the output, it will return the outputs on the right boundry
        RBMask = torch.isclose(colocationPoints[~tEqualsZeroMask, 1], torch.tensor(rightBoundary, dtype=colocs.dtype)).to(device)

        '''                note: In order for the outputs after the application of the masks to corrispond to the same times,
        colocationPoints must be orginised as follows:

        colocationPoints = [[t1,x1],[t1,x2],[t1,x3],[t1,x4],[t1,x5],[t1,x6],...,[t1,xn],
                            [t2,x1],[t2,x2],[t2,x3],[t2,x4],[t2,x5],[t2,x6],...,[t2,xn], ...]

        This yields:
        colocationPoints[LBMask] = [ [t1,x1], [t2,x1], [t3,x1], ...]
        colocationPoints[RBMask] = [ [t1,xn], [t2,xn], [t3,xn], ...]
        '''
        evenmask = torch.tensor([(i % 2 == 0) for i in range(2*numoutputs)]).to(device)
        oddmask = torch.tensor([(i % 2 != 0) for i in range(2*numoutputs)]).to(device)

        W = outmodel.output_layer.weight  # shape: [2D, H]
        duBydx = (W @ ReservoirFirstXDerivative.T).T  # shape [N, 2D]

        #using evaluation points as colocation points
        ODEloss = SchrodingerEqnResidualLoss(resOutput, ReservoirFirstTDerivative,ReservoirSecondXDerivative,colocationPoints[~tEqualsZeroMask,1], outmodel,evenmask,oddmask)

        output = outmodel(resOutput)
        #enforcing the boundary conditions for t!=0
        zero = torch.zeros(output[LBMask,:].shape).to(device)
        BCLoss = loss_fn(output[LBMask,:],zero) + loss_fn(duBydx[LBMask,:],zero)
        BCLoss += loss_fn(output[RBMask,:],zero) + loss_fn(duBydx[RBMask,:],zero)

        loss = ODEWeight*ODEloss + BCWeight*BCLoss + ICWeight*ICLoss + DataWeight*DataLoss

        #using backpropogation to get the derivitive of the parameters in the network with respect to the loss
        loss.backward()
        #taking a step in the direction of negative slope according to our optimizer
        OutputOptimizer.step() 

        epsilon = 1e-12
        wandb.log({"Log Loss": math.log(loss.item()+epsilon),"Log PDE Loss": math.log(ODEloss+epsilon),"Log IC Loss": math.log(ICLoss+epsilon),"Log BC Loss": math.log(BCLoss+epsilon),"Log Data Loss": math.log(DataLoss+epsilon)})

        averageLossOverTime.append(loss.item())

        if verbose:
            if epoch % 1000 == 0:
                print(f"-------------------------------\nEpoch {epoch}")
                print(f"Average Loss: {loss.item()}")
                print(f"ODEL = {ODEloss}")
                print(f"ICL = {ICLoss}")
                print(f"BCL = {BCLoss}")
        
        if (epoch) == 5000:
                scheduler.step()

    print(f"Average Loss: {loss.item()}")

    return (Reservoir, averageLossOverTime)

@contextmanager
def timer(name="Block"):
    start = perf_counter()
    yield
    end = perf_counter()
    print(f"[{name}] Elapsed time: {end - start:.6f} seconds")

noiseLevels = [1.0,5.0,10.0,20.0,30.0]
dataFolders = ["data_10000DPts","data_1000DPts","data_100DPts","data_10DPts"]
for PNoise in noiseLevels:
    print(f"% Noise = {PNoise}")
    for folder in dataFolders:
        print("Folder = " + folder)
        #print("Offline training Start!")
        with timer("Training Loop"):
            torch.manual_seed(42)
            #assigning device as done in tutorial
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            data = []
            maxK = 5
            trueParameters = [[]]
            numpointsperunit = 3
            percentNoise = PNoise
            print("training files used:")

            '''#validation set
            for i in range(0,maxK*numpointsperunit+1,numpointsperunit):
                #getting all of the non-whole number k values
                data_i = np.load(f"data/dataKis{(i)/numpointsperunit}.npy", allow_pickle=True)
                data.append(data_i)
                trueParameters[0].append((i)/numpointsperunit)
                print(f"data/dataKis{(i)/numpointsperunit}.npy")
            '''#Test set
            for i in range(0,maxK*numpointsperunit,numpointsperunit):
                #getting all of the non-whole number k values
                for j in range(1,numpointsperunit):
                    data_i = np.load(folder + f"/dataKis{(i+j)/numpointsperunit}.npy", allow_pickle=True)
                    #generating percent perterbation for each data point in a uniform distribution over [1-(%noise/100),1+(%noise/100)]
                    realPerterbation  = np.ones(data_i[:,2].shape) + ((np.random.random(data_i[:,2].shape) - 0.5)*percentNoise*(2/100))
                    imaginaryPerterbation  = np.ones(data_i[:,2].shape) + ((np.random.random(data_i[:,2].shape) - 0.5)*percentNoise*(2/100))
                    #print((data_i[:,2] - (data_i[:,2].real*realPerterbation + data_i[:,2].imag*imaginaryPerterbation*1j))/data_i[:,2])
                    data_i[:,2] = data_i[:,2].real*realPerterbation + data_i[:,2].imag*imaginaryPerterbation*1j
                    data.append(data_i)
                    trueParameters[0].append((i+j)/numpointsperunit)
            #print(1/0)
            numParams = len(trueParameters[0])
            trueParameters = torch.tensor(trueParameters)

            LeftBoundary = -5
            RightBoundary = 5
            domainDiameter = RightBoundary - LeftBoundary
            domainCenter = (LeftBoundary + RightBoundary)/2
            timeRange = 1.5
            numxvals = 100
            numtvals = 100

            # Generate evenly spaced points in time and space
            tvals = np.linspace(0, timeRange, numtvals)
            xvals = np.linspace(LeftBoundary, RightBoundary, numxvals)

            colocationPoints = [[t, x] for t in tvals for x in xvals]

            colocationPoints = torch.tensor(colocationPoints,dtype=torch.float32,requires_grad=True).to(device)

            resWidth = 100
            resDepth = 5
            #input and hidden layers
            Reservoir = MLPWithoutOutput(2,resWidth,resDepth).to(device)

            #loading pretrained reservoir
            Reservoir.load_state_dict(torch.load("Reservoir6.pt",map_location=device,weights_only=True))
            Reservoir.eval()

            nummodels = numParams#maxK*numpointsperunit + 1

            domainDiameter = (RightBoundary - LeftBoundary)/2

            kx    = 0.1                        # wave number
            m     = 1                          # mass
            sigma = 0.5                   # width of initial gaussian wave-packet

            A = 1.0 / (sigma * math.sqrt(torch.pi)) # normalization constant

            ICs = torch.zeros((numxvals,2*nummodels),dtype=torch.float32).to(device)

            #setting initial condiitons to u(0,x) = 2sech(x + offset) as done in [Raissi et al. 2019]
            #xvals = torch.tensor(xvals).reshape(-1).to(device)
            xvals = xvals.reshape(-1)
            #offset = ((torch.rand((nummodels))-0.5)*domainDiameter + domainCenter).to(device)
            offset = np.ones((nummodels))*-2
            for i in range(nummodels):
                #ICs[:,2*i] = 2*(torch.cosh(xvals + offset[i]).pow(-1))
                IC = math.sqrt(A) * np.exp(-(xvals-offset[i])**2 / (2.0 * sigma**2)) * np.exp(1j * kx * xvals)
                ICs[:,2*i] = torch.tensor(IC.real).to(device)
                ICs[:,2*i + 1] = torch.tensor(IC.imag).to(device)

            #initilizing loss function
            loss_fn = nn.MSELoss().to(device)

            #setting the number of training epochs
            trainingEpochs = 6000
            trainlr = 5e-3

            averageLossOverTime = []

            ODEWeight = 1.5e-2
            ICWeight = 1e-2#10
            BCWeight = 1e-2#1
            DataWeight = 1

            wandb.init(
            # Set the project where this run will be logged
            project="Time Dependent Schrodinger Equation (Online Training)",
            # Track hyperparameters and run metadata
            config={
            "learning_rate": trainlr,
            "epochs": trainingEpochs,
            "ODE_weight": ODEWeight,
            "resWidth": resWidth,
            "num_models": nummodels,
            "resDepth": resDepth
            })
            tscale = 1
            xscale = 1

            outmodel = MLPOutput(resWidth,2*nummodels).to(device)
            #Reservoir, averageLossOverTime = trainFullNetworkWithPrecomputing(Reservoir,data,(1/timeRange),(1/RightBoundary),outmodel,nummodels,ICs,LeftBoundary,RightBoundary,colocationPoints,ODEWeight,ICWeight,BCWeight,DataWeight,trainingEpochs,loss_fn,trainlr,averageLossOverTime,device,verbose=False)
            Reservoir, averageLossOverTime = trainFullNetworkWithPrecomputing(Reservoir,data,(tscale),(xscale),outmodel,nummodels,ICs,LeftBoundary,RightBoundary,colocationPoints,ODEWeight,ICWeight,BCWeight,DataWeight,trainingEpochs,loss_fn,trainlr,averageLossOverTime,device,verbose=False)
            wandb.finish()

        #print(outmodel.parameterSet)
        parameterLoss = loss_fn(outmodel.parameterSet,trueParameters)
        meanAbsoluteParameterError = torch.mean(torch.abs(trueParameters - outmodel.parameterSet))
        print(f"MSE = {parameterLoss.item()} | MAE = {meanAbsoluteParameterError}")

'''
logloss = torch.log(torch.tensor(averageLossOverTime))
plt.plot(logloss.detach().numpy())
plt.xlabel('Iteration')
plt.ylabel('loss')
plt.title('Training Log Average Loss Of The 30 Readout Layers')
plt.grid(True)
plt.savefig('Training Log Average Loss Of The 30 Readout Layers.png')
#plt.show()

plt.close()

# Plotting code below generated by chatGPT

# Ensure tvals is a tensor
tvals = torch.tensor(tvals, dtype=torch.float32)

# Recreate the grid of test points
test_points = torch.tensor([[t.item(), x.item()] for t in tvals for x in xvals],
                           dtype=torch.float32).to(device)

scalingfactor = torch.tensor([[(tscale),(xscale)]],requires_grad=False).to(device)

# Evaluate the network
with torch.no_grad():
    res_output = Reservoir(test_points*scalingfactor)
    prediction = outmodel(res_output)  # shape: [T*X, 2*nummodels]

# Convert to numpy and reshape
prediction = prediction.cpu().numpy()  # shape: [T*X, 2*nummodels]

num_rows = int(np.ceil(nummodels))
fig, axes = plt.subplots(num_rows, 3, figsize=(12, 4 * num_rows))

if num_rows == 1:
    axes = [axes]  # make iterable


for i in range(nummodels):
    real = prediction[:, 2 * i].reshape(len(tvals), len(xvals))
    imag = prediction[:, 2 * i + 1].reshape(len(tvals), len(xvals))
    magnitude = np.sqrt(real**2 + imag**2)

    ax_real, ax_imag, ax_mag = axes[i]

    im0 = ax_real.imshow(real, extent=[xvals[0], xvals[-1], tvals[-1], tvals[0]],
                         aspect='auto', cmap='RdBu_r')
    ax_real.set_title(f"Model {i+1}: Re(u)")
    fig.colorbar(im0, ax=ax_real)

    im1 = ax_imag.imshow(imag, extent=[xvals[0], xvals[-1], tvals[-1], tvals[0]],
                         aspect='auto', cmap='RdBu_r')
    ax_imag.set_title(f"Model {i+1}: Im(u)")
    fig.colorbar(im1, ax=ax_imag)

    im2 = ax_mag.imshow(magnitude, extent=[xvals[0], xvals[-1], tvals[-1], tvals[0]],
                        aspect='auto', cmap='viridis')
    ax_mag.set_title(f"Model {i+1}: |u|")
    fig.colorbar(im2, ax=ax_mag)

    for ax in (ax_real, ax_imag, ax_mag):
        ax.set_xlabel("x")
        ax.set_ylabel("t")

plt.tight_layout()
plt.savefig('Predicted psi(t,x).png')

plt.close()

plt.plot(xvals, prediction[:len(xvals), 2 * i], label="Re(u) at t=0")
plt.plot(xvals, ICs[:, 2 * i].cpu().numpy(), label="IC")
plt.legend()
plt.title("Initial Condition vs Model Output at t=0")
plt.grid(True)
plt.savefig('insanity check.png')
plt.close()
'''