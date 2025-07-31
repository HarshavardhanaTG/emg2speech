"""  Code taken from 
Jeong, Seungwoo, et al. "Deep efficient continuous manifold learning for time series modeling." 
IEEE Transactions on Pattern Analysis and Machine Intelligence 46.1 (2023): 171-184."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

from torchdiffeq import odeint_adjoint as odeint

class spdRnnNet(nn.Module):
    def __init__(self, classifications, hidden, ODE, device):
        super(spdRnnNet, self).__init__()
        self.classifications = classifications
        self.hidden = hidden
        self.ODE = ODE
        self.device = device


        self.toLinear = (self.hidden * (self.hidden - 1)//2) + self.hidden
        self.RNN = rnnNet(self.classifications, 31, self.hidden, self.ODE, self.device)
    
        self.cls = nn.Sequential(nn.Linear(self.toLinear * 2, self.classifications))

    def forward(self, x):
        
        x = self.RNN(x)
        x = self.cls(x)
        x = torch.nn.functional.log_softmax(x, dim = -1)
        return x
    
class rnnNet(nn.Module):
    def __init__(self, classifications, latentUnits, hiddenUnits, ODE, device = 'cuda:0'):
        super(rnnNet, self).__init__()
        self.ODE = ODE
        self.latents = latentUnits

        self.diagUnits = self.latents
        self.lowTriag = self.latents * (self.latents - 1) // 2

        self.diagHidden = hiddenUnits
        self.lowHidden = hiddenUnits * (hiddenUnits - 1) // 2

        self.classifications = classifications
        
        self.nLayers = 1
        
        self.rgruD = manifoldGRUCell(self.diagUnits, self.diagHidden, True)
        self.rgruL = manifoldGRUCell(self.lowTriag, self.lowHidden, False)

        self.rgruDRe = manifoldGRUCell(self.diagUnits, self.diagHidden, True)
        self.rgruLRe = manifoldGRUCell(self.lowTriag, self.lowHidden, False)
            

        self.odefunc = ODEFunc(nInputs = self.diagHidden + self.lowHidden, nLayers = self.nLayers, nUnits = (self.diagHidden + self.lowHidden))
        self.odefuncRe = ODEFunc(nInputs = self.diagHidden  + self.lowHidden, nLayers = self.nLayers, nUnits = (self.diagHidden + self.lowHidden))


        self.softplus = nn.Softplus()

        self.device = device

    def forward(self, x):
        b, s, _ , _ = x.shape
        xD, xL = self.cholDe(x)
        times = torch.from_numpy(np.arange(s + 1)).float().to(self.device)

        hD = torch.ones(x.shape[0], self.diagHidden, device = self.device)
        hL = torch.zeros(x.shape[0], self.lowHidden, device = self.device)

        hDRe = torch.ones(x.shape[0], self.diagHidden, device = self.device)
        hLRe = torch.zeros(x.shape[0], self.lowHidden, device = self.device)

        out = []
        outRe = []
        
        for t in range(x.shape[1]):

            if self.ODE:
                hp = odeint(self.odefunc, torch.cat((hD.log(), hL), dim = 1), times[t:t + 2], rtol = 1e-4, atol = 1e-5, method = 'euler')[1]
                hD = hp[:, :self.diagHidden].tanh().exp()
                hL = hp[:, self.diagHidden:]

                hpRe = odeint(self.odefuncRe, torch.cat((hDRe.log(), hLRe), dim = 1), times[t:t + 2], rtol = 1e-4, atol = 1e-5, method = 'euler')[1]
                hDRe = hpRe[:, :self.diagHidden].tanh().exp()
                hLRe = hpRe[:, self.diagHidden:]

            
            hD = self.rgruD(xD[:, t, :], hD)
            hL = self.rgruL(xL[:, t, :], hL)

            hDRe = self.rgruDRe(xD[:, x.shape[1] - t - 1, :], hDRe)
            hLRe = self.rgruLRe(xL[:, x.shape[1] - t - 1, :], hLRe)

            out.append(torch.cat((hD.log(), hL), dim = 1))
            outRe.append(torch.cat((hDRe.log(), hLRe), dim = 1))

        out = torch.stack(out)     
        outRe = torch.stack(outRe)   
        outRe = torch.flip(outRe, [0]) 
        h = torch.cat((out, outRe), dim = -1)

        return h

        


    def cholDe(self, x):
        b, s, n, n = x.shape
        x = x.reshape(-1, n, n)
        L = torch.linalg.cholesky(x)
        d = x.new_zeros(b * s, n)
        l = x.new_zeros(b * s, n * (n - 1) // 2)
        for i in range(b * s):
            d[i] = L[i].diag()
            l[i] = torch.cat([L[i][j: j + 1, :j] for j in range(1, n)], dim = 1)[0]
        return d.reshape(b, s, -1), l.reshape(b, s, -1)   



class manifoldGRUCell(nn.Module):

    def __init__(self, inputSize, hiddenSize, diag):
        super(manifoldGRUCell, self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.diag = diag
        if diag:
            layer = PosLinear
            self.nonlinear = nn.Softplus()
        else:
            layer = nn.Linear
            self.nonlinear = nn.Tanh()
        self.x2h = layer(inputSize, 3 * hiddenSize, bias = False)
        self.h2h = layer(hiddenSize, 3 * hiddenSize, bias = False)
        self.bias = nn.Parameter(torch.rand(3 * hiddenSize))
        self.resetParameters()

    def resetParameters(self):
        std = 1.0 / math.sqrt(self.hiddenSize)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        x = x.view(-1, x.size(1))
        gateX = self.x2h(x)
        gateH = self.h2h(hidden)

        gateX = gateX.squeeze()
        gateH = gateH.squeeze()

        wR, wZ, wH = gateX.chunk(3, 1)
        uR, uZ, uH = gateH.chunk(3, 1)
        bR, bZ, bH = self.bias.chunk(3, 0)

        if self.diag:

            zGate = (bZ.abs() * (wZ.log() + uZ.log()).exp()).sigmoid()
            rGate = (bR.abs() * (wR.log() + uR.log()).exp()).sigmoid()          
            hTilde = self.nonlinear((bH.abs() * (wH.log() + (rGate * uH).log()).exp()))
            ht = ((1 - zGate) * hidden.log() + hTilde.log() * (zGate)).exp()
        else:
            zGate = (wZ + uZ + bZ).sigmoid()
            rGate = (wR + uR + bR).sigmoid()
            hTilde = self.nonlinear(wH + (rGate * uH) + bH)
            ht = (1 - zGate) * hidden + zGate * hTilde

        return ht

class PosLinear(nn.Module):
    def __init__(self, inDim, outDim, bias = False):
        super(PosLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn((inDim, outDim)))

    def forward(self, x):
        return torch.matmul(x, torch.abs(self.weight))


    
class ODEFunc(nn.Module):
    def __init__(self, nInputs, nLayers, nUnits):
        super(ODEFunc, self).__init__()
        self.gradientNet = odefunc(nInputs, nLayers, nUnits)

    def forward(self, tLocal, y, backwards = False):
        grad = self.getOdeGradientNN(tLocal, y)
        if backwards:
            grad = -grad
        return grad

    def getOdeGradientNN(self, tLocal, y):
        return self.gradientNet(y)

    def sampleNextPointFromPrior(self, tLocal, y):
        return self.getOdeGradientNN(tLocal, y)

class odefunc(nn.Module):
    def __init__(self, nInputs, nLayers, nUnits):
        super(odefunc, self).__init__()
        self.Layers = nn.ModuleList()
        self.Layers.append(nn.Linear(nInputs, nUnits))
        self.Layers.append(nn.Linear(nInputs, nUnits))
        for i in range(nLayers):
            self.Layers.append(nn.Sequential(nn.Tanh(), nn.Linear(nUnits, nUnits)))
        self.Layers.append(nn.Tanh())
        self.Layers.append(nn.Linear(nUnits, nInputs))

    def forward(self, x):
        for layer in self.Layers:
            x = layer(x)
        return x