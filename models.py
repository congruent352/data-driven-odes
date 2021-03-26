#!/usr/bin/env python
# coding: utf-8



'''
Written by Daehan Choi
daehan-choi@uiowa.edu

==============================
          How to use
==============================

1. Define structure & activation function

Example)

layer = [3,                 10, 10,             2]
       input size,   two hidden layers, output size
       (include t)
act = nn.ReLU()


2. Create instance

net = model(layer, act)
    or
net = resnet(layer, act)



3. You can compute the output by following

output = net(x, t)


'''
import torch
import torch.nn as nn
class model(nn.Module):
    def __init__(self, layers, activation, time_lag = 'uniform'):
        super(model, self).__init__()
        sequential_list = []
        layer_list = layers
        if type(activation) != list:
            act = activation
            activation = []
            for i in range(len(layer_list) - 1):
                activation.append(act)
        sequential_list.append(activation[0])
        for i in range(len(layer_list)-2):
            self.c = nn.Linear(layer_list[i], layer_list[i+1])
            nn.init.normal_(self.c.weight, 0, 1)
            nn.init.zeros_(self.c.bias)
            sequential_list.append(self.c)
            sequential_list.append(activation[i+1])
        self.c = nn.Linear(layer_list[-2], layer_list[-1])
        nn.init.normal_(self.c.weight, 0, 1)
        nn.init.zeros_(self.c.bias)
        sequential_list.append(self.c)
        self.net = nn.Sequential(* sequential_list)
        del self.c
        self.type = time_lag

    def forward(self, x_input, dt = 0.1):
        if self.type == 'uniform':
            y = self.net(x_input)
            return y
        else:
            if type(dt)==float:
                dt = torch.Tensor([dt])
                y = torch.cat((x_input, dt), 0)
            elif type(t)!=float:
                dt = torch.Tensor(dt)
                y = torch.cat((x_input, dt), 1)
            y = self.net(y)
            return y
    
class resnet(model):
    def __init__(self, layers, activation, time_lag = 'uniform'):
        super(resnet, self).__init__(layers, activation)
        self.net = model(layers, activation, time_lag)
    def forward(self, x, dt = 0.1):
        return self.net(x, dt) + x

class euler(model):
    def __init__(self, layers, activation):
        super(euler, self).__init__(layers, activation)
        self.net = model(layers, activation)
    def forward(self, x, t, h):
        k1 = self.net(x, t)
        return x + h * k1
    
class rk2(model):
    def __init__(self, layers, activation):
        super(rk2, self).__init__(layers, activation)
        self.net = model(layers, activation)
    def forward(self, x, t, h):
        k1 = self.net(x, t)
        k2 = self.net(x + k1 / 2, t + h/2)
        return x + h * k2

class rk3(model):
    def __init__(self, layers, activation):
        super(rk3, self).__init__(layers, activation)
        self.net = model(layers, activation)
    def forward(self, x, t, h):
        k1 = self.net(x, t)
        k2 = self.net(x + k1 / 2, t + h/2)
        k3 = self.net(x - k1 + 2*k2, t + h)
        return x + h/6 * (k1 + 4*k2 + k3)
class rk4(model):
    def __init__(self, layers, activation):
        super(rk4, self).__init__(layers, activation)
        self.net = model(layers, activation)
    def forward(self, x, t, h):
        k1 = self.net(x, t)
        k2 = self.net(x + k1 / 2, t + h/2)
        k3 = self.net(x + k2 / 2, t + h/2)
        k4 = self.net(x + k3, t + h)
        return x + h/6 * (k1 + 2*k2 + 2*k3 + k4)
class rk5(model):
    def __init__(self, layers, activation):
        super(rk5, self).__init__(layers, activation)
        self.net = model(layers, activation)
    def forward(self, x, t, h):
        k1 = self.net(x, t)
        k2 = self.net(x + k1 / 4, t + h/4)
        k3 = self.net(x + k1 / 8 + k2 / 8, t + h/4)
        k4 = self.net(x - k1 / 2 + k3, t + h/2)
        k5 = self.net(x + 3*k1/16 + 9*k4/16, t + 3*h/4)
        k6 = self.net(x - 3*k1/7 + 2*k2/7 + 12*k3/7 - 12*k4/7 + 8*k5/7, t + h)
        return x+h/90 * (7*k1 + 32*k3 + 12*k4 + 32*k5 + 7*k6)
        return x + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    
class training(nn.Module):
    def __init__(self, net, loss = None):
        super(training, self).__init__()
        self.net = net
        if loss == None:
            def loss(x,y):
                return torch.mean((x-y)**2)
            self.loss = loss
        else:
            self.loss = loss
    def fit(x, y, lr, epochs):
        optimizer = torch.optim.Adam(self.net.parameters(), lr = lr)
        history = []
        def closure():
            optimizer.zero_grad()
            l = self.loss(x, y)
            l.backward()
            return l
        for i in range(epochs):
            optimizer.step(closure)
            with torch.no_grad():
                history.append(loss(x,y).numpy())
        return history

    
# ==============================================


class prediction_():
    def prediction_(net, x0, dt, nb):
        '''
        net: model / x0: initial point / dt: time-lag / nb: number of samples needed
        '''
        if type(x0) == list:
            shape = len(x0)
        else:
            shape = x0.shape[0]
        pred = np.zeros((nb+1, shape))
        pred[0,:] = x0
        for i in range(nb):
            next_point = net(torch.Tensor(pred[i,:]), dt)
            pred[i+1,:] = next_point.detach().numpy()
            del next_point
        return pred
        
 