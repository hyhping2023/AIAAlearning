import torch
from torch import nn
from d2l import torch as d2l
from matplotlib import pyplot as plt

def NET(X):
    return 2*torch.sin(X) + X ** 0.8 

def averageGather(y_hat):
    return y_hat.mean(dim=0)

def nonParamAttention(y_hat, x, x_hat):
    '''
    x is the true input
    x_hat is the train or sample input
    y_hat is the train or sample output
    '''
    x = x.reshape((-1, 1)).expand((x.shape[0], y_hat.shape[0]))
    x_hat = x_hat.expand((x_hat.shape[0], y_hat.shape[0]))
    # print(x, '\n', x_hat)
    up = torch.exp(-0.5*(x - x_hat)**2)
    down = torch.sum(up, dim = 1, keepdim = True)
    s = up / down
    # print(up, '\n', down, '\n', s)
    result = s@y_hat
    # print(result)
    return result

def paramAttention(y_hat, x, x_hat, w):
    '''
    x is the true input
    x_hat is the train or sample input
    y_hat is the train or sample output
    w is the parameter that can be learned
    '''
    x = x.reshape((-1, 1)).expand((x.shape[0], y_hat.shape[0]))
    x_hat = x_hat.expand((x_hat.shape[0], y_hat.shape[0]))
    # print(x, '\n', x_hat)
    up = torch.exp(-0.5*((x - x_hat)*w)**2)
    down = torch.sum(up, dim = 1, keepdim = True)
    s = up / down
    # print(up, '\n', down, '\n', s)
    result = s@y_hat
    # print(result)
    return result

def meanSquareLoss(y, y_hat):
    return torch.mean((y - y_hat) ** 2)

def train(y_hat, x_hat, lr = 50, epoches = 1000):
    w = torch.ones((x_hat.shape[0], x_hat.shape[0]), dtype=torch.float32, requires_grad = True)
    for _ in range(epoches):
        results = paramAttention(y_hat, x_hat, x_hat, w)
        loss = meanSquareLoss(y_hat, results)
        # print(loss)
        loss.sum().backward()
        with torch.no_grad():
            # print(w.grad)
            w -= lr * w.grad
            w.grad.zero_()
    print(loss)
    return w

if __name__ == "__main__":
    n_train = 1000
    ranges = 50.0
    noise = 0.1
    x_hat = torch.sort(torch.rand(n_train, dtype=torch.float32) * ranges)[0]
    X = torch.arange(0., ranges, ranges/n_train, dtype = torch.float32)
    y_train = NET(x_hat) + noise * torch.randn(X.shape, dtype=torch.float32)
    plt.scatter(x_hat, y_train, s=5, c='red', label='Origin')
    plt.plot(X, NET(X), label='True')
    plt.plot(X, averageGather(y_train).expand(y_train.shape), label = 'Average Predict')
    plt.plot(X, nonParamAttention(y_train, X, x_hat).expand(y_train.shape), label = 'Non-Param Predict')
    
    w = train(y_train, x_hat)
    print(w)
    # print(paramAttention(y_train, X, x_hat, w))
    plt.plot(X, paramAttention(y_train, X, x_hat, w).detach(), label = 'Param Predict')
    plt.legend()
    plt.show()